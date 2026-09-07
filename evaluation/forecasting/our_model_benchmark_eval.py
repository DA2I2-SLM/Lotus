#!/usr/bin/env python3
"""
Zero-shot benchmark evaluation of OpenTSLMSPFixPos2D on ETTh1, ETTh2, Weather.

Replicates run_eval.py protocol:
  - StandardScaler normalization (fit on train, apply to test)
  - Channel-independent (UV) sliding window evaluation
  - Autoregressive patch generation (patch_size=32 per decode step)
  - MSE + MAE averaged over all prediction elements

Usage:
    python evaluation/forecasting/our_model_benchmark_eval.py \
        --ckpt results/.../best_model.pt \
        --data_dir /path/to/csvs/ \
        --context_lengths 512 2048 \
        --horizons 96 192 336 720 \
        --n_eval_windows 500 \
        --inner_bs 16
"""

import argparse
import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from opentslm.model.llm.OpenTSLMSPFixPos2D import OpenTSLMSPFixPos2D

PATCH_SIZE = 32
MAX_CTX_TIMESTEPS = 2048   # max timesteps the model was trained on
NORM_STD_FLOOR = 1e-3


# ── Model ──────────────────────────────────────────────────────────────────────

def build_model_args():
    return SimpleNamespace(
        is_pretraining=True,
        patch_size=PATCH_SIZE,
        enc_dec_type="transformer",
        trans_enc_dim=192,
        trans_enc_num_layers=6,
        trans_enc_num_heads=8,
        trans_enc_ff_dim=896,
        trans_dec_dim=192,
        trans_dec_num_layers=4,
        trans_dec_num_heads=8,
        trans_dec_ff_dim=896,
        use_freq_embed=True,
        llm_mv_position_ids=True,
        llm_dtype="auto",
        gradient_checkpointing=False,
        mlp_enc_hidden_dim=None,
        mlp_enc_ff_dim=None,
        mlp_dec_hidden_dim=None,
        mlp_dec_ff_dim=None,
        use_only_time_pe=False,
        use_only_channel_pe=False,
    )


# ── Data loading ───────────────────────────────────────────────────────────────

def load_test_windows(csv_path, context_length, horizon, n_eval_windows=None, seed=42):
    """
    Load StandardScaler-normalized test windows using BenchmarkEvalDataset splits.
    Returns: (inputs, labels) as float32 arrays [N_windows, C, L] and [N_windows, C, H].
    """
    df = pd.read_csv(csv_path)
    base = os.path.basename(csv_path).lower()

    if "etth" in base:
        border1s = [0,
                    12*30*24 - context_length,
                    12*30*24 + 4*30*24 - context_length]
        border2s = [12*30*24,
                    12*30*24 + 4*30*24,
                    12*30*24 + 8*30*24]
    elif "ettm" in base:
        border1s = [0,
                    12*30*24*4 - context_length,
                    12*30*24*4 + 4*30*24*4 - context_length]
        border2s = [12*30*24*4,
                    12*30*24*4 + 4*30*24*4,
                    12*30*24*4 + 8*30*24*4]
    else:
        n_total = len(df)
        num_train = int(n_total * 0.7)
        num_test = int(n_total * 0.2)
        num_vali = n_total - num_train - num_test
        border1s = [0,
                    num_train - context_length,
                    n_total - num_test - context_length]
        border2s = [num_train,
                    num_train + num_vali,
                    n_total]

    cols = df.columns[1:]  # skip 'date'
    vals = df[cols].values.astype(np.float32)

    train_data = vals[border1s[0]:border2s[0]]
    test_data = vals[border1s[2]:border2s[2]]

    scaler = StandardScaler()
    scaler.fit(train_data)
    scaled_test = scaler.transform(test_data)  # [T_test, C]

    C = scaled_test.shape[1]
    window_len = context_length + horizon
    T_test = scaled_test.shape[0]
    data_ch = scaled_test.T  # [C, T_test] — channel-major for easy indexing

    # Sliding window with step=1 (matches run_eval.py: offset_i in range(window_len, n_points)
    # where n_points = T_test — last valid offset is T_test-1, same as BenchmarkEvalDataset)
    all_offsets = list(range(window_len, T_test))
    if not all_offsets:
        raise ValueError(
            f"No valid windows: T_test={T_test} < window_len={window_len} "
            f"for context={context_length}, horizon={horizon}"
        )

    if n_eval_windows is not None and n_eval_windows < len(all_offsets):
        rng = np.random.default_rng(seed)
        chosen = sorted(rng.choice(len(all_offsets), size=n_eval_windows, replace=False))
        all_offsets = [all_offsets[i] for i in chosen]

    inputs_list, labels_list = [], []
    for off in all_offsets:
        window = data_ch[:, off - window_len: off]      # [C, window_len]
        inputs_list.append(window[:, :context_length])  # [C, L]
        labels_list.append(window[:, -horizon:])         # [C, H]

    return np.stack(inputs_list), np.stack(labels_list)  # [N, C, L], [N, C, H]


# ── Inference helpers ──────────────────────────────────────────────────────────

def _make_uv_item(ctx_1d, eos_token, pre_norm_mean=0.0, pre_norm_std=1.0):
    """Build a UV batch-dict from a 1-D (already z-norm) context array.

    pre_norm_mean / pre_norm_std: the mean and std of the window BEFORE z-normalisation
    (i.e. in StandardScaler space).  These match what the model saw during training,
    where norm_mean/norm_std in the prompt were the original (pre-z-norm) statistics.
    """
    T = len(ctx_1d)
    prompt = (
        f"This is a univariate time series with 1 channel "
        f"and {T} time steps. "
        f"It has mean {pre_norm_mean:.4f} and std {pre_norm_std:.4f}. <ts>"
    )
    return {
        "type":                  "chronos",
        "source":                "benchmark",
        "answer":                eos_token,
        "post_prompt":           "",
        "pre_prompt":            "",
        "time_series":           [np.asarray(ctx_1d, dtype=np.float32)],
        "time_series_text":      [prompt],
        "ts_already_normalized": True,
        "time_series_norm_stats": [(pre_norm_mean, pre_norm_std)],
        "norm_mean":             float(pre_norm_mean),
        "norm_std":              float(pre_norm_std),
    }


@torch.no_grad()
def predict_next_patch(model, batch_items, device):
    """
    One forward pass: for each UV item, predict the next patch.

    Returns Tensor[B, P] in z-norm space.

    Implementation follows the "direct encoder -> projector -> decoder" path
    described in _generate_impl (the path that replaced the removed autoregressive
    patch-in-LLM-loop approach):
      pad_and_apply_batch -> LLM hidden states -> decoder_projector
        -> layer_norm -> decoder -> last-position output
    """
    B = len(batch_items)

    # Build embeddings (return_targets=True gives us ts_mask for free)
    inputs_embeds, attn_mask, _, ts_mask, _, _, _ = model.pad_and_apply_batch(
        batch_items, return_targets=True
    )

    # LLM backbone
    out = model.llm(
        inputs_embeds=inputs_embeds,
        attention_mask=attn_mask,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    last_hidden = out.hidden_states[-1]  # [B, L, H]

    # Extract TS-patch hidden states (flat, UV → one row per item)
    ts_hidden = last_hidden[ts_mask]  # [N_flat, H]
    ts_hidden = model.decoder_projector(ts_hidden.float())
    ts_hidden = F.layer_norm(ts_hidden, [ts_hidden.size(-1)])

    # Split by item and run the causal decoder; take the last position's output
    per_item_counts = [int(ts_mask[b].sum()) for b in range(B)]
    parts = list(torch.split(ts_hidden, per_item_counts))

    preds = []
    for part in parts:
        # part: [N_patches, enc_dim]
        dec_out = model.decoder(part.unsqueeze(0))  # [1, N_patches, P]
        preds.append(dec_out[0, -1, :])             # [P]

    return torch.stack(preds, dim=0)  # [B, P]


# ── Autoregressive forecasting ─────────────────────────────────────────────────

def forecast_windows(model, eos_token, contexts_sc, n_patches, device, inner_bs=16):
    """
    Autoregressively predict n_patches future patches for a batch of UV windows.

    contexts_sc: float32 ndarray [B, context_length] — StandardScaler space
    Returns: float32 ndarray [B, n_patches * PATCH_SIZE] — StandardScaler space
    """
    B, L = contexts_sc.shape

    # Per-sample z-norm on the context window
    means = contexts_sc.mean(axis=-1, keepdims=True)                          # [B, 1]
    stds = np.maximum(contexts_sc.std(axis=-1, keepdims=True), NORM_STD_FLOOR)  # [B, 1]
    ctx_norm = (contexts_sc - means) / stds                                    # [B, L]

    # Pre-z-norm stats used for prompt text (matches training protocol) and denorm
    per_sample_mean = means[:, 0].tolist()  # [B]
    per_sample_std  = stds[:, 0].tolist()   # [B]

    # Running context (grows by PATCH_SIZE each step); clip to MAX_CTX_TIMESTEPS
    curr_ctx = [ctx_norm[b].tolist() for b in range(B)]
    all_patch_preds = []  # list of n_patches × ndarray [B, P]

    for _k in range(n_patches):
        batch_preds = []
        for s in range(0, B, inner_bs):
            e = min(s + inner_bs, B)
            items = [
                _make_uv_item(curr_ctx[b][-MAX_CTX_TIMESTEPS:], eos_token,
                              per_sample_mean[b], per_sample_std[b])
                for b in range(s, e)
            ]
            pred_t = predict_next_patch(model, items, device)  # [b, P]
            batch_preds.append(pred_t.cpu().float().numpy())

        next_patch = np.concatenate(batch_preds, axis=0)  # [B, P]
        all_patch_preds.append(next_patch)

        # Extend each context with the predicted patch
        for b in range(B):
            curr_ctx[b].extend(all_patch_preds[-1][b].tolist())

    # [B, n_patches * P]
    pred_norm = np.concatenate(all_patch_preds, axis=-1)

    # Denormalize back to StandardScaler space
    return (pred_norm * stds + means).astype(np.float32)


# ── Evaluation loop ────────────────────────────────────────────────────────────

def evaluate_dataset(
    model, eos_token, inputs, labels, horizon, n_patches, device,
    outer_bs=256, inner_bs=16, amp_dtype=torch.bfloat16,
):
    """
    Evaluate MSE/MAE on all (window, channel) pairs.
    inputs: [N, C, L], labels: [N, C, H]
    """
    N, C, L = inputs.shape

    # Flatten: channel-independent, treat each channel as a separate series
    inp_flat = inputs.reshape(N * C, L)       # [N*C, L]
    lbl_flat = labels.reshape(N * C, horizon)  # [N*C, H]

    mse_sum = mae_sum = count = 0

    for start in tqdm(range(0, N * C, outer_bs), desc="  forecast", leave=False):
        end = min(start + outer_bs, N * C)
        ctx_batch = inp_flat[start:end]  # [B, L]
        lbl_batch = lbl_flat[start:end]  # [B, H]

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device == "cuda")):
            pred = forecast_windows(
                model, eos_token, ctx_batch, n_patches, device, inner_bs=inner_bs
            )

        pred = pred[:, :horizon]  # truncate overhanging patch tail to exact horizon

        mse_sum += float(np.sum((pred - lbl_batch) ** 2))
        mae_sum += float(np.sum(np.abs(pred - lbl_batch)))
        count += pred.size

    return mse_sum / count, mae_sum / count


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Benchmark eval of OpenTSLMSPFixPos2D")
    ap.add_argument("--ckpt", required=True, help="Path to best_model.pt")
    ap.add_argument("--llm_id", default="google/gemma-3-270m")
    ap.add_argument("--data_dir", required=True,
                    help="Directory containing ETTh1.csv, ETTh2.csv, weather.csv")
    ap.add_argument("--datasets", nargs="+",
                    default=["ETTh1", "ETTh2", "weather"],
                    help="Dataset names (CSV filenames without .csv)")
    ap.add_argument("--context_lengths", type=int, nargs="+", default=[512, 2048])
    ap.add_argument("--horizons", type=int, nargs="+", default=[96, 192, 336, 720])
    ap.add_argument("--n_eval_windows", type=int, default=500,
                    help="Max windows to sample per (dataset, ctx, horizon) combo")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outer_bs", type=int, default=256,
                    help="Windows per outer batch (memory)")
    ap.add_argument("--inner_bs", type=int, default=16,
                    help="Items per model forward pass")
    ap.add_argument("--out_json", default="output_job/benchmark_results.json")
    ap.add_argument("--lora_r",   type=int, default=0,
                    help="LoRA rank; if >0, enable_lora() is called before loading ckpt")
    ap.add_argument("--lora_alpha", type=int, default=8)
    args = ap.parse_args()

    print("=" * 70)
    print("OpenTSLMSPFixPos2D — Zero-Shot Benchmark Evaluation")
    print("=" * 70)
    print(f"  ckpt          : {args.ckpt}")
    print(f"  data_dir      : {args.data_dir}")
    print(f"  datasets      : {args.datasets}")
    print(f"  ctx lengths   : {args.context_lengths}")
    print(f"  horizons      : {args.horizons}")
    print(f"  n_eval_windows: {args.n_eval_windows}  seed={args.seed}")
    print(f"  device        : {args.device}  inner_bs={args.inner_bs}")
    print()

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"[model] loading {args.llm_id} ...")
    model = OpenTSLMSPFixPos2D(
        args=build_model_args(),
        llm_id=args.llm_id,
        device=args.device,
    )
    if args.lora_r > 0:
        model.enable_lora(lora_r=args.lora_r, lora_alpha=args.lora_alpha)
        print(f"[model] LoRA enabled  r={args.lora_r}  alpha={args.lora_alpha}")
    model.load_from_file(args.ckpt)
    model.eval().to(args.device)
    eos_token = model.get_eos_token()
    print(f"[model] loaded  eos={eos_token!r}\n")

    amp_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    results = []

    for ds_name in args.datasets:
        csv_path = os.path.join(args.data_dir, f"{ds_name}.csv")
        if not os.path.isfile(csv_path):
            print(f"[skip] {csv_path} not found")
            continue

        for ctx_len in args.context_lengths:
            for horizon in args.horizons:
                n_patches = math.ceil(horizon / PATCH_SIZE)
                tag = f"{ds_name}  ctx={ctx_len}  H={horizon}"
                print(f">>> {tag}  (n_patches={n_patches})")

                try:
                    inputs, labels = load_test_windows(
                        csv_path, ctx_len, horizon,
                        n_eval_windows=args.n_eval_windows,
                        seed=args.seed,
                    )
                except ValueError as e:
                    print(f"    [skip] {e}")
                    continue

                N, C, L = inputs.shape
                print(f"    windows={N}  channels={C}  context={L}")

                with torch.no_grad():
                    mse, mae = evaluate_dataset(
                        model, eos_token, inputs, labels,
                        horizon, n_patches, args.device,
                        outer_bs=args.outer_bs, inner_bs=args.inner_bs,
                        amp_dtype=amp_dtype,
                    )

                print(f"    MSE={mse:.4f}  MAE={mae:.4f}")
                results.append({
                    "dataset": ds_name,
                    "context_length": ctx_len,
                    "horizon": horizon,
                    "n_windows": N,
                    "n_channels": C,
                    "mse": round(mse, 6),
                    "mae": round(mae, 6),
                })

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("RESULTS SUMMARY (channel-independent, StandardScaler space)")
    print("=" * 70)
    print(f"{'Dataset':<10} {'ctx':>6} {'H':>4}   {'MSE':>8} {'MAE':>8}")
    print("-" * 45)
    for r in results:
        print(
            f"{r['dataset']:<10} {r['context_length']:>6} {r['horizon']:>4}"
            f"   {r['mse']:>8.4f} {r['mae']:>8.4f}"
        )
    print("=" * 70)

    # TimeMoE-50M reference numbers from paper Table 3 (ctx=512)
    ref = {
        ("ETTh1", 512, 96):  (0.395, 0.424),
        ("ETTh1", 512, 192): (0.440, 0.452),
        ("ETTh1", 512, 336): (0.485, 0.482),
        ("ETTh1", 512, 720): (0.475, 0.483),
        ("ETTh2", 512, 96):  (0.289, 0.340),
        ("ETTh2", 512, 192): (0.379, 0.393),
        ("ETTh2", 512, 336): (0.447, 0.438),
        ("ETTh2", 512, 720): (0.455, 0.453),
        ("weather", 512, 96):  (0.171, 0.218),
        ("weather", 512, 192): (0.220, 0.260),
        ("weather", 512, 336): (0.277, 0.298),
        ("weather", 512, 720): (0.361, 0.348),
    }
    if ref:
        print()
        print("TimeMoE-50M reference (ctx=512, from paper Table 3):")
        print(f"{'Dataset':<10} {'H':>4}   {'MSE_ref':>8} {'MAE_ref':>8}   {'our_MSE':>8}")
        print("-" * 55)
        for r in results:
            key = (r["dataset"], r["context_length"], r["horizon"])
            if key in ref:
                mse_ref, mae_ref = ref[key]
                delta = r["mse"] - mse_ref
                arrow = "▼" if delta < 0 else "▲"
                print(
                    f"{r['dataset']:<10} {r['horizon']:>4}"
                    f"   {mse_ref:>8.4f} {mae_ref:>8.4f}"
                    f"   {r['mse']:>8.4f}  {arrow}{abs(delta):.4f}"
                )

    # Save JSON
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {args.out_json}")


if __name__ == "__main__":
    main()
