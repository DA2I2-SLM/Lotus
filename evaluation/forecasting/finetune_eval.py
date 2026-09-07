#!/usr/bin/env python3
"""
Fine-tune + evaluate OpenTSLMSPFixPos2D on ETTh1/ETTh2/Weather.

Follows Time-MoE in-distribution protocol (Section 4.2):
  - Fine-tune for 1 epoch on the train split (StandardScaler normalized)
  - Evaluate on test split: ctx=[512,1024,2048,3072], horizons=[96,192,336,720]
  - Channel-independent (UV), patch_size=32, MSE+MAE

Fine-tuning uses the INFERENCE path (encoder→projector→LLM→decoder_projector→decoder)
so all inference-time parameters receive gradient signal.
LLM base weights remain frozen; LoRA weights updated if --lora_r > 0.

With --full_ft: ALL parameters (including LLM backbone) are unfrozen and trained.
LLM uses a separate lower learning rate (--llm_lr, default 1e-5) to avoid
catastrophic forgetting; encoder/projector/decoder use --ft_lr (default 1e-4).
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
MAX_CTX_TIMESTEPS = 2048
NORM_STD_FLOOR = 1e-3


# ── Model ──────────────────────────────────────────────────────────────────────

def build_model_args(mvpos=True):
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
        llm_mv_position_ids=mvpos,
        llm_dtype="auto",
        gradient_checkpointing=False,
        mlp_enc_hidden_dim=None,
        mlp_enc_ff_dim=None,
        mlp_dec_hidden_dim=None,
        mlp_dec_ff_dim=None,
        use_only_time_pe=False,
        use_only_channel_pe=False,
    )


# ── Data ───────────────────────────────────────────────────────────────────────

def get_splits(csv_path, context_length):
    """Return (train_data, test_data) as float32 [T, C] in StandardScaler space."""
    df = pd.read_csv(csv_path)
    base = os.path.basename(csv_path).lower()

    if "etth" in base:
        border1s = [0, 12*30*24 - context_length, 12*30*24 + 4*30*24 - context_length]
        border2s = [12*30*24, 12*30*24 + 4*30*24, 12*30*24 + 8*30*24]
    elif "ettm" in base:
        border1s = [0, 12*30*24*4 - context_length, 12*30*24*4 + 4*30*24*4 - context_length]
        border2s = [12*30*24*4, 12*30*24*4 + 4*30*24*4, 12*30*24*4 + 8*30*24*4]
    else:
        n = len(df)
        num_train = int(n * 0.7)
        num_test  = int(n * 0.2)
        border1s = [0, num_train - context_length, n - num_test - context_length]
        border2s = [num_train, num_train + (n - num_train - num_test), n]

    cols = df.columns[1:]
    vals = df[cols].values.astype(np.float32)

    scaler = StandardScaler()
    scaler.fit(vals[border1s[0]:border2s[0]])
    return (
        scaler.transform(vals[border1s[0]:border2s[0]]),  # [T_train, C]
        scaler.transform(vals[border1s[2]:border2s[2]]),  # [T_test,  C]
    )


def build_train_windows(train_sc, ft_ctx, patch_size, stride):
    """
    Sliding windows from train split.
    Returns list of (ctx_1d [ft_ctx], tgt_1d [patch_size]) in StandardScaler space.
    """
    T, C = train_sc.shape
    window = ft_ctx + patch_size
    windows = []
    for c in range(C):
        seq = train_sc[:, c]
        for s in range(0, T - window + 1, stride):
            windows.append((seq[s:s + ft_ctx].copy(), seq[s + ft_ctx:s + window].copy()))
    return windows


def load_test_windows(test_sc, context_length, horizon, n_windows=None, seed=42):
    """
    Sliding windows from test split (step-1), optionally subsampled.
    Returns (inputs [N, C, L], labels [N, C, H]) in StandardScaler space.
    """
    T, C = test_sc.shape
    window = context_length + horizon
    data_ch = test_sc.T  # [C, T]

    all_offsets = list(range(window, T))
    if not all_offsets:
        raise ValueError(f"No valid windows: T_test={T}, window={window}")

    if n_windows is not None and n_windows < len(all_offsets):
        rng = np.random.default_rng(seed)
        chosen = sorted(rng.choice(len(all_offsets), size=n_windows, replace=False))
        all_offsets = [all_offsets[i] for i in chosen]

    inputs_list, labels_list = [], []
    for off in all_offsets:
        win = data_ch[:, off - window:off]        # [C, window]
        inputs_list.append(win[:, :context_length])
        labels_list.append(win[:, -horizon:])

    return np.stack(inputs_list), np.stack(labels_list)  # [N, C, L], [N, C, H]


# ── Batch item builder ─────────────────────────────────────────────────────────

def make_uv_item(ctx_1d, eos_token):
    """Build UV batch dict from a z-normalized context with per-window z-norm."""
    mean_ = float(ctx_1d.mean())
    std_  = max(float(ctx_1d.std()), NORM_STD_FLOOR)
    ctx_n = (ctx_1d - mean_) / std_
    T = len(ctx_1d)
    prompt = (
        f"This is a univariate time series with 1 channel "
        f"and {T} time steps. "
        f"It has mean {mean_:.4f} and std {std_:.4f}. <ts>"
    )
    item = {
        "type":  "chronos",
        "source": "benchmark",
        "answer": eos_token,
        "post_prompt": "",
        "pre_prompt": "",
        "time_series": [np.asarray(ctx_n, dtype=np.float32)],
        "time_series_text": [prompt],
        "ts_already_normalized": True,
        "time_series_norm_stats": [(mean_, std_)],
        "norm_mean": mean_,
        "norm_std": std_,
    }
    return item, mean_, std_


# ── Training step ──────────────────────────────────────────────────────────────

def train_step(model, batch_windows, eos_token, device):
    """
    One training step via inference path (decoder_projector).
    batch_windows: list of (ctx_1d, tgt_1d) in StandardScaler space.
    Returns: loss tensor.
    """
    B = len(batch_windows)
    batch_items, tgt_norms = [], []

    for ctx, tgt in batch_windows:
        item, mean_, std_ = make_uv_item(ctx, eos_token)
        batch_items.append(item)
        tgt_n = torch.tensor((tgt - mean_) / std_, dtype=torch.float32, device=device)
        tgt_norms.append(tgt_n)

    inputs_embeds, attn_mask, _, ts_mask, _, _, _ = model.pad_and_apply_batch(
        batch_items, return_targets=True
    )

    out = model.llm(
        inputs_embeds=inputs_embeds,
        attention_mask=attn_mask,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    last_hidden = out.hidden_states[-1]

    ts_hidden = last_hidden[ts_mask]
    ts_hidden = model.decoder_projector(ts_hidden.float())
    ts_hidden = F.layer_norm(ts_hidden, [ts_hidden.size(-1)])

    per_item_counts = [int(ts_mask[b].sum()) for b in range(B)]
    parts = list(torch.split(ts_hidden, per_item_counts))

    losses = []
    for i, part in enumerate(parts):
        dec_out = model.decoder(part.unsqueeze(0))   # [1, N_patches, P]
        pred_next = dec_out[0, -1, :]                # [P] — next patch in z-norm space
        losses.append(F.mse_loss(pred_next, tgt_norms[i]))

    return torch.stack(losses).mean()


# ── Fine-tune for 1 epoch ──────────────────────────────────────────────────────

def finetune(model, train_windows, eos_token, optimizer, scheduler, device,
             batch_size, grad_accum, full_ft=False):
    """One epoch of fine-tuning. Returns average loss."""
    if full_ft:
        model.train()   # all modules (including LLM) in train mode
    else:
        # LLM stays in eval mode (frozen, no dropout); trainable modules in train mode
        model.llm.eval()
        model.encoder.train()
        model.projector.train()
        model.decoder_projector.train()
        model.decoder.train()
        # LoRA layers inside the LLM are in train mode
        for name, mod in model.llm.named_modules():
            if "lora" in name.lower():
                mod.train()

    indices = np.random.permutation(len(train_windows))
    total_loss, n_steps = 0.0, 0
    optimizer.zero_grad()
    accum = 0

    pbar = tqdm(range(0, len(indices), batch_size), desc="  ft", leave=False)
    for batch_start in pbar:
        idx = indices[batch_start: batch_start + batch_size]
        batch = [train_windows[i] for i in idx]

        loss = train_step(model, batch, eos_token, device)
        (loss / grad_accum).backward()
        accum += 1
        total_loss += float(loss)
        n_steps += 1

        if accum >= grad_accum:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            accum = 0
        pbar.set_postfix(loss=f"{float(loss):.4f}")

    if accum > 0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

    return total_loss / max(n_steps, 1)


# ── Inference ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_next_patch(model, batch_items, device):
    """One forward pass → predicted next patch [B, P] in z-norm space."""
    B = len(batch_items)
    inputs_embeds, attn_mask, _, ts_mask, _, _, _ = model.pad_and_apply_batch(
        batch_items, return_targets=True
    )
    out = model.llm(
        inputs_embeds=inputs_embeds,
        attention_mask=attn_mask,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    last_hidden = out.hidden_states[-1]
    ts_hidden = last_hidden[ts_mask]
    ts_hidden = model.decoder_projector(ts_hidden.float())
    ts_hidden = F.layer_norm(ts_hidden, [ts_hidden.size(-1)])

    per_item_counts = [int(ts_mask[b].sum()) for b in range(B)]
    parts = list(torch.split(ts_hidden, per_item_counts))

    preds = []
    for part in parts:
        dec_out = model.decoder(part.unsqueeze(0))
        preds.append(dec_out[0, -1, :])
    return torch.stack(preds, dim=0)


def forecast_windows(model, eos_token, contexts_sc, n_patches, device, inner_bs=16):
    """Autoregressive patch forecasting. Returns [B, n_patches*P] in SC space."""
    B, L = contexts_sc.shape
    means = contexts_sc.mean(axis=-1, keepdims=True)
    stds  = np.maximum(contexts_sc.std(axis=-1, keepdims=True), NORM_STD_FLOOR)
    ctx_n = (contexts_sc - means) / stds

    curr_ctx = [ctx_n[b].tolist() for b in range(B)]
    all_preds = []

    for _ in range(n_patches):
        batch_preds = []
        for s in range(0, B, inner_bs):
            e = min(s + inner_bs, B)
            items = []
            for b in range(s, e):
                clip = curr_ctx[b][-MAX_CTX_TIMESTEPS:]
                ctx_arr = np.array(clip, dtype=np.float32)
                m = float(means[b, 0])
                st = float(stds[b, 0])
                T = len(ctx_arr)
                prompt = (
                    f"This is a univariate time series with 1 channel "
                    f"and {T} time steps. "
                    f"It has mean {m:.4f} and std {st:.4f}. <ts>"
                )
                items.append({
                    "type": "chronos", "source": "benchmark",
                    "answer": eos_token, "post_prompt": "", "pre_prompt": "",
                    "time_series": [ctx_arr],
                    "time_series_text": [prompt],
                    "ts_already_normalized": True,
                    "time_series_norm_stats": [(m, st)],
                    "norm_mean": m, "norm_std": st,
                })
            pred_t = predict_next_patch(model, items, device)
            batch_preds.append(pred_t.cpu().float().numpy())

        next_patch = np.concatenate(batch_preds, axis=0)
        all_preds.append(next_patch)
        for b in range(B):
            curr_ctx[b].extend(next_patch[b].tolist())

    pred_norm = np.concatenate(all_preds, axis=-1)
    return (pred_norm * stds + means).astype(np.float32)


def evaluate_dataset(model, eos_token, inputs, labels, horizon, n_patches, device,
                     outer_bs=256, inner_bs=16, amp_dtype=torch.bfloat16):
    """MSE/MAE over all (window, channel) pairs. inputs [N,C,L], labels [N,C,H]."""
    N, C, L = inputs.shape
    inp_flat = inputs.reshape(N * C, L)
    lbl_flat = labels.reshape(N * C, horizon)

    mse_sum = mae_sum = count = 0
    for start in tqdm(range(0, N * C, outer_bs), desc="  eval", leave=False):
        end = min(start + outer_bs, N * C)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device != "cpu")):
            pred = forecast_windows(
                model, eos_token, inp_flat[start:end], n_patches, device, inner_bs
            )
        pred = pred[:, :horizon]
        lbl  = lbl_flat[start:end]
        mse_sum += float(np.sum((pred - lbl) ** 2))
        mae_sum += float(np.sum(np.abs(pred - lbl)))
        count   += pred.size

    return mse_sum / count, mae_sum / count


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",      required=True)
    ap.add_argument("--llm_id",    default="google/gemma-3-270m")
    ap.add_argument("--lora_r",    type=int, default=0)
    ap.add_argument("--lora_alpha", type=int, default=8)
    ap.add_argument("--no_mvpos",  action="store_true",
                    help="Disable llm_mv_position_ids (for seqrope baseline)")
    ap.add_argument("--full_ft",   action="store_true",
                    help="Unfreeze LLM backbone for full fine-tuning")
    ap.add_argument("--llm_lr",    type=float, default=1e-5,
                    help="LLM backbone learning rate when --full_ft is set")
    ap.add_argument("--data_dir",  required=True)
    ap.add_argument("--datasets",  nargs="+", default=["ETTh1", "ETTh2", "weather"])
    ap.add_argument("--zero_shot", action="store_true",
                    help="Skip fine-tuning; evaluate checkpoint directly (zero-shot)")
    # Fine-tuning
    ap.add_argument("--ft_ctx",    type=int, default=512, help="Context length during fine-tuning")
    ap.add_argument("--ft_bs",     type=int, default=8,   help="Batch size per fine-tune step")
    ap.add_argument("--ft_accum",  type=int, default=4,   help="Gradient accumulation steps")
    ap.add_argument("--ft_lr",     type=float, default=1e-4)
    ap.add_argument("--ft_min_lr", type=float, default=1e-5)
    ap.add_argument("--ft_stride", type=int, default=32,  help="Stride for train windows (=PATCH_SIZE)")
    ap.add_argument("--ft_epochs", type=int, default=1)
    # Evaluation
    ap.add_argument("--eval_ctx",  type=int, nargs="+", default=[512, 1024, 2048, 3072])
    ap.add_argument("--horizons",  type=int, nargs="+", default=[96, 192, 336, 720])
    ap.add_argument("--n_windows", type=int, default=500)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--outer_bs",  type=int, default=256)
    ap.add_argument("--inner_bs",  type=int, default=16)
    ap.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_json",  default="output_job/finetune_results.json")
    args = ap.parse_args()

    print("=" * 70)
    print("OpenTSLMSPFixPos2D — Fine-tune + Evaluate (Time-MoE Protocol)")
    print("=" * 70)
    print(f"  ckpt       : {args.ckpt}")
    print(f"  lora_r     : {args.lora_r}")
    print(f"  full_ft    : {args.full_ft}  llm_lr={args.llm_lr}")
    print(f"  ft_ctx     : {args.ft_ctx}  ft_bs={args.ft_bs}  accum={args.ft_accum}")
    print(f"  ft_lr      : {args.ft_lr}   epochs={args.ft_epochs}")
    print(f"  eval_ctx   : {args.eval_ctx}")
    print(f"  horizons   : {args.horizons}")
    print()

    amp_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    results = []

    for ds_name in args.datasets:
        csv_path = os.path.join(args.data_dir, f"{ds_name}.csv")
        if not os.path.isfile(csv_path):
            # Hard failure, not a skip. Skipping wrote an empty results list and
            # exited 0, so a whole SLURM array reported COMPLETED in ~20s with
            # 2-byte JSONs and nothing in stderr — a silently missing dataset
            # looked exactly like a successful run.
            raise FileNotFoundError(
                f"Dataset CSV not found: {csv_path}\n"
                f"  --data_dir was: {args.data_dir}\n"
                f"  --datasets requested: {args.datasets}"
            )

        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        # ── Load model fresh for each dataset (independent fine-tuning) ──
        print(f"[model] loading {args.llm_id} ...")
        model = OpenTSLMSPFixPos2D(
            args=build_model_args(mvpos=not args.no_mvpos),
            llm_id=args.llm_id,
            device=args.device,
        )
        if args.lora_r > 0:
            model.enable_lora(lora_r=args.lora_r, lora_alpha=args.lora_alpha)
            print(f"[model] LoRA enabled  r={args.lora_r}  alpha={args.lora_alpha}")
        model.load_from_file(args.ckpt)
        if args.full_ft:
            for p in model.llm.parameters():
                p.requires_grad_(True)
            print(f"[model] full_ft: LLM backbone unfrozen ({sum(p.numel() for p in model.llm.parameters())/1e6:.1f}M params)")
        model.to(args.device)
        eos_token = model.get_eos_token()
        print(f"[model] loaded  eos={eos_token!r}")

        if args.zero_shot:
            print("[zero-shot] skipping fine-tuning — evaluating checkpoint directly")
        else:
            # ── Data splits ──────────────────────────────────────────────────
            train_sc, _ = get_splits(csv_path, args.ft_ctx)
            print(f"[data] train={train_sc.shape}")

            # ── Fine-tuning ──────────────────────────────────────────────────
            train_windows = build_train_windows(train_sc, args.ft_ctx, PATCH_SIZE, args.ft_stride)
            print(f"[ft] {len(train_windows)} training windows  (ctx={args.ft_ctx}, stride={args.ft_stride})")

            if args.full_ft:
                llm_ids = {id(p) for p in model.llm.parameters()}
                other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in llm_ids]
                llm_params   = [p for p in model.llm.parameters() if p.requires_grad]
                n_trainable  = sum(p.numel() for p in other_params) + sum(p.numel() for p in llm_params)
                print(f"[ft] trainable params: {n_trainable:,}  "
                      f"(enc/dec {sum(p.numel() for p in other_params):,} @ lr={args.ft_lr}  "
                      f"llm {sum(p.numel() for p in llm_params):,} @ lr={args.llm_lr})")
                param_groups = [
                    {"params": other_params, "lr": args.ft_lr},
                    {"params": llm_params,   "lr": args.llm_lr},
                ]
            else:
                param_groups = [p for p in model.parameters() if p.requires_grad]
                n_trainable  = sum(p.numel() for p in param_groups)
                print(f"[ft] trainable params: {n_trainable:,}")

            optimizer = torch.optim.AdamW(
                param_groups, lr=args.ft_lr, weight_decay=0.1,
                betas=(0.9, 0.95), eps=1e-8,
            )
            n_updates = math.ceil(len(train_windows) / args.ft_bs / args.ft_accum) * args.ft_epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(n_updates, 1), eta_min=args.ft_min_lr
            )
            for ep in range(args.ft_epochs):
                avg_loss = finetune(
                    model, train_windows, eos_token, optimizer, scheduler,
                    args.device, args.ft_bs, args.ft_accum,
                    full_ft=args.full_ft,
                )
                print(f"[ft] epoch {ep+1}/{args.ft_epochs}  loss={avg_loss:.4f}")

        # ── Evaluation on test split ─────────────────────────────────────
        model.eval()
        model.llm.eval()

        # We must re-compute test data with each evaluation context length
        for ctx_len in args.eval_ctx:
            # Re-load splits for each ctx to get the correct test window offsets
            train_sc_eval, test_sc_eval = get_splits(csv_path, ctx_len)

            for horizon in args.horizons:
                n_patches = math.ceil(horizon / PATCH_SIZE)
                tag = f"{ds_name}  ctx={ctx_len}  H={horizon}"
                print(f">>> {tag}  (n_patches={n_patches})")

                try:
                    inputs, labels = load_test_windows(
                        test_sc_eval, ctx_len, horizon,
                        n_windows=args.n_windows, seed=args.seed
                    )
                except ValueError as e:
                    print(f"    [skip] {e}")
                    continue

                N, C, L = inputs.shape
                print(f"    windows={N}  channels={C}")

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

        # Free model memory before next dataset
        del model
        torch.cuda.empty_cache()

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("FINE-TUNED RESULTS (Time-MoE protocol, 1 epoch)")
    print("=" * 70)
    print(f"{'Dataset':<10} {'ctx':>6} {'H':>4}   {'MSE':>8} {'MAE':>8}")
    print("-" * 45)
    for r in results:
        print(
            f"{r['dataset']:<10} {r['context_length']:>6} {r['horizon']:>4}"
            f"   {r['mse']:>8.4f} {r['mae']:>8.4f}"
        )
    print("=" * 70)

    # Time-MoE reference (in-distribution Table 4, TimeMoE_base, ctx=512)
    ref_ft = {
        ("ETTh1", 512, 96):  (0.345, 0.375), ("ETTh1", 512, 192): (0.388, 0.411),
        ("ETTh1", 512, 336): (0.414, 0.427), ("ETTh1", 512, 720): (0.425, 0.433),
        ("ETTh2", 512, 96):  (0.274, 0.333), ("ETTh2", 512, 192): (0.343, 0.386),
        ("ETTh2", 512, 336): (0.362, 0.402), ("ETTh2", 512, 720): (0.380, 0.424),
        ("weather", 512, 96):  (0.149, 0.201), ("weather", 512, 192): (0.193, 0.244),
        ("weather", 512, 336): (0.245, 0.285), ("weather", 512, 720): (0.318, 0.337),
    }
    has_ref = any((r["dataset"], r["context_length"], r["horizon"]) in ref_ft for r in results)
    if has_ref:
        print()
        print("vs. Time-MoE_base fine-tuned (Table 4):")
        print(f"{'Dataset':<10} {'H':>4}   {'TimeMoE':>8} {'Ours':>8}  {'Δ':>7}")
        for r in results:
            key = (r["dataset"], r["context_length"], r["horizon"])
            if key in ref_ft:
                ref_mse, _ = ref_ft[key]
                delta = r["mse"] - ref_mse
                arrow = "▼" if delta < 0 else "▲"
                print(f"{r['dataset']:<10} {r['horizon']:>4}   "
                      f"{ref_mse:>8.4f} {r['mse']:>8.4f}  {arrow}{abs(delta):.4f}")

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {args.out_json}")


if __name__ == "__main__":
    main()
