#!/usr/bin/env python3
"""
Evaluate our pretrained OpenTSLMSPFixPos2D checkpoint on the test_synthetic split.

Replicates curriculum_learning.py's val_mse_synthetic computation exactly:
  - Loads test_synthetic rows from the pre-built HF cache (avoids OOM from
    rebuilding the full imputed chronos cache)
  - Applies Phase1ChronosFuturePatchDataset._sample_phase1_window(mode="test")
    logic directly: last WINDOW_LEN steps as history, next PATCH_SIZE as future,
    per-channel z-norm (std_floor=1e-3, clip=20)
  - Calls model.compute_pretrain_loss(batch, mse_only=True) → recon_loss
  - Averages recon_loss across batches

Usage:
    python evaluation/forecasting/our_model_synthetic_eval.py \
        --ckpt results/sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/phase1_chronos_mse_freq_mvpos/checkpoints/best_model.pt \
        --test_split_dir _preprocessed_chronos_uv_synth_phase1_mse/all/test_synthetic \
        --n_samples 2000 --seed 42 --device cuda
"""

import argparse
import os
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from opentslm.model.llm.OpenTSLMSPFixPos2D import OpenTSLMSPFixPos2D


PATCH_SIZE        = 32
WINDOW_LEN        = 2048
MIN_HIST_PATCHES  = 8
NORM_STD_FLOOR    = 1e-3
NORM_CLIP         = 20.0
VALID_WINDOW_LENS = [WINDOW_LEN]     # matches WINDOW_LENS in model_config.py


# ── Preprocessing: mirrors Phase1ChronosFuturePatchDataset._sample_phase1_window ──

def _sample_test_window(series_np: np.ndarray, eos_token: str) -> dict | None:
    """Extract history + future for one series (univariate or multivariate).

    Replicates Phase1ChronosFuturePatchDataset._sample_phase1_window with
    mode='test' (deterministic: take the last valid window).

    Returns a batch dict ready for model.compute_pretrain_loss, or None.
    """
    series = torch.as_tensor(series_np, dtype=torch.float32)
    is_mv = series.dim() == 2
    T = series.shape[0]

    min_history_len = MIN_HIST_PATCHES * PATCH_SIZE   # 256
    min_total_len   = min_history_len + PATCH_SIZE    # 288

    if T < 2 * PATCH_SIZE:
        return None

    if T < min_total_len:
        # Pad short series (last-value fill) to reach min_total_len
        pad_len = min_total_len - T
        if is_mv:
            pad_block = series[-1:].expand(pad_len, -1)
        else:
            pad_block = series.new_full((pad_len,), float(series[-1]))
        series = torch.cat([series, pad_block], dim=0)
        T = series.shape[0]

    max_history_len = ((T - PATCH_SIZE) // PATCH_SIZE) * PATCH_SIZE
    max_history_len = max(PATCH_SIZE, max_history_len)

    candidates = [w for w in VALID_WINDOW_LENS if min_history_len <= w <= max_history_len]
    history_len = max(candidates) if candidates else max_history_len

    start = max(0, T - history_len - PATCH_SIZE)   # deterministic (mode="test")

    history_raw = series[start : start + history_len]
    future_raw  = series[start + history_len : start + history_len + PATCH_SIZE]

    # Per-channel z-norm on history window
    if is_mv:
        ch_mean = history_raw.mean(dim=0, keepdim=True)           # [1, C]
        ch_std  = history_raw.std(dim=0, keepdim=True).clamp(min=NORM_STD_FLOOR)
        history_norm = (history_raw - ch_mean) / ch_std
        future_norm  = (future_raw  - ch_mean) / ch_std
        if NORM_CLIP is not None:
            history_norm = history_norm.clamp(-NORM_CLIP, NORM_CLIP)
            future_norm  = future_norm.clamp(-NORM_CLIP, NORM_CLIP)
        norm_mean = float(ch_mean[0, 0])
        norm_std  = float(ch_std[0, 0])
        ts_future = future_norm.transpose(0, 1).to(torch.float32)  # [C, P]
        n_channels = series.shape[1]
        modality   = "multivariate"
        ch_word    = "channels"
    else:
        h_mean = float(history_raw.mean())
        h_std  = max(float(history_raw.std()), NORM_STD_FLOOR)
        history_norm = ((history_raw - h_mean) / h_std).clamp(-NORM_CLIP, NORM_CLIP)
        future_norm  = ((future_raw  - h_mean) / h_std).clamp(-NORM_CLIP, NORM_CLIP)
        norm_mean = h_mean
        norm_std  = h_std
        ts_future = future_norm.to(torch.float32)                  # [P]
        n_channels = 1
        modality   = "univariate"
        ch_word    = "channel"

    T_hist = history_norm.shape[0]
    prompt_text = (
        f"This is a {modality} time series with {n_channels} {ch_word} "
        f"and {T_hist} time steps. "
        f"It has mean {norm_mean:.4f} and std {norm_std:.4f}. <ts>"
    )

    return {
        "type":                     "chronos",
        "source":                   "synthetic",
        "answer":                   eos_token,
        "post_prompt":              "",
        "pre_prompt":               "",
        "time_series":              [history_norm.numpy()],
        "time_series_text":         [prompt_text],
        "ts_future":                ts_future,
        "ts_already_normalized":    True,
        "time_series_norm_stats":   [(norm_mean, norm_std)],
        "norm_mean":                norm_mean,
        "norm_std":                 norm_std,
    }


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        default=(
            "results/sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D"
            "/phase1_chronos_mse_freq_mvpos/checkpoints/best_model.pt"
        ),
    )
    ap.add_argument("--llm_id",          default="google/gemma-3-270m")
    ap.add_argument("--test_split_dir",
                    default="_preprocessed_chronos_uv_synth_phase1_mse/all/test_synthetic",
                    help="Path to the HF test_synthetic split on disk")
    ap.add_argument("--n_samples",       type=int, default=2000)
    ap.add_argument("--seed",            type=int, default=42)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size",      type=int, default=8)
    ap.add_argument("--lora_r",   type=int, default=0,
                    help="LoRA rank; if >0, enable_lora() is called before loading ckpt")
    ap.add_argument("--lora_alpha", type=int, default=8)
    ap.add_argument("--filter_source", default=None,
                    help="If set, keep only rows where source == this value (e.g. 'synthetic')")
    args = ap.parse_args()

    print("=" * 70)
    print("Our Model — Synthetic Test Evaluation (direct HF cache load)")
    print("=" * 70)
    print(f"  ckpt            : {args.ckpt}")
    print(f"  test_split_dir  : {args.test_split_dir}")
    print(f"  n_samples       : {args.n_samples}  seed={args.seed}")
    print(f"  device          : {args.device}  batch_size={args.batch_size}")
    print(f"  WINDOW_LEN      : {WINDOW_LEN}  PATCH_SIZE={PATCH_SIZE}")
    print()

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"[model] loading {args.llm_id} + checkpoint …")
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
    print(f"[model] loaded OK  eos={eos_token!r}")
    print()

    # ── Load test split (already built during training) ────────────────────
    print(f"[data] loading {args.test_split_dir} …")
    if not os.path.isdir(args.test_split_dir):
        sys.exit(f"[error] test split not found at {args.test_split_dir}")
    ds = load_from_disk(args.test_split_dir, keep_in_memory=False)
    if args.filter_source is not None:
        ds = ds.filter(lambda x: x["source"] == args.filter_source)
        print(f"[data] filtered to source='{args.filter_source}': {len(ds)} rows")
    else:
        print(f"[data] {len(ds)} rows total")

    rng = random.Random(args.seed)
    n   = min(args.n_samples, len(ds))
    idxs = rng.sample(range(len(ds)), n)
    print(f"[data] sampled {n} rows (seed={args.seed})\n")

    # ── Build batch items ───────────────────────────────────────────────────
    print("[preprocess] extracting windows …")
    items   = []
    skipped = 0
    for idx in tqdm(idxs, desc="preprocessing"):
        row    = ds[idx]
        target = np.asarray(row["target"], dtype=np.float32)
        sample = _sample_test_window(target, eos_token)
        if sample is None:
            skipped += 1
            continue
        items.append(sample)

    print(f"[preprocess] {len(items)} windows, {skipped} skipped (too short)\n")

    # ── Evaluation loop ─────────────────────────────────────────────────────
    val_mse        = 0.0
    num_batches    = 0
    B              = args.batch_size
    amp_dtype      = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    with torch.no_grad(), torch.amp.autocast(
        "cuda", dtype=amp_dtype, enabled=(args.device == "cuda")
    ):
        for i in tqdm(range(0, len(items), B), desc="eval"):
            batch = items[i : i + B]
            if not batch:
                continue
            _, _, recon_loss = model.compute_pretrain_loss(batch, mse_only=True)
            val_mse     += float(recon_loss.item())
            num_batches += 1

    avg_mse = val_mse / max(1, num_batches)

    # ── Report ─────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print(f"Results  (teacher-forced next-{PATCH_SIZE}-step MSE, z-norm space)")
    print(f"  N_rows={n}  N_batches={num_batches}  batch_size={B}")
    print("=" * 70)
    print(f"  {'Our model (this ckpt)':<42s}: {avg_mse:.4f}")
    print(f"  {'TimeMoE-50M (separate eval, same split)':<42s}: 0.4198")
    print(f"  {'baseline/mean (predict 0)':<42s}: 1.0024")
    print()
    print("Checkpoint:", args.ckpt)


if __name__ == "__main__":
    main()
