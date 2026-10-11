#!/usr/bin/env python3
"""
Evaluate TimeMoE-50M on the held-out synthetic test split.

Fair-comparison protocol matching val_mse_synthetic from curriculum_learning.py:
  Data      : test_synthetic split of _preprocessed_chronos_uv_synth_phase1_mse/all
  Preprocess: per-channel z-norm (std_floor=1e-3, clip=±20) — same as
              Phase1JointDataset._sample_phase1_window
  Window    : WINDOW_LEN=2048, PATCH_SIZE=32 — same as model_config.py
  Metric    : teacher-forced next-patch MSE in z-norm space, averaged over
              ALL valid patch positions (k=0..N_patches-2) AND the ts_future
              patch beyond the window — identical to compute_pretrain_loss
  Mode      : ONE batched forward pass per chunk of channels (no generation loop)
              TimeMoE causal attention gives simultaneous predictions at all
              positions, matching the teacher-forced evaluation of our model

Usage:
    python evaluation/forecasting/timemoe_synthetic_eval.py \\
        --splits_dir _preprocessed_chronos_uv_synth_phase1_mse/all \\
        --n_samples 2000 --seed 42 --batch_size 16
"""

import argparse
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM

# ── Hyper-params that MUST match curriculum_learning.py / model_config.py ────
WINDOW_LEN = 2048   # WINDOW_LENS[0]
PATCH_SIZE  = 32    # PATCH_SIZE
# Imported rather than re-declared, so these cannot drift from training.
from opentslm.time_series_datasets.Phase1JointDataset import (  # noqa: E402
    DEFAULT_NORM_CLIP as CLIP,
    DEFAULT_NORM_STD_FLOOR as STD_FLOOR,
)


# ─── Preprocessing (mirrors Phase1JointDataset._sample_phase1_window) ─────────

def _extract_window(raw_ch: np.ndarray):
    """Extract z-normed history window + future patch for one channel.

    Matches the validation mode of _sample_phase1_window:
      * T >= WINDOW_LEN + PATCH_SIZE : use last WINDOW_LEN steps as history,
                                        next PATCH_SIZE steps as ts_future.
      * T in [2*PATCH_SIZE, WINDOW_LEN+PATCH_SIZE) : use first T-PATCH_SIZE
                                        steps as history (no padding applied
                                        here; TimeMoE handles variable length),
                                        last PATCH_SIZE as ts_future.
      * T < 2*PATCH_SIZE : skip.

    Returns (norm_win, norm_fut, real_len) or (None, None, 0) to skip.
    """
    T = len(raw_ch)
    if T < 2 * PATCH_SIZE:
        return None, None, 0

    if T >= WINDOW_LEN + PATCH_SIZE:
        start    = T - WINDOW_LEN - PATCH_SIZE
        raw_win  = raw_ch[start : start + WINDOW_LEN]
        raw_fut  = raw_ch[start + WINDOW_LEN : start + WINDOW_LEN + PATCH_SIZE]
        real_len = WINDOW_LEN
    else:
        raw_win  = raw_ch[: T - PATCH_SIZE]
        raw_fut  = raw_ch[T - PATCH_SIZE :]
        real_len = len(raw_win)

    # Z-norm on the history window (same formula as training).
    mean = float(raw_win.mean())
    std  = max(float(raw_win.std()), STD_FLOOR)
    norm_win = np.clip((raw_win - mean) / std, -CLIP, CLIP).astype(np.float32)
    norm_fut = np.clip((raw_fut - mean) / std, -CLIP, CLIP).astype(np.float32)
    return norm_win, norm_fut, real_len


def _pad_to_window(norm_win: np.ndarray, real_len: int) -> np.ndarray:
    """Pad short windows to WINDOW_LEN with the last real value (last-value fill)."""
    if real_len >= WINDOW_LEN:
        return norm_win
    pad_len = WINDOW_LEN - real_len
    pad_val = float(norm_win[real_len - 1]) if real_len > 0 else 0.0
    return np.concatenate([norm_win, np.full(pad_len, pad_val, dtype=np.float32)])


# ─── Baseline helpers ──────────────────────────────────────────────────────────

def _trivial_mses(norm_win: np.ndarray, norm_fut: np.ndarray, real_len: int):
    """Compute per-position MSE for trivial baselines (same logic as
    synthetic_nextpatch_baselines.py) and return dicts of lists."""
    acc = defaultdict(list)
    N_patches = real_len // PATCH_SIZE
    for k in range(N_patches - 1):
        cur    = norm_win[k * PATCH_SIZE : (k + 1) * PATCH_SIZE]
        target = norm_win[(k + 1) * PATCH_SIZE : (k + 2) * PATCH_SIZE]
        _add_baseline_mses(acc, cur, target)
    # ts_future
    if norm_fut is not None and len(norm_fut) == PATCH_SIZE:
        cur = norm_win[real_len - PATCH_SIZE : real_len]
        _add_baseline_mses(acc, cur, norm_fut)
    return acc


def _add_baseline_mses(acc, cur, target):
    acc["mean"].append(float(np.mean(target ** 2)))           # predict 0 (z-norm mean)
    acc["persistence"].append(float(np.mean((cur - target) ** 2)))
    last_val = np.full(PATCH_SIZE, cur[-1], dtype=np.float32)
    acc["last_value"].append(float(np.mean((last_val - target) ** 2)))
    # Linear extrapolation of cur, projected one patch forward
    t_idx  = np.arange(PATCH_SIZE, dtype=np.float32)
    t_mean = t_idx.mean()
    slope  = np.sum((t_idx - t_mean) * (cur - cur.mean())) / (np.sum((t_idx - t_mean) ** 2) + 1e-8)
    proj   = cur.mean() + slope * (t_idx + PATCH_SIZE - t_mean)
    acc["linear"].append(float(np.mean((proj - target) ** 2)))


# ─── TimeMoE forward-pass evaluation ──────────────────────────────────────────

@torch.inference_mode()
def _eval_batch(model, padded_wins, real_lens, futures, device):
    """Run one batched TimeMoE forward pass and return per-item MSE lists.

    padded_wins : list of np arrays, each shape [WINDOW_LEN] (already padded)
    real_lens   : list of int
    futures     : list of np arrays, each shape [PATCH_SIZE]
    Returns     : list of float (per-item average MSE, one entry per window)
    """
    B = len(padded_wins)
    inp = torch.tensor(
        np.stack(padded_wins), dtype=torch.float32, device=device
    ).unsqueeze(-1)   # [B, WINDOW_LEN, 1]

    # use_cache=False: DynamicCache.from_legacy_cache was removed in
    # Transformers 5.x; TimeMoE was written against 4.x.
    out   = model(input_ids=inp, max_horizon_length=PATCH_SIZE, use_cache=False)
    # logits: [B, WINDOW_LEN, PATCH_SIZE]  — prediction at position t predicts [t+1..t+PATCH_SIZE]
    preds = out.logits.cpu().numpy()

    item_mses = []
    for j in range(B):
        real_len = real_lens[j]
        win      = padded_wins[j]          # [WINDOW_LEN]
        fut      = futures[j]              # [PATCH_SIZE]
        N_patches = real_len // PATCH_SIZE

        pos_mses = []
        # In-window positions: at the end of patch k, predict patch k+1
        for k in range(N_patches - 1):
            pos    = (k + 1) * PATCH_SIZE - 1          # last timestep of patch k
            pred   = preds[j, pos, :PATCH_SIZE]         # [PATCH_SIZE]
            target = win[(k + 1) * PATCH_SIZE : (k + 2) * PATCH_SIZE]
            pos_mses.append(float(np.mean((pred - target) ** 2)))

        # ts_future: predict the patch AFTER the real window
        if fut is not None and len(fut) == PATCH_SIZE:
            pos  = real_len - 1                         # last real timestep
            pred = preds[j, pos, :PATCH_SIZE]
            pos_mses.append(float(np.mean((pred - fut) ** 2)))

        if pos_mses:
            item_mses.append(float(np.mean(pos_mses)))

    return item_mses


# ─── RoPE fix ─────────────────────────────────────────────────────────────────

def _fix_rotary_embeddings(model):
    """Recompute inv_freq and cos/sin cache for all attention layers.

    Transformers 5.x corrupts persistent=False buffers in TimeMoeRotaryEmbedding
    during from_pretrained (inv_freq is zeroed out, cos_cached gets NaN values).
    Recomputing from config values restores correct rotary embeddings.
    """
    device = next(model.parameters()).device
    for layer in model.model.layers:
        re = layer.self_attn.rotary_emb
        inv_freq = 1.0 / (re.base ** (
            torch.arange(0, re.dim, 2, dtype=torch.float32, device=device) / re.dim
        ))
        re.inv_freq = inv_freq
        re._set_cos_sin_cache(
            seq_len=re.max_position_embeddings,
            device=device,
            dtype=torch.float32,
        )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--splits_dir",
        default=os.environ.get(
            "FTS_SPLITS_DIR",
            os.path.join(
                os.environ.get("CHRONOS_PREPROCESSED_SPLITS_DIR",
                               "_preprocessed_chronos_uv_synth_phase1_mse"),
                "all",
            ),
        ),
        help="Path to the 'all' directory that contains test_synthetic/",
    )
    ap.add_argument("--n_samples",  type=int, default=2000,
        help="Number of rows to sample from test_synthetic (seed-stable)")
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=16,
        help="Number of padded windows per TimeMoE forward pass")
    ap.add_argument("--model_id",   default="Maple728/TimeMoE-50M",
        help="HuggingFace model ID for TimeMoE")
    args = ap.parse_args()

    print("=" * 70)
    print("TimeMoE Synthetic Evaluation — fair comparison with val_mse_synthetic")
    print("=" * 70)
    print(f"  splits_dir  : {args.splits_dir}")
    print(f"  n_samples   : {args.n_samples}  seed={args.seed}")
    print(f"  device      : {args.device}  batch_size={args.batch_size}")
    print(f"  model_id    : {args.model_id}")
    print(f"  WINDOW_LEN  : {WINDOW_LEN}  PATCH_SIZE={PATCH_SIZE}")
    print(f"  STD_FLOOR   : {STD_FLOOR}   CLIP={CLIP}")
    print()

    # ── Load data ──────────────────────────────────────────────────────────
    test_dir = os.path.join(args.splits_dir, "test_synthetic")
    if not os.path.isdir(test_dir):
        sys.exit(f"[error] test_synthetic split not found at {test_dir}\n"
                 "        Run Phase1JointDataset to build the cache first.")
    print(f"[data] loading {test_dir} …")
    ds = load_from_disk(test_dir, keep_in_memory=False)
    print(f"[data] {len(ds)} rows total")

    rng  = random.Random(args.seed)
    n    = min(args.n_samples, len(ds))
    idxs = rng.sample(range(len(ds)), n)
    print(f"[data] sampled {n} rows (seed={args.seed})\n")

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"[model] loading {args.model_id} …")
    # Load in float32. Bfloat16 produces NaN on V100 for 2048-step sequences.
    model = (
        AutoModelForCausalLM
        .from_pretrained(args.model_id, trust_remote_code=True, torch_dtype=torch.float32)
        .to(args.device)
        .eval()
    )
    # Transformers 5.x corrupts persistent=False RoPE buffers (inv_freq, cos/sin cache)
    # during from_pretrained. Recompute from scratch using the config values.
    _fix_rotary_embeddings(model)
    print(f"[model] horizon_lengths : {model.config.horizon_lengths}")
    print(f"[model] max_pos_emb     : {model.config.max_position_embeddings}")
    assert PATCH_SIZE in model.config.horizon_lengths, (
        f"PATCH_SIZE={PATCH_SIZE} not in model.config.horizon_lengths="
        f"{model.config.horizon_lengths}"
    )
    print()

    # ── Collect all channel windows ────────────────────────────────────────
    all_wins      = []   # padded to WINDOW_LEN
    all_real_lens = []
    all_futs      = []
    all_raw_wins  = []   # unpadded, for baselines
    n_skipped     = 0

    for idx in tqdm(idxs, desc="preprocessing"):
        row    = ds[idx]
        target = np.asarray(row["target"], dtype=np.float32)
        if target.ndim == 1:
            target = target[:, None]
        T, C = target.shape

        for c in range(C):
            norm_win, norm_fut, real_len = _extract_window(target[:, c])
            if norm_win is None:
                n_skipped += 1
                continue
            padded = _pad_to_window(norm_win, real_len)
            all_wins.append(padded)
            all_real_lens.append(real_len)
            all_futs.append(norm_fut)
            all_raw_wins.append(norm_win)   # original (unpadded)

    print(f"\n[preprocess] total channel-windows : {len(all_wins)}")
    print(f"[preprocess] skipped (too short)   : {n_skipped}")

    # ── TimeMoE batched evaluation ─────────────────────────────────────────
    timemoe_mse_all = []
    B = args.batch_size

    for i in tqdm(range(0, len(all_wins), B), desc="TimeMoE eval"):
        chunk_wins  = all_wins[i : i + B]
        chunk_rlens = all_real_lens[i : i + B]
        chunk_futs  = all_futs[i : i + B]
        item_mses   = _eval_batch(model, chunk_wins, chunk_rlens, chunk_futs, args.device)
        timemoe_mse_all.extend(item_mses)

    # ── Trivial baselines ──────────────────────────────────────────────────
    baseline_all = defaultdict(list)
    for raw_win, norm_fut, real_len in zip(all_raw_wins, all_futs, all_real_lens):
        acc = _trivial_mses(raw_win, norm_fut, real_len)
        for k, v in acc.items():
            baseline_all[k].extend(v)

    # ── Report ─────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print(f"Results  (teacher-forced next-{PATCH_SIZE}-step MSE, z-norm space)")
    print(f"  Window={WINDOW_LEN}  Patch={PATCH_SIZE}  N_windows={len(timemoe_mse_all)}")
    print("=" * 70)

    tmse = float(np.mean(timemoe_mse_all)) if timemoe_mse_all else float("nan")
    print(f"  {'TimeMoE-50M (ours, teacher-forced)':<42s}: {tmse:.4f}")
    for name in ("mean", "persistence", "last_value", "linear"):
        vals = baseline_all.get(name, [])
        v    = float(np.mean(vals)) if vals else float("nan")
        print(f"  {'baseline/' + name:<42s}: {v:.4f}")
    print(f"  {'our model (val_mse_synthetic, reported)':<42s}: ~0.35")
    print()
    print("Interpretation:")
    print("  * 'mean' baseline predicts 0 (z-norm mean) → MSE ≈ variance ≈ 1.0")
    print("  * 'persistence' repeats the last patch → captures local smoothness")
    print("  * Lower MSE is better.")
    print("  * Both TimeMoE and our model use the same data, normalization, and")
    print("    teacher-forced evaluation protocol, making the comparison direct.")


if __name__ == "__main__":
    main()
