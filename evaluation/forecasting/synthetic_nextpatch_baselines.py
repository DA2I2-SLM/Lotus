"""Compute next-patch MSE baselines on the synthetic corpus to find the
intrinsic predictability floor.

The training objective is shifted next-patch MSE: at each patch position t,
predict patch t+1. This script computes that MSE for trivial baselines on the
SAME per-channel z-normalised window the model sees, so we can tell whether the
model's synthetic_mse≈0.6 is near the data's intrinsic limit or there's
headroom.

Baselines (all in per-channel z-norm space, averaged over valid patch positions):
  * mean        : predict patch[t+1] = 0  (z-norm mean)   → MSE ≈ Var ≈ 1.0
  * persistence : predict patch[t+1] = patch[t]            (repeat last patch)
  * last_value  : predict patch[t+1] = last value of patch[t], repeated P times
  * linear      : per-patch linear extrapolation of the last patch

Run:
    python evaluation/forecasting/synthetic_nextpatch_baselines.py \
        --synthetic_data_dir $LOTUS_SYNTHETIC_DIR \
        --n_samples 2000 --window_len 2048 --patch_size 32 --seed 42
"""
import argparse
import glob
import os
import random
from collections import defaultdict

import numpy as np
import pyarrow as pa

_STD_FLOOR = 1e-3
_CLIP = 20.0


def _list_arrow_files(d):
    paths = sorted(glob.glob(os.path.join(d, "unified_*.arrow")))
    if not paths:
        raise FileNotFoundError(f"No unified_*.arrow under {d}")
    return paths


def _row_counts(d):
    out = []
    for p in _list_arrow_files(d):
        with pa.OSFile(p, "rb") as src:
            r = pa.ipc.open_file(src)
            n = sum(r.get_batch(i).num_rows for i in range(r.num_record_batches))
        out.append((p, n))
    return out


def _read_row(path, idx):
    with pa.OSFile(path, "rb") as src:
        reader = pa.ipc.open_file(src)
        cum = 0
        for i in range(reader.num_record_batches):
            b = reader.get_batch(i)
            if idx < cum + b.num_rows:
                row = b.column("target")[idx - cum].as_py()
                arr = np.asarray(row, dtype=np.float32)   # [C, L]
                if arr.ndim == 1:
                    arr = arr[None, :]
                return arr.T                               # [L, C]
            cum += b.num_rows
    raise IndexError(idx)


def _source_of(path):
    # unified_<source>.arrow → <source>
    stem = os.path.basename(path)[len("unified_"):-len(".arrow")]
    return stem


def _patchify(window, patch_size):
    """window: [T, C] → [n_patches, patch_size, C]."""
    T, C = window.shape
    n_p = T // patch_size
    return window[: n_p * patch_size].reshape(n_p, patch_size, C)


def _eval_sample(series, window_len, patch_size, acc):
    T = series.shape[0]
    if T < window_len + patch_size:
        return
    start = T - window_len - patch_size
    window = series[start: start + window_len]                       # [W, C]
    # per-channel z-norm on the window (history)
    mean = window.mean(axis=0, keepdims=True)
    std = np.maximum(window.std(axis=0, keepdims=True), _STD_FLOOR)
    wn = np.clip((window - mean) / std, -_CLIP, _CLIP)              # [W, C]

    patches = _patchify(wn, patch_size)                             # [n_p, P, C]
    n_p = patches.shape[0]
    if n_p < 2:
        return

    # next-patch targets: patch[t+1] for t in 0..n_p-2
    tgt = patches[1:]            # [n_p-1, P, C]
    cur = patches[:-1]          # [n_p-1, P, C]

    # baselines
    pred_mean = np.zeros_like(tgt)
    pred_persist = cur                                             # repeat last patch
    pred_lastval = np.repeat(cur[:, -1:, :], patch_size, axis=1)  # last value of cur, repeated
    # linear extrapolation: fit slope from last patch, project forward
    t_idx = np.arange(patch_size)[None, :, None]
    cur_mean = cur.mean(axis=1, keepdims=True)
    t_mean = t_idx.mean()
    slope = ((t_idx - t_mean) * (cur - cur_mean)).sum(axis=1, keepdims=True) / \
            (((t_idx - t_mean) ** 2).sum() + 1e-8)
    # project: continue the line into the next patch window
    proj_t = np.arange(patch_size, 2 * patch_size)[None, :, None]
    pred_linear = cur_mean + slope * (proj_t - t_mean)

    for name, pred in [
        ("mean", pred_mean),
        ("persistence", pred_persist),
        ("last_value", pred_lastval),
        ("linear", pred_linear),
    ]:
        mse = float(np.mean((pred - tgt) ** 2))
        acc[name].append(mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic_data_dir",
                    default=os.environ.get("LOTUS_SYNTHETIC_DIR", ""))
    ap.add_argument("--n_samples", type=int, default=2000)
    ap.add_argument("--window_len", type=int, default=2048)
    ap.add_argument("--patch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    counts = _row_counts(args.synthetic_data_dir)
    total = sum(n for _, n in counts)
    n = min(args.n_samples, total)
    rng = random.Random(args.seed)
    picks = rng.sample(range(total), n)

    # global acc + per-source acc
    acc = defaultdict(list)
    per_source = defaultdict(lambda: defaultdict(list))
    n_used = 0
    n_skipped = 0
    for fi in picks:
        off = 0
        for path, cnt in counts:
            if fi < off + cnt:
                row_idx = fi - off
                src = _source_of(path)
                break
            off += cnt
        try:
            series = _read_row(path, row_idx)
        except Exception:
            n_skipped += 1
            continue
        before = len(acc["mean"])
        _eval_sample(series, args.window_len, args.patch_size, acc)
        if len(acc["mean"]) > before:
            n_used += 1
            # also accumulate per-source
            _eval_sample(series, args.window_len, args.patch_size, per_source[src])
        else:
            n_skipped += 1

    print(f"\n[baselines] samples used={n_used}  skipped(too short)={n_skipped}  "
          f"window_len={args.window_len}  patch_size={args.patch_size}")
    print("\n=== GLOBAL next-patch MSE (per-channel z-norm space) ===")
    for name in ("mean", "persistence", "last_value", "linear"):
        vals = acc[name]
        if vals:
            print(f"  {name:12s}: {np.mean(vals):.4f}  (median {np.median(vals):.4f}, n={len(vals)})")

    print("\n=== PER-SOURCE persistence vs mean ===")
    for src in sorted(per_source):
        m = np.mean(per_source[src]["mean"]) if per_source[src]["mean"] else float("nan")
        p = np.mean(per_source[src]["persistence"]) if per_source[src]["persistence"] else float("nan")
        lv = np.mean(per_source[src]["last_value"]) if per_source[src]["last_value"] else float("nan")
        nrows = len(per_source[src]["mean"])
        print(f"  {src:32s}: mean={m:.3f}  persist={p:.3f}  last_val={lv:.3f}  (n={nrows})")

    print("\nInterpretation:")
    print("  * model synthetic_mse ≈ 0.6. If 'persistence' or 'linear' baseline is")
    print("    also ≈0.6, the data's intrinsic next-patch predictability floor is ~0.6")
    print("    and the model is already near-optimal. If baselines are >>0.6, the")
    print("    model has learned something; if <<0.6, there is headroom / a bug.")


if __name__ == "__main__":
    main()
