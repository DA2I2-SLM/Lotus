"""Draw raw synthetic samples in the same visual style as run_autoregressive's
_save_plot, so they can be compared side-by-side with model predictions.

Picks N (default 100) random rows from the synthetic corpus dir ($FTS_SYNTHETIC_DIR)
unified_*.arrow, splits each into a (history_len)-step history + (pred_len)-step
future window using the same per-channel z-norm + std-floor + value-clip rules
as evaluation/forecasting/fixed_inference.py, and writes one PNG per (sample, channel).

Output:
    <out_dir>/univariate/raw_step{N:06d}_ch0.png
    <out_dir>/multivariate/raw_step{N:06d}_ch{c}.png

There is no "predicted" curve — the second/third panels show GT patches only.
Caps each modality directory at --max_plot_samples (default 100).

Run:
    python evaluation/forecasting/draw_raw_synthetic.py \
        --synthetic_data_dir $FTS_SYNTHETIC_DIR \
        --history_len 576 --pred_len 192 --patch_size 32 \
        --out_dir ./eval_results/mlp_phase1_chronos_mse_freq/time_series/synthetic_raw \
        --n_samples 100 --seed 42
"""
import argparse
import glob
import os
import random
from typing import List, Tuple

import numpy as np
import pyarrow as pa

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


_STD_FLOOR = 1e-3
_CLIP = 20.0


def _list_arrow_files(synthetic_dir: str) -> List[str]:
    paths = sorted(glob.glob(os.path.join(synthetic_dir, "unified_*.arrow")))
    if not paths:
        raise FileNotFoundError(f"No unified_*.arrow under {synthetic_dir}")
    return paths


def _read_one_row(arrow_path: str, row_idx: int) -> np.ndarray:
    """Return the row's `target` as float32. Native layout is [C, L]
    (collect_synthetic.py contract); the merger guarantees this.
    Transpose to [L, C] for downstream slicing convenience.
    """
    with pa.OSFile(arrow_path, "rb") as src:
        reader = pa.ipc.open_file(src)
        # Locate the batch containing row_idx.
        cum = 0
        for i in range(reader.num_record_batches):
            n = reader.get_batch(i).num_rows
            if row_idx < cum + n:
                batch = reader.get_batch(i)
                rel = row_idx - cum
                row = batch.column("target")[rel].as_py()  # list of lists or list
                arr = np.asarray(row, dtype=np.float32)    # [C, L] for MV / [L] for UV
                if arr.ndim == 1:
                    arr = arr[None, :]                     # treat UV as [1, L]
                return arr.T                                # [L, C]
            cum += n
        raise IndexError(f"row {row_idx} out of range for {arrow_path}")


def _row_counts(synthetic_dir: str) -> List[Tuple[str, int]]:
    """Return [(path, n_rows)] for each unified_*.arrow."""
    out = []
    for p in _list_arrow_files(synthetic_dir):
        with pa.OSFile(p, "rb") as src:
            reader = pa.ipc.open_file(src)
            n = sum(reader.get_batch(i).num_rows for i in range(reader.num_record_batches))
        out.append((p, n))
    return out


def _pick_random_rows(synthetic_dir: str, n_samples: int, seed: int) -> List[Tuple[str, int]]:
    """Globally-random selection of (path, row_idx) pairs across all unified_*.arrow."""
    counts = _row_counts(synthetic_dir)
    total = sum(n for _, n in counts)
    if n_samples > total:
        n_samples = total
    rng = random.Random(seed)
    flat_indices = rng.sample(range(total), n_samples)
    picks: List[Tuple[str, int]] = []
    for fi in flat_indices:
        offset = 0
        for path, n in counts:
            if fi < offset + n:
                picks.append((path, fi - offset))
                break
            offset += n
    return picks


def _normalize_history_and_future(history_raw: np.ndarray, future_raw: np.ndarray):
    """Per-channel z-norm computed on HISTORY only (matches fixed_inference.py)."""
    mean = history_raw.mean(axis=0, keepdims=True)
    std = history_raw.std(axis=0, keepdims=True)
    std = np.maximum(std, _STD_FLOOR)
    history_norm = (history_raw - mean) / std
    future_norm = (future_raw - mean) / std
    history_norm = np.clip(history_norm, -_CLIP, _CLIP)
    future_norm = np.clip(future_norm, -_CLIP, _CLIP)
    return history_norm, future_norm, float(mean[0, 0]), float(std[0, 0])


def _draw_one_sample(
    series: np.ndarray,       # [L, C] raw
    history_len: int,
    pred_len: int,
    patch_size: int,
    out_dir_uv: str,
    out_dir_mv: str,
    sample_idx: int,
    uv_counter: List[int],
    mv_counter: List[int],
    max_per_modality: int,
):
    T = series.shape[0]
    if T < history_len + pred_len:
        return  # series too short; skip

    # Pick the LATEST window (last `pred_len` steps as future) to match eval semantics.
    start = T - history_len - pred_len
    history_raw = series[start: start + history_len]                            # [H, C]
    future_raw  = series[start + history_len: start + history_len + pred_len]   # [P, C]

    history_norm, future_norm, m0, s0 = _normalize_history_and_future(history_raw, future_raw)

    n_ch = series.shape[1]
    modality = "univariate" if n_ch == 1 else "multivariate"
    out_dir = out_dir_uv if modality == "univariate" else out_dir_mv

    counter = uv_counter if modality == "univariate" else mv_counter
    if counter[0] >= max_per_modality:
        return
    counter[0] += 1

    os.makedirs(out_dir, exist_ok=True)

    hist_len = history_norm.shape[0]
    L = future_norm.shape[0]
    hist_x = np.arange(hist_len)
    fore_x = np.arange(hist_len, hist_len + L)
    fore_rel = np.arange(L)
    n_patches = (L + patch_size - 1) // patch_size

    def _draw_patch_bg(ax):
        for p in range(n_patches):
            s, e = p * patch_size, min((p + 1) * patch_size, L)
            if p % 2 == 0:
                ax.axvspan(s, e, alpha=0.06, color="grey")
        if patch_size > 1:
            for t in range(0, L, patch_size):
                ax.axvline(t, color="lightgrey", ls=":", lw=0.7)

    for c in range(n_ch):
        hist_c = history_norm[:, c]
        gt_c   = future_norm[:, c]

        fig, axes = plt.subplots(2, 1, figsize=(13, 7),
                                 gridspec_kw={"height_ratios": [2.5, 1]})
        fig.suptitle(
            f"raw_synthetic  |  ch{c}  |  sample={sample_idx}  |  "
            f"L={T}  raw_mean(ch0)={m0:.4f}  raw_std(ch0)={s0:.4f}",
            fontsize=11,
        )

        # Panel 1 — history + future (no prediction)
        ax = axes[0]
        ax.plot(hist_x, hist_c, color="steelblue", lw=1.2, label="history")
        ax.axvline(hist_len, color="grey", ls="--", lw=0.8, label="forecast start")
        ax.plot(fore_x, gt_c, color="seagreen", lw=1.5, label="ground truth")
        if patch_size > 1:
            for t in range(hist_len, hist_len + L, patch_size):
                ax.axvline(t, color="lightgrey", ls=":", lw=0.5)
        ax.set_ylabel("value (normalised)")
        ax.set_title(f"Channel {c} — Context + Forecast (raw, no prediction)", fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

        # Panel 2 — ground truth patches (forecast region)
        ax2 = axes[1]
        _draw_patch_bg(ax2)
        ax2.plot(fore_rel, gt_c, color="seagreen", lw=1.5, label="ground truth")
        ax2.set_xlabel("forecast timestep")
        ax2.set_ylabel("value (normalised)")
        ax2.set_title("Ground Truth Patches (raw)", fontsize=9)
        ax2.legend(fontsize=8, loc="upper left")

        fig.tight_layout()
        fname = os.path.join(out_dir, f"raw_step{sample_idx:06d}_ch{c}.png")
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Plot raw synthetic samples in eval style.")
    ap.add_argument("--synthetic_data_dir",
                    default=os.environ.get("FTS_SYNTHETIC_DIR", ""),
                    help="Directory containing unified_*.arrow files.")
    ap.add_argument("--out_dir",
                    default="./eval_results/raw_synthetic_plots",
                    help="Output dir; will create univariate/ and multivariate/ subdirs.")
    ap.add_argument("--n_samples", type=int, default=100,
                    help="Number of random rows to attempt (some may be skipped if T < H+P).")
    ap.add_argument("--history_len", type=int, default=576)
    ap.add_argument("--pred_len",    type=int, default=192)
    ap.add_argument("--patch_size",  type=int, default=32)
    ap.add_argument("--max_plot_samples", type=int, default=100,
                    help="Cap on plots per modality (univariate / multivariate).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"[raw-synthetic] scanning {args.synthetic_data_dir} …")
    picks = _pick_random_rows(args.synthetic_data_dir, args.n_samples, args.seed)
    print(f"[raw-synthetic] picked {len(picks)} random rows from "
          f"{len(set(p for p, _ in picks))} files")

    out_dir_uv = os.path.join(args.out_dir, "univariate")
    out_dir_mv = os.path.join(args.out_dir, "multivariate")
    uv_counter = [0]
    mv_counter = [0]

    for i, (path, row_idx) in enumerate(picks):
        try:
            arr = _read_one_row(path, row_idx)  # [L, C]
        except Exception as exc:
            print(f"  [skip] {os.path.basename(path)}#{row_idx}: {exc}")
            continue
        _draw_one_sample(
            arr,
            history_len=args.history_len,
            pred_len=args.pred_len,
            patch_size=args.patch_size,
            out_dir_uv=out_dir_uv,
            out_dir_mv=out_dir_mv,
            sample_idx=i,
            uv_counter=uv_counter,
            mv_counter=mv_counter,
            max_per_modality=args.max_plot_samples,
        )
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(picks)}] uv={uv_counter[0]}  mv={mv_counter[0]}")
        if uv_counter[0] >= args.max_plot_samples and mv_counter[0] >= args.max_plot_samples:
            print("[raw-synthetic] both modality caps reached; stopping.")
            break

    print(f"[raw-synthetic] done — univariate={uv_counter[0]} files in {out_dir_uv}")
    print(f"[raw-synthetic] done — multivariate={mv_counter[0]} files in {out_dir_mv}")


if __name__ == "__main__":
    main()
