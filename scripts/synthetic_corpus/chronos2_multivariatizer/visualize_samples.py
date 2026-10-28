"""Render N sample time series from a GluonTS Arrow file to per-sample PNGs.

Auto-detects schema:
  - univariate:        item["target"] is 1D                 -> one line per PNG.
  - multivariate:      item["target"] is 2D (D, L)          -> one subplot per channel.
  - covariate-informed: item has "feat_dynamic_real" (D_cov, L)
                       -> target channels (solid) + covariates (dashed, separate panels).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
from gluonts.dataset.arrow import ArrowFile


def _plot_univariate(target: np.ndarray, ax) -> None:
    ax.plot(target, linewidth=0.8)
    ax.set_xlabel("t")
    ax.set_ylabel("value")
    ax.grid(alpha=0.3)


def _plot_panels(target: np.ndarray, covs: np.ndarray | None, out_path: Path, title: str) -> None:
    """Multi-channel plot. `target` is (D_target, L). `covs` is (D_cov, L) or None."""
    d_t = target.shape[0]
    d_c = 0 if covs is None else covs.shape[0]
    n_rows = d_t + d_c
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(10, 1.4 * n_rows + 0.4), sharex=True, squeeze=False
    )
    axes = axes[:, 0]

    for i in range(d_t):
        axes[i].plot(target[i], linewidth=0.8, color="C0")
        axes[i].set_ylabel(f"y[{i}]")
        axes[i].grid(alpha=0.3)
    for j in range(d_c):
        ax = axes[d_t + j]
        ax.plot(covs[j], linewidth=0.8, color="C3", linestyle="--")
        ax.set_ylabel(f"cov[{j}]")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("t")
    axes[0].set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def visualize(arrow_path: Path, out_dir: Path, n_samples: int, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    items = list(ArrowFile(str(arrow_path)))
    if len(items) == 0:
        print(f"[viz] {arrow_path}: empty, skipping")
        return

    n = min(n_samples, len(items))
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(items), size=n, replace=False).tolist()

    for k, idx in enumerate(idxs):
        item = items[idx]
        target = np.asarray(item["target"])
        cov = item.get("feat_dynamic_real")
        if cov is not None:
            cov = np.asarray(cov)

        out_path = out_dir / f"sample_{k:02d}_idx{idx:06d}.png"

        if target.ndim == 1:
            fig, ax = plt.subplots(1, 1, figsize=(10, 2.2))
            _plot_univariate(target, ax)
            ax.set_title(f"{arrow_path.name} | item #{idx}", fontsize=10)
            fig.tight_layout()
            fig.savefig(out_path, dpi=120)
            plt.close(fig)
        else:
            # target is (D, L)
            d_t = target.shape[0]
            d_c = 0 if cov is None else cov.shape[0]
            title = (
                f"{arrow_path.name} | item #{idx} | "
                f"D_target={d_t}" + (f", D_cov={d_c}" if cov is not None else "")
            )
            _plot_panels(target, cov, out_path, title)

    print(f"[viz] wrote {n} PNGs to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arrow", type=Path, required=True, help="Input Arrow file")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for PNGs")
    parser.add_argument("-n", "--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    visualize(args.arrow, args.out_dir, args.num_samples, args.seed)


if __name__ == "__main__":
    main()
