#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deakin University and the LOTUS authors
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT
"""Memory vs total input length (N x L), one panel per backbone.

Companion to plot_memory_scaling.py (same CSV, same OOM convention) that
collapses the (L, N) grid onto the total number of time steps the encoder
sees, N x L, so the curves of the model families (LOTUS, SoftPrompt,
Flamingo) can be read against one axis. Restored from OpenTSLM's
evaluation/memory/plot_memory_simulation.py with the panel list derived from
the CSV instead of hard-coded, and a CLI.

Usage:
    python scripts/analysis/memory/plot_memory_total_length.py \
        --csv $LOTUS_RESULTS_ROOT/memory/memory_use.csv --out memory_total_length
"""

import argparse

import matplotlib
import matplotlib.pyplot as plt

from plot_memory_scaling import FAMILY_ORDER, MODEL_ORDER, load

FAMILY_COLOR = {"LOTUS": "#228833", "SoftPrompt": "#4477AA", "Flamingo": "#CC6677"}


def plot_total_length(csv_file: str, out_prefix: str, oom_threshold: float, show: bool = False):
    plt.style.use("seaborn-v0_8-white")
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Palatino", "Times New Roman", "DejaVu Serif"],
        "font.size": 14, "axes.labelsize": 16, "axes.titlesize": 16,
        "legend.fontsize": 12, "xtick.labelsize": 13, "ytick.labelsize": 13,
        "axes.linewidth": 0.6, "axes.edgecolor": "0.15",
    })
    df = load(csv_file, oom_threshold)
    if df.empty:
        raise SystemExit(f"no Simulation-L*-N* rows in {csv_file}")
    df["total_length"] = df["L"] * df["N"]

    base_models = [m for m in MODEL_ORDER if m in set(df["base_model"])] + \
                  sorted(set(df["base_model"]) - set(MODEL_ORDER))
    families = [f for f in FAMILY_ORDER if f in set(df["family"])] + \
               sorted(set(df["family"]) - set(FAMILY_ORDER))

    fig, axes = plt.subplots(1, len(base_models), figsize=(3.4 * len(base_models), 3.2),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, bm in zip(axes, base_models):
        sub = df[df["base_model"] == bm]
        for fam in families:
            fdf = sub[sub["family"] == fam].sort_values("total_length")
            if fdf.empty:
                continue
            color = FAMILY_COLOR.get(fam, "#888888")
            ok = fdf[~fdf["oom"]]
            oom = fdf[fdf["oom"]]
            if not ok.empty:
                ax.plot(ok["total_length"], ok["peak_gb"], label=fam, color=color,
                        linewidth=3.0, marker="o", markersize=4, alpha=0.9)
            if not oom.empty:
                first = oom.iloc[0]
                y_top = (float(ok["peak_gb"].max()) if not ok.empty else 1.0) * 1.15 + 1.0
                x0 = ok["total_length"].iloc[-1] if not ok.empty else first["total_length"]
                y0 = ok["peak_gb"].iloc[-1] if not ok.empty else 0.0
                ax.plot([x0, first["total_length"]], [y0, y_top], color=color,
                        linestyle="--", linewidth=2.0, alpha=0.8)
                ax.scatter(first["total_length"], y_top, color="red", marker="x",
                           s=80, linewidth=3, zorder=5)
                ax.text(first["total_length"], y_top, "OOM", color="red", fontsize=9,
                        fontweight="bold", ha="center", va="bottom")
        ax.set_title(bm, fontweight="bold")
        ax.set_xscale("log")
        ax.set_xlabel("Total input length N x L")
        ax.set_facecolor("#F8F9FA")
        ax.grid(True, which="major", linestyle="-", linewidth=0.4, alpha=0.5)
        ax.minorticks_on()
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Peak CUDA reserved memory (GB)", fontweight="bold")
    axes[0].legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="0.3")
    plt.tight_layout(pad=0.6)
    for fmt in ("png", "pdf"):
        fig.savefig(f"{out_prefix}.{fmt}", dpi=300 if fmt == "png" else None,
                    bbox_inches="tight", pad_inches=0.02, facecolor="white", format=fmt)
        print(f"wrote {out_prefix}.{fmt}")
    if show:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--csv", default="memory_use.csv")
    ap.add_argument("--out", default="memory_total_length")
    ap.add_argument("--oom_threshold", type=float, default=180.0)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    plot_total_length(args.csv, args.out, args.oom_threshold, args.show)


if __name__ == "__main__":
    main()
