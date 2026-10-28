#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deakin University and the LOTUS authors
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT
"""Memory-scaling plot (paper Fig. 7 / Appendix "Computational Cost").

Reads the CSV written by get_memory_use.py / run_all_memory.sh and plots peak
CUDA reserved memory against the number of series N, one column per series
length L and one row per model family:

- Rows   = model family (LOTUS = OpenTSLMSPFixPos2D, SoftPrompt = OpenTSLMSP,
           Flamingo = OpenTSLMFlamingo) -- only the families present in the CSV
- Cols   = sequence lengths L (L = 1 excluded)
- Hue    = backbone LLM
- OOM    = rows with status != ok, or peak above --oom_threshold GB, are drawn
           as a dashed extension to a red X marked "OOM"

Provenance: OpenTSLM's evaluation/memory/plot_memory_scaling.py was truncated
mid-statement in every branch of this fork (and upstream), so this is the
complete sibling evaluation/memory/plot_memory_simulation_per_length.py with
the hard-coded SoftPrompt/Flamingo + Llama assumptions generalised and a CLI.

Usage:
    python scripts/analysis/memory/plot_memory_scaling.py \
        --csv $FTS_RESULTS_ROOT/memory/memory_use.csv --out memory_scaling
"""

import argparse
import re

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

FAMILY = {
    "OpenTSLMSPFixPos2D": "LOTUS",
    "OpenTSLMSP": "SoftPrompt",
    "OpenTSLMFlamingo": "Flamingo",
}
FAMILY_ORDER = ["LOTUS", "SoftPrompt", "Flamingo"]
MODEL_ORDER = ["Gemma-3-270M", "Qwen3-0.6B", "Gemma-3-1B-pt", "Llama-3.2-1B", "Llama-3.2-3B"]
COLOR_MAP = {
    "Gemma-3-270M": "#4477AA",
    "Qwen3-0.6B": "#EE6677",
    "Gemma-3-1B-pt": "#66CCEE",
    "Llama-3.2-1B": "#228833",
    "Llama-3.2-3B": "#CC6677",
}
MARKERS = ["o", "s", "^", "D", "p", "X", "*"]


def parse_model_name(llm_id: str, model_type: str):
    """Return (backbone display name, model family)."""
    base = llm_id.split("/")[-1]
    low = base.lower()
    if "llama-3.2-1b" in low:
        base = "Llama-3.2-1B"
    elif "llama-3.2-3b" in low:
        base = "Llama-3.2-3B"
    elif "gemma-3-270m" in low:
        base = "Gemma-3-270M"
    elif "gemma-3-1b" in low:
        base = "Gemma-3-1B-pt"
    elif "qwen3-0.6b" in low:
        base = "Qwen3-0.6B"
    return base, FAMILY.get(model_type, model_type)


def parse_simulation_dataset(name: str):
    """'Simulation-L10-N5' -> (10, 5)."""
    m = re.match(r"Simulation-L(\d+)-N(\d+)", str(name))
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def load(csv_file: str, oom_threshold: float) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    df = df[df["dataset"].astype(str).str.startswith("Simulation-")].copy()
    df["peak_gb"] = pd.to_numeric(df["peak_cuda_reserved_gb"], errors="coerce")
    df.loc[df["peak_gb"] < 0, "peak_gb"] = pd.NA
    if "status" not in df.columns:
        df["status"] = "ok"
    df["oom"] = (df["status"].astype(str) != "ok") | df["peak_gb"].isna() | (df["peak_gb"] > oom_threshold)
    df[["base_model", "family"]] = df.apply(
        lambda r: pd.Series(parse_model_name(str(r["llm_id"]), str(r["model"]))), axis=1)
    df[["L", "N"]] = df["dataset"].apply(lambda s: pd.Series(parse_simulation_dataset(s)))
    df = df.dropna(subset=["L", "N"])
    df["L"] = df["L"].astype(int)
    df["N"] = df["N"].astype(int)
    df = df[df["L"] != 1]
    # keep the last measurement of each cell
    df = df.sort_values("timestamp").groupby(["family", "base_model", "L", "N"], as_index=False).last()
    return df.sort_values(["family", "base_model", "L", "N"])


def plot_memory_scaling(csv_file: str, out_prefix: str, oom_threshold: float, show: bool = False):
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

    families = [f for f in FAMILY_ORDER if f in set(df["family"])] + \
               sorted(set(df["family"]) - set(FAMILY_ORDER))
    base_models = [m for m in MODEL_ORDER if m in set(df["base_model"])] + \
                  sorted(set(df["base_model"]) - set(MODEL_ORDER))
    markers = dict(zip(base_models, MARKERS * 3))
    unique_L = sorted(df["L"].unique())

    fig, axes = plt.subplots(len(families), len(unique_L),
                             figsize=(3.2 * len(unique_L), 3.0 * len(families)),
                             sharex="col", squeeze=False)

    for i, fam in enumerate(families):
        fam_df = df[df["family"] == fam]
        for j, L in enumerate(unique_L):
            ax = axes[i, j]
            sub = fam_df[fam_df["L"] == L]
            ok_all = sub[~sub["oom"]]
            ymax = float(ok_all["peak_gb"].max()) if not ok_all.empty else 1.0
            oom_y = ymax * 1.15 + 1.0
            for bm in base_models:
                sdf = sub[sub["base_model"] == bm].sort_values("N")
                if sdf.empty:
                    continue
                color = COLOR_MAP.get(bm, "#888888")
                ok = sdf[~sdf["oom"]]
                oom = sdf[sdf["oom"]]
                if not ok.empty:
                    ax.plot(ok["N"], ok["peak_gb"], label=bm, color=color,
                            marker=markers[bm], linewidth=2.2, markersize=5, alpha=0.9)
                if not oom.empty:
                    first = oom.iloc[0]
                    x0 = ok["N"].iloc[-1] if not ok.empty else first["N"]
                    y0 = ok["peak_gb"].iloc[-1] if not ok.empty else 0.0
                    ax.plot([x0, first["N"]], [y0, oom_y], color=color,
                            linestyle="--", linewidth=1.5, alpha=0.8)
                    ax.scatter(first["N"], oom_y, color="red", marker="x", s=70,
                               linewidth=2, zorder=5)
                    ax.text(first["N"], oom_y, "OOM", color="red", fontsize=9,
                            fontweight="bold", ha="center", va="bottom")
            if i == 0:
                ax.set_title(f"L = {L}", fontweight="bold")
            if i == len(families) - 1:
                ax.set_xlabel("Number of series N")
            ax.set_ylabel(fam if j == 0 else "", fontweight="bold")
            ax.set_facecolor("#F8F9FA")
            ax.grid(True, which="major", linestyle="-", linewidth=0.4, alpha=0.5)
            ax.minorticks_on()
            ax.set_xticks(sorted(df["N"].unique()))
            ax.tick_params(axis="x", which="both", labelbottom=True)
            ax.set_ylim(bottom=0)

    # Global legend: backbones in order + OOM marker
    handles = {}
    for ax in axes.ravel():
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    ordered = [(handles[bm], bm) for bm in base_models if bm in handles]
    ordered.append((Line2D([0], [0], color="red", marker="x", linestyle="--",
                           markersize=8), "Out of memory (OOM)"))
    axes[0, 0].legend([h for h, _ in ordered], [l for _, l in ordered], loc="upper left",
                      frameon=True, framealpha=0.95, edgecolor="0.3")
    fig.text(0.005, 0.5, "Peak CUDA reserved memory (GB)", rotation=90,
             fontweight="bold", ha="left", va="center")
    plt.tight_layout(pad=0.6)
    plt.subplots_adjust(left=0.09)
    for fmt in ("png", "pdf"):
        fig.savefig(f"{out_prefix}.{fmt}", dpi=300 if fmt == "png" else None,
                    bbox_inches="tight", pad_inches=0.02, facecolor="white", format=fmt)
        print(f"wrote {out_prefix}.{fmt}")
    if show:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--csv", default="memory_use.csv",
                    help="CSV from get_memory_use.py (default ./memory_use.csv)")
    ap.add_argument("--out", default="memory_scaling",
                    help="output prefix; writes <out>.png and <out>.pdf")
    ap.add_argument("--oom_threshold", type=float, default=180.0,
                    help="peak GB above which a run is treated as OOM (default 180)")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    plot_memory_scaling(args.csv, args.out, args.oom_threshold, args.show)


if __name__ == "__main__":
    main()
