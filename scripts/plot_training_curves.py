#!/usr/bin/env python3
"""Plot train/val curves for every pretraining run with a loss_history.txt.

Reads results/**/checkpoints/loss_history.txt (one row per validation event;
for the trans_enc runs that is every 1000 optimizer steps). Emits:
  results/plots/training_curves_r8_sweep.png   — the 4-run r8 sweep (2x2)
  results/plots/training_curves_all.png        — every run on disk (grid)

Design notes: small multiples with INDEPENDENT log-y axes on purpose — the
runs optimise different objectives (CE+MSE joint, MSE-only, CE-only) over
different tokenizers, so overlaying totals on one axis would invite exactly
the cross-run comparison the numbers cannot support. Train is the in-epoch
running average the trainer logs; val is the full validation loss. The dot
marks the best (checkpoint-selected) validation point.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Reference dataviz palette (pre-validated, light mode): slot 1 blue = val,
# slot 2 orange = train. Text/grid wear text tokens, not series colors.
C_VAL, C_TRAIN = "#2a78d6", "#eb6834"
C_TEXT, C_SUB, C_GRID, C_SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

RUNS = [  # (relpath under results/, label, objective, in_r8_sweep)
    ("sincos_patch32/gemma_3_1b_pt/OpenTSLMSPFixPos2D/joint_phase1_extend_balanced_lora_r8_freq_mvpos",
     "Gemma-3-1B · LoRA r8 · freq", "joint CE+MSE", True),
    ("sincos_patch32/Qwen3_0_6B/OpenTSLMSPFixPos2D/joint_phase1_extend_balanced_lora_r8_freq_mvpos",
     "Qwen3-0.6B · LoRA r8 · freq", "joint CE+MSE", True),
    ("sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase1_extend_balanced_lora_r8_nofreq_mvpos",
     "Gemma-270M · LoRA r8 · nofreq", "joint CE+MSE", True),
    ("sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase1_tsqa_m4_freq_mvpos_lora_r8",
     "Gemma-270M · language-only r8 · freq", "CE only (TSQA+M4)", True),
    ("sincos_patch32/gemma_3_1b_pt/OpenTSLMSPFixPos2D/joint_phase1_extend_balanced_freq_mvpos",
     "Gemma-3-1B · no-LoRA · freq", "joint CE+MSE", False),
    ("sincos_patch32/gemma_3_1b_pt/OpenTSLMSPFixPos2D/phase1_chronos_mse_freq_mvpos",
     "Gemma-3-1B · TS-only · freq", "MSE only", False),
    ("sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase1_extend_balanced_lora_freq_mvpos",
     "Gemma-270M · LoRA r4 · freq (no m4)", "joint CE+MSE", False),
    ("sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase1_extend_balanced_lora_nofreq_mvpos",
     "Gemma-270M · LoRA r4 · nofreq", "joint CE+MSE", False),
    ("sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/phase1_chronos_mse_lora_freq_mvpos",
     "Gemma-270M · TS-only LoRA r4 · freq", "MSE only", False),
    ("sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/phase1_chronos_mse_freq_seqrope",
     "Gemma-270M · TS-only · seqrope", "MSE only", False),
    ("sincos_patch32_mlp/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase1_extend_balanced_freq",
     "270M-MLP · joint · freq", "joint CE+MSE", False),
    ("sincos_patch32_mlp/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase1_extend_balanced_lora_nofreq",
     "270M-MLP · joint LoRA · nofreq", "joint CE+MSE", False),
    ("sincos_patch32_mlp/gemma_3_270m/OpenTSLMSPFixPos2D/phase1_chronos_mse_freq",
     "270M-MLP · TS-only · freq", "MSE only", False),
    ("sincos_patch32_mlp/gemma_3_270m/OpenTSLMSPFixPos2D/phase1_chronos_mse_lora_nofreq",
     "270M-MLP · TS-only LoRA · nofreq", "MSE only", False),
    ("sincos_patch32_mlp/gemma_3_270m/OpenTSLMSPFixPos2D/phase1_chronos_mse_full_nofreq",
     "270M-MLP · TS-only full-FT · nofreq", "MSE only", False),
]


def load(rel):
    path = os.path.join(ROOT, "results", rel, "checkpoints", "loss_history.txt")
    ep, tr, va = [], [], []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 3 or not parts[0].isdigit():
                continue
            ep.append(int(parts[0])); tr.append(float(parts[1])); va.append(float(parts[2]))
    return ep, tr, va


def panel(ax, rel, label, objective):
    ep, tr, va = load(rel)
    x = list(range(1, len(va) + 1))
    # faint epoch boundaries
    for i in range(1, len(ep)):
        if ep[i] != ep[i - 1]:
            ax.axvline(x[i], color=C_GRID, lw=0.8, zorder=0)
    ax.plot(x, tr, color=C_TRAIN, lw=1.4, zorder=2)
    ax.plot(x, va, color=C_VAL, lw=2.0, zorder=3)
    bi = min(range(len(va)), key=lambda i: va[i])
    ax.scatter([x[bi]], [va[bi]], s=42, color=C_VAL, edgecolor=C_SURF,
               linewidth=1.5, zorder=4)
    ax.annotate(f"best {va[bi]:.3f}", (x[bi], va[bi]),
                textcoords="offset points", xytext=(6, 8),
                fontsize=8, color=C_TEXT)
    ax.set_yscale("log")
    ax.set_title(label, fontsize=10, color=C_TEXT, loc="left", pad=10)
    ax.text(0, 1.015, f"{objective} · {ep[-1]} epoch{'s' if ep[-1] > 1 else ''}",
            transform=ax.transAxes, fontsize=8, color=C_SUB, va="bottom")
    ax.grid(axis="y", color=C_GRID, lw=0.7)
    ax.set_facecolor(C_SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_GRID)
    ax.tick_params(colors=C_SUB, labelsize=8)
    return va[bi], bi + 1, len(va)


def make_fig(runs, ncols, out, suptitle, note):
    n = len(runs)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.3 * nrows))
    fig.patch.set_facecolor(C_SURF)
    axf = axes.flat if n > 1 else [axes]
    for ax, (rel, label, obj, _) in zip(axf, runs):
        best, at, tot = panel(ax, rel, label, obj)
        print(f"  {label:<42} evals={tot:<4} best_val={best:.4f} @eval {at}")
    for ax in list(axf)[n:]:
        ax.axis("off")
    # shared legend + axis captions (identity never by color alone: labeled here)
    fig.legend(handles=[
        plt.Line2D([], [], color=C_VAL, lw=2.0, label="validation loss"),
        plt.Line2D([], [], color=C_TRAIN, lw=1.4, label="train loss (epoch running avg)"),
    ], loc="upper right", frameon=False, fontsize=9,
        bbox_to_anchor=(0.995, 0.995))
    fig.suptitle(suptitle, fontsize=13, color=C_TEXT, x=0.005,
                 ha="left", y=0.997, fontweight="bold")
    fig.supxlabel("validation checkpoint # (trans-enc runs: 1 per 1,000 optimizer steps; "
                  "thin vertical lines = epoch boundaries)", fontsize=9, color=C_SUB)
    fig.supylabel("loss (log scale — per-panel axes; totals are NOT comparable across "
                  "objectives/tokenizers)", fontsize=9, color=C_SUB)
    if note:
        fig.text(0.005, 0.004, note, fontsize=8, color=C_SUB, style="italic")
    fig.tight_layout(rect=(0.012, 0.045, 1, 0.965))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=C_SURF)
    plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    sweep = [r for r in RUNS if r[3]]
    print("== r8 sweep ==")
    make_fig(sweep, 2, os.path.join(ROOT, "results/plots/training_curves_r8_sweep.png"),
             "Phase-1 pretraining — LoRA r=8 backbone sweep",
             "Gemma-1B curve spans two chained 5-day jobs (resumed from epoch 3); "
             "language-only run converged via early stopping.")
    print("== all runs ==")
    make_fig(RUNS, 3, os.path.join(ROOT, "results/plots/training_curves_all.png"),
             "Phase-1 pretraining — all runs with recorded loss history",
             "Older runs pre-date the m4 corpus and the shared _1b split cache; "
             "epoch lengths and eval cadence differ across families.")
