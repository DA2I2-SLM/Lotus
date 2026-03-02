#!/usr/bin/env python3
# Plot OpenTSLM-SoftPrompt training curves (learnable vs fixed positional encoding)
# Usage:
#   1) From logs:
#      python plot_posenc_curves.py --fixpos_log fixpos.log --base_log base.log --out val_loss.pdf
#   2) Or use the hard-coded arrays at the bottom (set --use_arrays)

import re
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

VAL_RE = re.compile(r"Epoch\s+(\d+)\s+—\s+val\s+loss:\s+([0-9]*\.?[0-9]+)")
TRAIN_RE = re.compile(r"Epoch\s+(\d+)\s+—\s+train\s+loss:\s+([0-9]*\.?[0-9]+)")

def parse_losses(text: str):
    val = {}
    trn = {}

    for m in TRAIN_RE.finditer(text):
        trn[int(m.group(1))] = float(m.group(2))
    for m in VAL_RE.finditer(text):
        val[int(m.group(1))] = float(m.group(2))

    def to_series(d):
        if not d:
            return [], []
        xs = sorted(d.keys())
        ys = [d[x] for x in xs]
        return xs, ys

    return (*to_series(trn), *to_series(val))  # tr_x, tr_y, val_x, val_y

def best_point(xs, ys):
    if not xs:
        return None
    j = min(range(len(ys)), key=lambda i: ys[i])
    return xs[j], ys[j]

def plot_one(title, fix_x, fix_y, base_x, base_y, onlytime_x, onlytime_y, outpath):
    plt.figure()
    if fix_x:
        plt.plot(fix_x, fix_y, marker="o", linewidth=1.5, markersize=3, label="Add PosEnc (Sinusoidal)")
        bp = best_point(fix_x, fix_y)
        if bp:
            plt.scatter([bp[0]], [bp[1]], marker="x", s=60)
            plt.annotate(f"best: {bp[1]:.4f} @ ep {bp[0]}", (bp[0], bp[1]),
                         textcoords="offset points", xytext=(8, -12))
    if base_x:
        plt.plot(base_x, base_y, marker="o", linewidth=1.5, markersize=3, label="OpenTSLM-SoftPrompt")
        bp = best_point(base_x, base_y)
        if bp:
            plt.scatter([bp[0]], [bp[1]], marker="x", s=60)
            plt.annotate(f"best: {bp[1]:.4f} @ ep {bp[0]}", (bp[0], bp[1]),
                         textcoords="offset points", xytext=(8, 8))
    if onlytime_x:
        plt.plot(onlytime_x, onlytime_y, marker="o", linewidth=1.5, markersize=3, label="Only Time PE")
        bp = best_point(onlytime_x, onlytime_y)
        if bp:
            plt.scatter([bp[0]], [bp[1]], marker="x", s=60)
            plt.annotate(f"best: {bp[1]:.4f} @ ep {bp[0]}", (bp[0], bp[1]),
                         textcoords="offset points", xytext=(8, 20))

    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.grid(True, linewidth=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    print(f"[OK] Saved: {outpath}")

    # save as png too for easier viewing
    png_out = str(Path(outpath).with_suffix(".png"))
    plt.savefig(png_out)
    print(f"[OK] Saved: {png_out}")

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixpos_log", type=str, default="/home/s223540177/FoundationalTS/output_job/tsqa-mn_2840117.out")
    ap.add_argument("--base_log", type=str, default="/home/s223540177/FoundationalTS/output_job/tsqa-mn_2840118.out")
    ap.add_argument("--onlytimepe_log", type=str, default="/home/s223540177/FoundationalTS/output_job/tsqa-mn_2840181.out")
    ap.add_argument("--out", type=str, default="val_loss.pdf")
    ap.add_argument("--plot_train", action="store_true")
    ap.add_argument("--use_arrays", action="store_true",
                    help="Use the arrays copied from your pasted logs instead of parsing files")
    args = ap.parse_args()

    if args.use_arrays:
        # From your pasted output
        fix_val = [0.687156, 0.6620, 0.3393, 0.3303, 0.2442, 0.1697, 0.1452, 0.1500, 0.1633, 0.1065,
                   0.0914, 0.0878, 0.0961, 0.0640, 0.0699, 0.0638, 0.0518, 0.0508, 0.0412, 0.0414]
        base_val = [0.4937, 0.2993, 0.2977, 0.2550, 0.2031, 0.1513, 0.1413, 0.1378, 0.1209, 0.0883,
                    0.0750, 0.0719, 0.0647, 0.0641, 0.0527, 0.0522, 0.0486, 0.0450, 0.0415, 0.0412]
        onlytime_val = []  # Add your data here when using --use_arrays
        fix_x = list(range(1, len(fix_val) + 1))
        base_x = list(range(1, len(base_val) + 1))
        onlytime_x = list(range(1, len(onlytime_val) + 1)) if onlytime_val else []
        plot_one("Validation loss", fix_x, fix_val, base_x, base_val, onlytime_x, onlytime_val, args.out)
        return

    if not args.fixpos_log or not args.base_log:
        raise SystemExit("Provide --fixpos_log and --base_log, or use --use_arrays")

    fix_text = read_text(Path(args.fixpos_log))
    base_text = read_text(Path(args.base_log))
    
    # Parse third log if provided
    if args.onlytimepe_log:
        onlytime_text = read_text(Path(args.onlytimepe_log))
        onlytime_tr_x, onlytime_tr_y, onlytime_val_x, onlytime_val_y = parse_losses(onlytime_text)
    else:
        onlytime_tr_x, onlytime_tr_y, onlytime_val_x, onlytime_val_y = [], [], [], []

    fix_tr_x, fix_tr_y, fix_val_x, fix_val_y = parse_losses(fix_text)
    base_tr_x, base_tr_y, base_val_x, base_val_y = parse_losses(base_text)

    plot_one("Validation loss", fix_val_x, fix_val_y, base_val_x, base_val_y, 
             onlytime_val_x, onlytime_val_y, args.out)

    if args.plot_train:
        out2 = str(Path(args.out).with_name(Path(args.out).stem + "_train" + Path(args.out).suffix))
        plot_one("Training loss", fix_tr_x, fix_tr_y, base_tr_x, base_tr_y, 
                 onlytime_tr_x, onlytime_tr_y, out2)

if __name__ == "__main__":
    main()