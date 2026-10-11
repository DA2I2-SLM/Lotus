#!/usr/bin/env python3
"""
Merge results_task{0-4}.csv into all_results.csv and print a summary table.
Usage: python merge_gift_eval_results.py --out_dir results/OpenTSLMSPFixPos2D
"""
import argparse
import csv
import os
from pathlib import Path

import pandas as pd
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.path.join(os.environ.get("FTS_RESULTS_ROOT", "results"), "OpenTSLMSPFixPos2D"))
    ap.add_argument("--n_tasks", type=int, default=5)
    args = ap.parse_args()

    dfs = []
    for prefix in ["results_task", "retry_task", "retry3_task"]:
        for i in range(args.n_tasks):
            f = Path(args.out_dir) / f"{prefix}{i}.csv"
            if f.exists():
                dfs.append(pd.read_csv(f))
                print(f"  Loaded {f}: {len(dfs[-1])} rows")

    if not dfs:
        print("No result files found.")
        return

    merged = pd.concat(dfs, ignore_index=True)
    # Keep last result per dataset (retry wins over original)
    merged = merged.drop_duplicates(subset=["dataset"], keep="last")
    merged = merged.sort_values("dataset").reset_index(drop=True)

    out = Path(args.out_dir) / "all_results.csv"
    merged.to_csv(out, index=False)
    print(f"\nMerged {len(merged)} rows → {out}")

    # Summary: avg per term and overall
    print("\n=== Summary (mean MSE[mean] and MASE[0.5]) ===")
    for term in ["short", "medium", "long"]:
        subset = merged[merged["dataset"].str.endswith(f"/{term}")]
        if len(subset) == 0: continue
        mse  = subset["MSE[mean]"].dropna().mean()
        mase = subset["MASE[0.5]"].dropna().mean()
        print(f"  {term:6s}: MSE={mse:.4f}  MASE={mase:.4f}  (n={len(subset)})")

    mse  = merged["MSE[mean]"].dropna().mean()
    mase = merged["MASE[0.5]"].dropna().mean()
    wql  = merged["mean_weighted_sum_quantile_loss"].dropna().mean()
    print(f"  {'overall':6s}: MSE={mse:.4f}  MASE={mase:.4f}  WQL={wql:.4f}  (n={len(merged)})")

    # By domain
    print("\n=== By domain (mean MSE[mean]) ===")
    for domain, grp in merged.groupby("domain"):
        mse = grp["MSE[mean]"].dropna().mean()
        print(f"  {domain:12s}: MSE={mse:.4f}  (n={len(grp)})")


if __name__ == "__main__":
    main()
