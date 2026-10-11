#!/usr/bin/env python3
"""
Compare our GIFT-Eval results against paper baselines.
Baselines pulled from: https://github.com/SalesforceAIResearch/gift-eval
"""
import os
import sys
import pandas as pd
import numpy as np

# Set these in the environment; there is no portable default (they point at
# a GIFT-Eval results checkout and at our own results csv).
BASE = os.environ.get("FTS_GIFT_RESULTS_BASE", "")
OURS = os.environ.get(
    "FTS_GIFT_OURS_CSV",
    os.path.join(os.environ.get("FTS_RESULTS_ROOT", "results"),
                 "OpenTSLMSPFixPos2D", "all_results.csv"),
)
if not BASE:
    raise SystemExit(
        "FTS_GIFT_RESULTS_BASE is not set - point it at the GIFT-Eval "
        "results directory (see bashfiles/_paths.sh)."
    )

MODELS = {
    "SeasonalNaive":  f"{BASE}/seasonal_naive/all_results.csv",
    "Chronos-Small":  f"{BASE}/Chronos_small/all_results.csv",
    "Chronos-Base":   f"{BASE}/chronos_base/all_results.csv",
    "Chronos-Large":  f"{BASE}/chronos_large/all_results.csv",
    "Moirai-Small":   f"{BASE}/moirai_small/all_results.csv",
    "Moirai-Base":    f"{BASE}/Moirai_base/all_results.csv",
    "Moirai-Large":   f"{BASE}/Moirai_large/all_results.csv",
    "TimesFM-1.0":    f"{BASE}/timesfm/all_results.csv",
    "TTM-R2-PT":      f"{BASE}/TTM-R2-Pretrained/all_results.csv",
}

def get_col(df, name):
    """Get column by name, handling eval_metrics/ prefix in baseline CSVs."""
    if name in df.columns:
        return df[name]
    prefixed = f"eval_metrics/{name}"
    if prefixed in df.columns:
        return df[prefixed]
    return pd.Series(dtype=float)

def summarize(df):
    df = df.copy()
    df["term"] = df["dataset"].str.split("/").str[-1]
    res = {}
    for term in ["short", "medium", "long", "overall"]:
        sub = df if term == "overall" else df[df["term"] == term]
        mase = get_col(sub, "MASE[0.5]").dropna().mean()
        wql  = get_col(sub, "mean_weighted_sum_quantile_loss").dropna().mean()
        res[term] = (round(mase, 4), round(wql, 4), len(sub))
    return res

def main():
    dfs = {}
    for name, path in MODELS.items():
        if os.path.exists(path):
            dfs[name] = pd.read_csv(path)
        else:
            print(f"  [MISSING] {name}: {path}", file=sys.stderr)

    ours = pd.read_csv(OURS)
    dfs["Ours"] = ours

    summaries = {name: summarize(df) for name, df in dfs.items()}

    # ── Overall comparison table ──────────────────────────────────────────────
    print("GIFT-Eval Comparison: MASE[0.5] and WQL (mean_weighted_sum_quantile_loss)")
    print("Lower is better for both metrics.\n")

    header = f"{'Model':<18}  {'Short':>14}  {'Medium':>14}  {'Long':>14}  {'Overall':>14}  n"
    subhdr = f"{'':18}  {'MASE   WQL':>14}  {'MASE   WQL':>14}  {'MASE   WQL':>14}  {'MASE   WQL':>14}"
    print(header)
    print(subhdr)
    print("-" * 95)

    for name, s in summaries.items():
        sm, sw, sn = s["short"]
        mm, mw, mn = s["medium"]
        lm, lw, ln = s["long"]
        om, ow, on = s["overall"]
        is_ours = name == "Ours"
        marker = " ◄" if is_ours else ""
        print(f"{'**' if is_ours else ''}{name:<18}{'**' if is_ours else ''}  "
              f"{sm:5.3f}  {sw:5.3f}  "
              f"{mm:5.3f}  {mw:5.3f}  "
              f"{lm:5.3f}  {lw:5.3f}  "
              f"{om:5.3f}  {ow:5.3f}  "
              f"{on}{marker}")

    # ── MASE ranking ──────────────────────────────────────────────────────────
    print("\n\nOverall MASE ranking (lower = better):")
    ranked = sorted(summaries.items(), key=lambda x: x[1]["overall"][0])
    for rank, (name, s) in enumerate(ranked, 1):
        om, ow, on = s["overall"]
        marker = " ◄ OUR MODEL" if name == "Ours" else ""
        print(f"  {rank:2d}. {name:<18}  MASE={om:.4f}  WQL={ow:.4f}{marker}")

    # ── Per-domain breakdown for ours vs best baseline ──────────────────────
    print("\n\nPer-domain: Ours vs best available baseline (MASE)")
    ours_df = dfs["Ours"].copy()
    ours_df["term"] = ours_df["dataset"].str.split("/").str[-1]

    # Best non-ours model by overall MASE
    best_name = ranked[0][0] if ranked[0][0] != "Ours" else ranked[1][0]
    best_df = dfs[best_name].copy()
    best_df["term"] = best_df["dataset"].str.split("/").str[-1]

    print(f"  Comparing Ours vs {best_name}")
    print(f"  {'Domain':<16}  {'Ours MASE':>10}  {best_name+' MASE':>16}  {'Δ':>8}")
    print("  " + "-" * 56)
    for domain in sorted(ours_df["domain"].dropna().unique()):
        o_sub = ours_df[ours_df["domain"] == domain]
        b_sub = best_df[best_df["domain"] == domain] if "domain" in best_df.columns else pd.DataFrame()
        o_mase = get_col(o_sub, "MASE[0.5]").dropna().mean()
        b_mase = get_col(b_sub, "MASE[0.5]").dropna().mean() if len(b_sub) > 0 else float("nan")
        delta = f"{((o_mase - b_mase) / b_mase * 100):+.1f}%" if not np.isnan(b_mase) else "N/A"
        print(f"  {domain:<16}  {o_mase:>10.4f}  {b_mase:>16.4f}  {delta:>8}")

if __name__ == "__main__":
    main()
