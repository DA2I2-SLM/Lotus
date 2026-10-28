#!/usr/bin/env python3
"""
Download Chronos datasets from autogluon/chronos_datasets on the HF Hub and save
them as local HF datasets ready for load_dataset() via CHRONOS_UV_BASE_PATH.

Run on the login node (has internet). Compute nodes do not.

Usage:
    # default: the 10 Phase-1 pretraining sources (CHRONOS_DATASET_REGISTRY in
    # src/opentslm/time_series_datasets/Phase1JointDataset.py)
    python scripts/prefetch_chronos_uv.py \
        --out_dir $FTS_VAST/FoundationalTS/_chronos_uv_datasets \
        --hf_token "$HF_TOKEN"   # never hard-code a token here

    # a subset (comma-separated local names)
    python scripts/prefetch_chronos_uv.py --only ushcn_daily,wiki_daily_100k

    # every known config (36 datasets, ~10 GB)
    python scripts/prefetch_chronos_uv.py --all

The script saves each dataset as a Hugging Face dataset on disk at:
    <out_dir>/<name>/

Then set in your training script (only needed when --out_dir differs from
the default, which is what the training code falls back to):
    export CHRONOS_UV_BASE_PATH=$FTS_VAST/FoundationalTS/_chronos_uv_datasets
"""

import argparse
import os
import subprocess

from datasets import load_dataset

HF_REPO = "autogluon/chronos_datasets"

# The 10 Phase-1 pretraining sources. Must stay in sync with
# CHRONOS_DATASET_REGISTRY in src/opentslm/time_series_datasets/Phase1JointDataset.py.
PHASE1_DATASETS = [
    "dominick",
    "monash_fred_md",
    "electricity_15min",
    "ercot",
    "exchange_rate",
    "ushcn_daily",
    "weatherbench_daily",
    "weatherbench_weekly",
    "wiki_daily_100k",
    "monash_australian_electricity",
]

# (hf_repo, config_name, local_name, target_keys)
# local_name = directory name under out_dir (matches the registry keys).
DATASETS = [
    # ── Phase-1 pretraining sources ────────────────────────────────────────
    (HF_REPO, "dominick",                      "dominick",                      ["target"]),
    (HF_REPO, "monash_fred_md",                "monash_fred_md",                ["target"]),
    (HF_REPO, "electricity_15min",             "electricity_15min",             ["consumption_kW"]),
    (HF_REPO, "ercot",                         "ercot",                         ["target"]),
    (HF_REPO, "exchange_rate",                 "exchange_rate",                 ["target"]),
    (HF_REPO, "ushcn_daily",                   "ushcn_daily",                   ["PRCP", "SNOW", "SNWD", "TMAX", "TMIN"]),
    (HF_REPO, "weatherbench_daily",            "weatherbench_daily",            ["target"]),
    (HF_REPO, "weatherbench_weekly",           "weatherbench_weekly",           ["target"]),
    (HF_REPO, "wiki_daily_100k",               "wiki_daily_100k",               ["target"]),
    (HF_REPO, "monash_australian_electricity", "monash_australian_electricity", ["target"]),
    # ── Zero-shot eval sets (real-world diverse UV) ────────────────────────
    (HF_REPO, "monash_traffic",                "monash_traffic",                ["target"]),
    (HF_REPO, "monash_weather",                "monash_weather",                ["target"]),
    (HF_REPO, "monash_covid_deaths",           "monash_covid_deaths",           ["target"]),
    (HF_REPO, "m4_quarterly",                  "m4_quarterly",                  ["target"]),
    (HF_REPO, "m4_yearly",                     "m4_yearly",                     ["target"]),
    (HF_REPO, "m5",                            "m5",                            ["target"]),
    (HF_REPO, "monash_car_parts",              "monash_car_parts",              ["target"]),
    (HF_REPO, "monash_hospital",               "monash_hospital",               ["target"]),
    (HF_REPO, "monash_m3_monthly",             "monash_m3_monthly",             ["target"]),
    (HF_REPO, "monash_m3_quarterly",           "monash_m3_quarterly",           ["target"]),
    (HF_REPO, "monash_m3_yearly",              "monash_m3_yearly",              ["target"]),
    (HF_REPO, "nn5",                           "nn5",                           ["target"]),
    (HF_REPO, "monash_nn5_weekly",             "monash_nn5_weekly",             ["target"]),
    (HF_REPO, "monash_tourism_monthly",        "monash_tourism_monthly",        ["target"]),
    (HF_REPO, "monash_tourism_quarterly",      "monash_tourism_quarterly",      ["target"]),
    (HF_REPO, "monash_tourism_yearly",         "monash_tourism_yearly",         ["target"]),
    # ── In-domain (larger datasets) ────────────────────────────────────────
    (HF_REPO, "monash_electricity_hourly",     "monash_electricity_hourly",     ["target"]),
    (HF_REPO, "monash_electricity_weekly",     "monash_electricity_weekly",     ["target"]),
    (HF_REPO, "m4_daily",                      "m4_daily",                      ["target"]),
    (HF_REPO, "m4_hourly",                     "m4_hourly",                     ["target"]),
    (HF_REPO, "m4_monthly",                    "m4_monthly",                    ["target"]),
    (HF_REPO, "m4_weekly",                     "m4_weekly",                     ["target"]),
    (HF_REPO, "monash_pedestrian_counts",      "monash_pedestrian_counts",      ["target"]),
    (HF_REPO, "taxi_30min",                    "taxi_30min",                    ["target"]),
    (HF_REPO, "uber_tlc_hourly",               "uber_tlc_hourly",               ["target"]),
    (HF_REPO, "uber_tlc_daily",                "uber_tlc_daily",                ["target"]),
    (HF_REPO, "monash_rideshare",              "monash_rideshare",              ["target"]),
    (HF_REPO, "monash_temperature_rain",       "monash_temperature_rain",       ["target"]),
    (HF_REPO, "monash_london_smart_meters",    "monash_london_smart_meters",    ["target"]),
    (HF_REPO, "monash_kdd_cup_2018",           "monash_kdd_cup_2018",           ["target"]),
]

_KNOWN = {d[2] for d in DATASETS}
assert set(PHASE1_DATASETS) <= _KNOWN, "PHASE1_DATASETS must be a subset of DATASETS"


def du_mb(path):
    r = subprocess.run(["du", "-sm", path], capture_output=True, text=True)
    if r.stdout:
        return int(r.stdout.split()[0])
    return 0


def _parse_only(values):
    """Accept `--only a,b` as well as `--only a b` (and a mix of both)."""
    names = []
    for v in values or []:
        names.extend(n.strip() for n in v.split(",") if n.strip())
    return names


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir",
                    default=os.environ.get(
                        "CHRONOS_UV_BASE_PATH",
                        os.path.join(os.environ.get("FTS_VAST",
                                     f"/vast/{os.environ.get('USER','')}"),
                                     "FoundationalTS", "_chronos_uv_datasets"),
                    ),
                    help="where <name>/ directories are written "
                         "(default: $CHRONOS_UV_BASE_PATH, else "
                         "$FTS_VAST/FoundationalTS/_chronos_uv_datasets)")
    ap.add_argument("--hf_token", default=os.environ.get("HF_TOKEN", ""))
    sel = ap.add_mutually_exclusive_group()
    sel.add_argument("--only", nargs="+", metavar="NAME[,NAME...]",
                     help="download only these local names (comma- or space-separated)")
    sel.add_argument("--all", action="store_true",
                     help=f"download every known config ({len(DATASETS)} datasets) "
                          "instead of just the Phase-1 set")
    ap.add_argument("--no_skip_existing", dest="skip_existing", action="store_false",
                    default=True, help="re-download datasets that already exist on disk")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    if args.all:
        wanted = [d[2] for d in DATASETS]
    elif args.only:
        wanted = _parse_only(args.only)
        unknown = sorted(set(wanted) - _KNOWN)
        if unknown:
            ap.error(f"unknown dataset name(s): {unknown}. Known: {sorted(_KNOWN)}")
    else:
        wanted = list(PHASE1_DATASETS)
    by_name = {d[2]: d for d in DATASETS}
    to_fetch = [by_name[n] for n in wanted]

    print(f"Fetching {len(to_fetch)} dataset(s) into {args.out_dir}")
    total_mb = 0
    ok, failed = [], []

    for repo, config, local_name, keys in to_fetch:
        out_path = os.path.join(args.out_dir, local_name)
        if args.skip_existing and os.path.isdir(out_path):
            mb = du_mb(out_path)
            total_mb += mb
            print(f"  [skip]  {local_name:<40} {mb:>6} MB  (already exists)")
            ok.append(local_name)
            continue

        print(f"  [fetch] {local_name:<40}", end=" ", flush=True)
        try:
            ds = load_dataset(repo, config, trust_remote_code=True)
            # Concatenate all splits into one flat dataset
            from datasets import concatenate_datasets
            all_splits = list(ds.values())
            combined = concatenate_datasets(all_splits)
            combined.save_to_disk(out_path)
            mb = du_mb(out_path)
            total_mb += mb
            print(f"{len(combined):>7} rows  {mb:>6} MB  -> {out_path}")
            ok.append(local_name)
        except Exception as e:
            print(f"ERROR: {e}")
            failed.append(local_name)

    print()
    print(f"Done: {len(ok)} fetched, {len(failed)} failed")
    if failed:
        print(f"Failed: {failed}")
    print(f"Total size: {total_mb / 1024:.1f} GB at {args.out_dir}")
    print()
    print("Add to training script:")
    print(f"  export CHRONOS_UV_BASE_PATH={args.out_dir}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
