#!/usr/bin/env python3
"""
Download Chronos UV datasets from autogluon/chronos_datasets HF Hub and save
them as local HF datasets ready for load_dataset() via CHRONOS_UV_BASE_PATH.

Run on the login node (has internet). Compute nodes do not.

Usage:
    python scripts/prefetch_chronos_uv.py \
        --out_dir /vast/s225250685/project/FoundationalTS/_chronos_uv_datasets \
        --hf_token <HF_TOKEN_REDACTED>

The script saves each dataset as a Hugging Face dataset on disk at:
    <out_dir>/<name>/

Then set in your training script:
    export CHRONOS_UV_BASE_PATH=/vast/s225250685/project/FoundationalTS/_chronos_uv_datasets
"""

import argparse
import os
import subprocess
import sys

from datasets import load_dataset

# (hf_repo, config_name, local_name, target_keys)
# local_name = directory name under out_dir (matches CHRONOS_DATASET_REGISTRY key)
DATASETS = [
    # Already available via shared path — included here for completeness
    ("autogluon/chronos_datasets", "monash_australian_electricity", "monash_australian_electricity", ["target"]),
    # Zero-shot eval sets (real-world diverse UV)
    ("autogluon/chronos_datasets", "dominick",                  "dominick",                  ["target"]),
    ("autogluon/chronos_datasets", "monash_fred_md",            "monash_fred_md",            ["target"]),
    ("autogluon/chronos_datasets", "electricity_15min",         "electricity_15min",         ["consumption_kW"]),
    ("autogluon/chronos_datasets", "ercot",                     "ercot",                     ["target"]),
    ("autogluon/chronos_datasets", "exchange_rate",             "exchange_rate",             ["target"]),
    ("autogluon/chronos_datasets", "monash_traffic",            "monash_traffic",            ["target"]),
    ("autogluon/chronos_datasets", "monash_weather",            "monash_weather",            ["target"]),
    ("autogluon/chronos_datasets", "monash_covid_deaths",       "monash_covid_deaths",       ["target"]),
    ("autogluon/chronos_datasets", "m4_quarterly",              "m4_quarterly",              ["target"]),
    ("autogluon/chronos_datasets", "m4_yearly",                 "m4_yearly",                 ["target"]),
    ("autogluon/chronos_datasets", "m5",                        "m5",                        ["target"]),
    ("autogluon/chronos_datasets", "monash_car_parts",          "monash_car_parts",          ["target"]),
    ("autogluon/chronos_datasets", "monash_hospital",           "monash_hospital",           ["target"]),
    ("autogluon/chronos_datasets", "monash_m3_monthly",         "monash_m3_monthly",         ["target"]),
    ("autogluon/chronos_datasets", "monash_m3_quarterly",       "monash_m3_quarterly",       ["target"]),
    ("autogluon/chronos_datasets", "monash_m3_yearly",          "monash_m3_yearly",          ["target"]),
    ("autogluon/chronos_datasets", "nn5",                       "nn5",                       ["target"]),
    ("autogluon/chronos_datasets", "monash_nn5_weekly",         "monash_nn5_weekly",         ["target"]),
    ("autogluon/chronos_datasets", "monash_tourism_monthly",    "monash_tourism_monthly",    ["target"]),
    ("autogluon/chronos_datasets", "monash_tourism_quarterly",  "monash_tourism_quarterly",  ["target"]),
    ("autogluon/chronos_datasets", "monash_tourism_yearly",     "monash_tourism_yearly",     ["target"]),
    # In-domain (larger datasets)
    ("autogluon/chronos_datasets", "monash_electricity_hourly", "monash_electricity_hourly", ["target"]),
    ("autogluon/chronos_datasets", "monash_electricity_weekly", "monash_electricity_weekly", ["target"]),
    ("autogluon/chronos_datasets", "m4_daily",                  "m4_daily",                  ["target"]),
    ("autogluon/chronos_datasets", "m4_hourly",                 "m4_hourly",                 ["target"]),
    ("autogluon/chronos_datasets", "m4_monthly",                "m4_monthly",                ["target"]),
    ("autogluon/chronos_datasets", "m4_weekly",                 "m4_weekly",                 ["target"]),
    ("autogluon/chronos_datasets", "monash_pedestrian_counts",  "monash_pedestrian_counts",  ["target"]),
    ("autogluon/chronos_datasets", "taxi_30min",                "taxi_30min",                ["target"]),
    ("autogluon/chronos_datasets", "uber_tlc_hourly",           "uber_tlc_hourly",           ["target"]),
    ("autogluon/chronos_datasets", "uber_tlc_daily",            "uber_tlc_daily",            ["target"]),
    ("autogluon/chronos_datasets", "monash_rideshare",          "monash_rideshare",          ["target"]),
    ("autogluon/chronos_datasets", "monash_temperature_rain",   "monash_temperature_rain",   ["target"]),
    ("autogluon/chronos_datasets", "monash_london_smart_meters","monash_london_smart_meters",["target"]),
    ("autogluon/chronos_datasets", "monash_kdd_cup_2018",       "monash_kdd_cup_2018",       ["target"]),
]


def du_mb(path):
    r = subprocess.run(["du", "-sm", path], capture_output=True, text=True)
    if r.stdout:
        return int(r.stdout.split()[0])
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/vast/s225250685/project/FoundationalTS/_chronos_uv_datasets")
    ap.add_argument("--hf_token", default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--only", nargs="*", help="Only download these names (local_name)")
    ap.add_argument("--skip_existing", action="store_true", default=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    to_fetch = DATASETS
    if args.only:
        to_fetch = [d for d in DATASETS if d[2] in args.only]

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


if __name__ == "__main__":
    main()
