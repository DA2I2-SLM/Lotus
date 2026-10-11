#!/usr/bin/env python3
"""
Export train and test splits for: TSQA, M4, HAR-CoT, Sleep-CoT, ECG-QA-CoT.

For each dataset:
  - Checks existing data at $FTS_DATA_DIR/
  - Downloads to /scratch/Projects/rl_timeseries/test_data/raw/<dataset>/ if missing
  - Writes train.jsonl + test.jsonl to /scratch/Projects/rl_timeseries/test_data/<dataset>/

Splits use the exact same seeds as training so rows match.
"""

import os
import ast
import json
import shutil
import zipfile
import urllib.request
import subprocess
from typing import List, Dict, Tuple, Optional

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ── paths ────────────────────────────────────────────────────────────────────
OUT_ROOT      = "/scratch/Projects/rl_timeseries/test_data"
RAW_DL_ROOT   = os.path.join(OUT_ROOT, "raw")          # download target if missing
EXISTING_DATA = os.environ.get("FTS_DATA_DIR", f"/scratch/{os.environ.get('USER','')}/FoundationalTS/data")

# Same seeds used in training
SEED      = 42
TEST_FRAC = 0.10
VAL_FRAC  = 0.10          # only needed to reproduce train boundary; we skip val export

os.makedirs(OUT_ROOT,    exist_ok=True)
os.makedirs(RAW_DL_ROOT, exist_ok=True)


# ── helpers ──────────────────────────────────────────────────────────────────
def _out_dir(name: str) -> str:
    d = os.path.join(OUT_ROOT, name)
    os.makedirs(d, exist_ok=True)
    return d


def _write_jsonl(records: List[Dict], path: str) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"  → wrote {len(records):,} records to {path}")


def _download_file(url: str, dest: str, desc: str = "") -> None:
    """Download with wget (fast) falling back to urllib."""
    if os.path.exists(dest):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"  Downloading {desc or url} …")
    try:
        subprocess.run(["wget", "-q", "--show-progress", "-O", dest, url], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        with urllib.request.urlopen(url) as resp:
            total = int(resp.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pb:
                for chunk in iter(lambda: resp.read(65536), b""):
                    f.write(chunk)
                    pb.update(len(chunk))


def _extract_zip(zip_path: str, dest_dir: str) -> None:
    print(f"  Extracting {zip_path} → {dest_dir} …")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)


# ─────────────────────────────────────────────────────────────────────────────
# 1. HAR-CoT
# ─────────────────────────────────────────────────────────────────────────────
HAR_URL     = "https://polybox.ethz.ch/index.php/s/kD74GnMYxn3HBEM/download"
HAR_EXISTING = os.path.join(EXISTING_DATA, "har_cot")
HAR_DL_DIR  = os.path.join(RAW_DL_ROOT, "har_cot")


def _get_har_dir() -> str:
    for d in (HAR_EXISTING, HAR_DL_DIR):
        if (os.path.exists(os.path.join(d, "har_cot_train_cot.csv")) and
                os.path.exists(os.path.join(d, "har_cot_test_cot.csv"))):
            return d
    # download
    zip_path = os.path.join(RAW_DL_ROOT, "har_cot.zip")
    _download_file(HAR_URL, zip_path, "HAR-CoT")
    os.makedirs(HAR_DL_DIR, exist_ok=True)
    _extract_zip(zip_path, HAR_DL_DIR)
    return HAR_DL_DIR


def export_har_cot():
    print("\n=== HAR-CoT ===")
    d = _get_har_dir()

    def load_csv(path):
        df = pd.read_csv(path)
        df = df[df["x_axis"] != "x_axis"].reset_index(drop=True)  # drop dup headers
        for col in ("x_axis", "y_axis", "z_axis"):
            df[col] = df[col].apply(ast.literal_eval)
        return df

    train_df = load_csv(os.path.join(d, "har_cot_train_cot.csv"))
    test_df  = load_csv(os.path.join(d, "har_cot_test_cot.csv"))

    out = _out_dir("har_cot")
    _write_jsonl(train_df.to_dict("records"), os.path.join(out, "train.jsonl"))
    _write_jsonl(test_df.to_dict("records"),  os.path.join(out, "test.jsonl"))
    print(f"  HAR-CoT: train={len(train_df):,}  test={len(test_df):,}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Sleep-CoT
# ─────────────────────────────────────────────────────────────────────────────
SLEEP_URL      = "https://polybox.ethz.ch/index.php/s/ZryWSdCFJZ9JR3R/download"
SLEEP_EXISTING = os.path.join(EXISTING_DATA, "sleep", "sleep_cot.csv")
SLEEP_DL_CSV   = os.path.join(RAW_DL_ROOT, "sleep", "sleep_cot.csv")


def _get_sleep_csv() -> str:
    if os.path.exists(SLEEP_EXISTING):
        return SLEEP_EXISTING
    _download_file(SLEEP_URL, SLEEP_DL_CSV, "Sleep-CoT")
    return SLEEP_DL_CSV


def export_sleep_cot():
    print("\n=== Sleep-CoT ===")
    csv = _get_sleep_csv()

    df = pd.read_csv(csv)
    print(f"  Parsing time_series column ({len(df):,} rows) …")
    def _parse(s):
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], list):
            return parsed[0]
        return parsed
    df["time_series"] = [_parse(s) for s in tqdm(df["time_series"], desc="  parsing")]

    train_val, test = train_test_split(
        df, test_size=TEST_FRAC, random_state=SEED, stratify=df["label"])
    val_frac_adj = VAL_FRAC / (1.0 - TEST_FRAC)
    train, _ = train_test_split(
        train_val, test_size=val_frac_adj, random_state=SEED + 1, stratify=train_val["label"])

    out = _out_dir("sleep_cot")
    _write_jsonl(train.to_dict("records"), os.path.join(out, "train.jsonl"))
    _write_jsonl(test.to_dict("records"),  os.path.join(out, "test.jsonl"))
    print(f"  Sleep-CoT: train={len(train):,}  test={len(test):,}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. ECG-QA-CoT
# ─────────────────────────────────────────────────────────────────────────────
ECG_COT_URL      = "https://polybox.ethz.ch/index.php/s/D5QaJSEw4dXkzXm/download/ecg_qa_cot_final.zip"
ECG_COT_EXISTING = os.path.join(EXISTING_DATA, "ecg_qa_cot")
ECG_COT_DL_DIR   = os.path.join(RAW_DL_ROOT, "ecg_qa_cot")


def _get_ecg_cot_dir() -> str:
    for d in (ECG_COT_EXISTING, ECG_COT_DL_DIR):
        if os.path.exists(os.path.join(d, "ecg_qa_cot_train.csv")):
            return d
    zip_path = os.path.join(RAW_DL_ROOT, "ecg_qa_cot.zip")
    _download_file(ECG_COT_URL, zip_path, "ECG-QA-CoT")
    _extract_zip(zip_path, RAW_DL_ROOT)
    # The zip may extract to a subdir; find it.
    for candidate in (ECG_COT_DL_DIR, os.path.join(RAW_DL_ROOT, "ecg_qa_cot_final")):
        if os.path.exists(os.path.join(candidate, "ecg_qa_cot_train.csv")):
            if candidate != ECG_COT_DL_DIR:
                shutil.move(candidate, ECG_COT_DL_DIR)
            return ECG_COT_DL_DIR
    raise FileNotFoundError("ecg_qa_cot_train.csv not found after extraction")


def export_ecg_qa_cot():
    print("\n=== ECG-QA-CoT ===")
    d = _get_ecg_cot_dir()

    # Export the CSV rows directly; ecg_id preserved so signals can be resolved later
    train_df = pd.read_csv(os.path.join(d, "ecg_qa_cot_train.csv"))
    test_df  = pd.read_csv(os.path.join(d, "ecg_qa_cot_test.csv"))

    out = _out_dir("ecg_qa_cot")
    _write_jsonl(train_df.to_dict("records"), os.path.join(out, "train.jsonl"))
    _write_jsonl(test_df.to_dict("records"),  os.path.join(out, "test.jsonl"))
    print(f"  ECG-QA-CoT: train={len(train_df):,}  test={len(test_df):,}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. M4
# ─────────────────────────────────────────────────────────────────────────────
M4_URL       = "https://polybox.ethz.ch/index.php/s/MT3y9WdEebT8wfj/download/M4TimeSeriesCaptionDatasetV02.zip"
M4_EXISTING  = os.path.join(EXISTING_DATA, "M4TimeSeriesCaptionDataset", "M4TimeSeriesCaptionDataset")
M4_DL_ROOT   = os.path.join(RAW_DL_ROOT, "M4TimeSeriesCaptionDataset")
M4_GENERATED = os.path.join(M4_DL_ROOT, "M4TimeSeriesCaptionDataset")
FREQUENCIES  = ["Daily", "Hourly", "Monthly", "Quarterly", "Weekly", "Yearly"]


def _get_m4_generated_dir() -> str:
    if os.path.exists(M4_EXISTING) and any(
        os.path.exists(os.path.join(M4_EXISTING, f"m4_series_{f}.csv")) for f in FREQUENCIES
    ):
        return M4_EXISTING
    zip_path = os.path.join(RAW_DL_ROOT, "M4TimeSeriesCaptionDatasetV02.zip")
    _download_file(M4_URL, zip_path, "M4 captions")
    os.makedirs(M4_DL_ROOT, exist_ok=True)
    _extract_zip(zip_path, M4_DL_ROOT)
    return M4_GENERATED


def export_m4():
    print("\n=== M4 ===")
    gen_dir = _get_m4_generated_dir()

    all_samples = []
    for freq in FREQUENCIES:
        s_file = os.path.join(gen_dir, f"m4_series_{freq}.csv")
        c_file = os.path.join(gen_dir, f"m4_captions_{freq}.csv")
        if not os.path.exists(s_file) or not os.path.exists(c_file):
            print(f"  skipping {freq} (files not found)")
            continue
        s_df = pd.read_csv(s_file)
        c_df = pd.read_csv(c_file)
        merged = s_df.merge(c_df, on="id", how="inner")
        for _, row in merged.iterrows():
            series = row["series"]
            if isinstance(series, str):
                try:
                    series = json.loads(series)
                except json.JSONDecodeError:
                    series = ast.literal_eval(series)
            all_samples.append({
                "id": row["id"],
                "frequency": freq,
                "series": series,
                "caption": row["caption"],
            })
        print(f"  {freq}: {len(merged):,} samples")

    if not all_samples:
        print("  No M4 samples loaded — skipping export.")
        return

    # Reproduce training split exactly (same seeds)
    import random
    random.seed(SEED)
    random.shuffle(all_samples)
    n = len(all_samples)
    test_end  = int(n * TEST_FRAC)
    # train_test_split used HF Dataset.train_test_split with seed=42 then seed=43
    # Reproduce with sklearn to match ordering (HF uses same underlying logic)
    df = pd.DataFrame({"_idx": range(n)})
    train_val_df, test_df = train_test_split(df, test_size=TEST_FRAC, random_state=SEED)
    val_frac_adj = VAL_FRAC / (1.0 - TEST_FRAC)
    train_df, _ = train_test_split(train_val_df, test_size=val_frac_adj, random_state=SEED + 1)

    train_recs = [all_samples[i] for i in train_df["_idx"].tolist()]
    test_recs  = [all_samples[i] for i in test_df["_idx"].tolist()]

    out = _out_dir("m4")
    _write_jsonl(train_recs, os.path.join(out, "train.jsonl"))
    _write_jsonl(test_recs,  os.path.join(out, "test.jsonl"))
    print(f"  M4: train={len(train_recs):,}  test={len(test_recs):,}  total={n:,}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. TSQA
# ─────────────────────────────────────────────────────────────────────────────
HF_CACHE = os.path.join(RAW_DL_ROOT, "hf_cache")


def export_tsqa():
    print("\n=== TSQA ===")
    os.makedirs(HF_CACHE, exist_ok=True)

    from datasets import load_dataset, DownloadMode
    ds = load_dataset(
        "ChengsenWang/TSQA",
        split="train",
        cache_dir=HF_CACHE,
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
    )
    print(f"  Loaded {len(ds):,} rows from ChengsenWang/TSQA")

    # Reproduce training split (same seeds as Phase1JointDataset._load_tsqa_lists)
    tv, test_hf = ds.train_test_split(test_size=TEST_FRAC, seed=SEED).values()
    val_frac_adj = VAL_FRAC / (1.0 - TEST_FRAC)
    train_hf, _ = tv.train_test_split(test_size=val_frac_adj, seed=SEED).values()

    out = _out_dir("tsqa")
    _write_jsonl(list(train_hf), os.path.join(out, "train.jsonl"))
    _write_jsonl(list(test_hf),  os.path.join(out, "test.jsonl"))
    print(f"  TSQA: train={len(train_hf):,}  test={len(test_hf):,}")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Export train/test splits for all datasets")
    ap.add_argument("--datasets", nargs="+",
                    choices=["har_cot", "sleep_cot", "ecg_qa_cot", "m4", "tsqa", "all"],
                    default=["all"])
    args = ap.parse_args()

    want = set(args.datasets)
    run_all = "all" in want

    if run_all or "har_cot"    in want: export_har_cot()
    if run_all or "sleep_cot"  in want: export_sleep_cot()
    if run_all or "ecg_qa_cot" in want: export_ecg_qa_cot()
    if run_all or "m4"         in want: export_m4()
    if run_all or "tsqa"       in want: export_tsqa()

    print(f"\nDone. Splits written to {OUT_ROOT}/")
