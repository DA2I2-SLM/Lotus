# SPDX-License-Identifier: MIT
# Chronos-2 evaluation: MSE & MAE across multiple forecast horizons

import pandas as pd
import numpy as np
import torch
from chronos import BaseChronosPipeline

# ── Config ──────────────────────────────────────────────────────────
DATASET_PATH = "/scratch/Projects/rl_timeseries/pretraining_data/multivariate/chronos_datasets/dominick"
CONTEXT_LEN = 512
HORIZONS = [96, 192, 336, 720]
MAX_SERIES = None  # Set to e.g. 500 to limit for faster testing; None = use all
BATCH_SIZE = 64    # Number of series per predict() call

# ── Load model ──────────────────────────────────────────────────────
print("Loading Chronos-2...")
pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)

# ── Load & prepare data ────────────────────────────────────────────
print(f"Loading dataset from {DATASET_PATH}...")
df = pd.read_parquet(f"{DATASET_PATH}/train-00000-of-00001.parquet")

# Each row is one series with array-valued columns: id, timestamp, target
# Extract as list of numpy arrays (avoid exploding to long format — much faster)
all_series = []
for _, row in df.iterrows():
    values = np.array(row["target"], dtype=np.float32)
    # Skip series that are too short for context + largest horizon
    if len(values) >= CONTEXT_LEN + max(HORIZONS):
        all_series.append(values)

if MAX_SERIES is not None:
    all_series = all_series[:MAX_SERIES]

print(f"Usable series (len >= {CONTEXT_LEN + max(HORIZONS)}): {len(all_series)}")

if len(all_series) == 0:
    # Fallback: use shorter series with the smallest horizon
    print("No series long enough for all horizons. Trying per-horizon filtering...")
    all_series_raw = []
    for _, row in df.iterrows():
        all_series_raw.append(np.array(row["target"], dtype=np.float32))
    if MAX_SERIES is not None:
        all_series_raw = all_series_raw[:MAX_SERIES]

# ── Evaluation ──────────────────────────────────────────────────────
results = {}

for horizon in HORIZONS:
    print(f"\n{'='*60}")
    print(f"  Horizon K = {horizon}, Context = {CONTEXT_LEN}")
    print(f"{'='*60}")

    # Filter series that are long enough for this horizon
    valid_series = [s for s in all_series if len(s) >= CONTEXT_LEN + horizon]
    if len(valid_series) == 0:
        print(f"  SKIP: No series long enough (need >= {CONTEXT_LEN + horizon})")
        results[horizon] = {"mse": float("nan"), "mae": float("nan"), "n_series": 0}
        continue

    print(f"  Evaluating on {len(valid_series)} series...")

    all_mse = []
    all_mae = []

    # Process in batches
    for batch_start in range(0, len(valid_series), BATCH_SIZE):
        batch_series = valid_series[batch_start : batch_start + BATCH_SIZE]

        # Extract context (last CONTEXT_LEN before the test window) and ground truth
        contexts = []
        ground_truths = []
        for s in batch_series:
            # Test window = last `horizon` steps
            # Context = `CONTEXT_LEN` steps immediately before that
            gt = s[-horizon:]
            ctx = s[-(CONTEXT_LEN + horizon) : -horizon]
            contexts.append(torch.tensor(ctx, dtype=torch.float32))
            ground_truths.append(gt)

        # Chronos-2 predict: returns [B, num_samples, horizon]
        forecast = pipeline.predict(
            context=contexts,
            prediction_length=horizon,
            num_samples=20,       # sample 20 trajectories
            limit_prediction_length=False,
        )
        # Use median forecast as point prediction: [B, horizon]
        median_forecast = forecast.median(dim=1).values.cpu().numpy()

        # Compute per-series metrics
        for i, (pred, gt) in enumerate(zip(median_forecast, ground_truths)):
            pred = pred[:len(gt)]  # safety trim
            mse = np.mean((pred - gt) ** 2)
            mae = np.mean(np.abs(pred - gt))
            all_mse.append(mse)
            all_mae.append(mae)

        if (batch_start // BATCH_SIZE) % 10 == 0:
            print(f"    Processed {min(batch_start + BATCH_SIZE, len(valid_series))}/{len(valid_series)} series")

    avg_mse = np.mean(all_mse)
    avg_mae = np.mean(all_mae)
    results[horizon] = {"mse": avg_mse, "mae": avg_mae, "n_series": len(valid_series)}

    print(f"  MSE: {avg_mse:.6f}")
    print(f"  MAE: {avg_mae:.6f}")
    print(f"  N:   {len(valid_series)}")

# ── Summary table ───────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  RESULTS SUMMARY — Chronos-2 on Dominick")
print(f"  Context length: {CONTEXT_LEN}")
print(f"{'='*60}")
print(f"{'Horizon':>10} {'MSE':>12} {'MAE':>12} {'N_series':>10}")
print(f"{'-'*10} {'-'*12} {'-'*12} {'-'*10}")

valid_results = {k: v for k, v in results.items() if not np.isnan(v["mse"])}
for h in HORIZONS:
    r = results[h]
    print(f"{h:>10} {r['mse']:>12.6f} {r['mae']:>12.6f} {r['n_series']:>10}")

if valid_results:
    avg_mse_all = np.mean([v["mse"] for v in valid_results.values()])
    avg_mae_all = np.mean([v["mae"] for v in valid_results.values()])
    print(f"{'-'*10} {'-'*12} {'-'*12} {'-'*10}")
    print(f"{'Avg':>10} {avg_mse_all:>12.6f} {avg_mae_all:>12.6f}")
print(f"{'='*60}")