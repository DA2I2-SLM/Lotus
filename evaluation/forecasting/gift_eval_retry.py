#!/usr/bin/env python3
"""
Retry failed GIFT-Eval configs (NaN-in-context datasets) with nanmean/nanstd fix.
Saves to <out_dir>/retry_task<N>.csv
Usage:
    python evaluation/forecasting/gift_eval_retry.py \
        --ckpt results/.../best_model.pt \
        --gift_eval_data /path/to/gift_eval_data \
        --task_id 0 --n_tasks 5
"""

import argparse
import csv
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# Configs lost when retry2 overwrote retry1 — re-evaluate with full fixes applied
RETRY_CONFIGS = [
    ("bitbrains_fast_storage/H", "long"),
    ("bitbrains_fast_storage/H", "medium"),
    ("bitbrains_rnd/H", "medium"),
    ("car_parts_with_missing", "short"),
    ("electricity/D", "short"),
    ("electricity/H", "long"),
    ("electricity/H", "medium"),
    ("electricity/W", "long"),
    ("electricity/W", "medium"),
    ("hierarchical_sales/D", "short"),
    ("kdd_cup_2018_with_missing/D", "medium"),
    ("kdd_cup_2018_with_missing/D", "short"),
    ("restaurant", "short"),
    ("temperature_rain_with_missing", "long"),
    ("temperature_rain_with_missing", "medium"),
    ("temperature_rain_with_missing", "short"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",           required=True)
    ap.add_argument("--llm_id",         default=os.environ.get("FTS_GEMMA_LOCAL") or "google/gemma-3-270m")
    ap.add_argument("--gift_eval_data", required=True)
    ap.add_argument("--task_id",  type=int, default=0)
    ap.add_argument("--n_tasks",  type=int, default=1)
    ap.add_argument("--out_dir",  default="results/OpenTSLMSPFixPos2D")
    ap.add_argument("--prefix",   default="retry3_task")
    args = ap.parse_args()

    os.environ["GIFT_EVAL"] = args.gift_eval_data

    # Import after setting env
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

    # Import the shared helpers from gift_eval_eval
    from gift_eval_eval import (
        build_model_args, CSV_HEADER, evaluate_one,
    )
    from opentslm.model.llm.OpenTSLMSPFixPos2D import OpenTSLMSPFixPos2D

    my_configs = [c for i, c in enumerate(RETRY_CONFIGS) if i % args.n_tasks == args.task_id]
    print(f"Task {args.task_id}/{args.n_tasks}: {len(my_configs)} retry configs", flush=True)

    if not my_configs:
        print("Nothing to do.", flush=True)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    print(f"Loading model from {args.llm_id} ...", flush=True)

    model = OpenTSLMSPFixPos2D(args=build_model_args(), llm_id=args.llm_id, device=device)
    model.load_from_file(args.ckpt)
    model.eval().to(device)
    eos_token = model.get_eos_token()
    print("Model loaded.", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    out_file = os.path.join(args.out_dir, f"{args.prefix}{args.task_id}.csv")

    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)

        for idx, (ds_name, term) in enumerate(my_configs):
            print(f"\n[{idx+1}/{len(my_configs)}] {ds_name}/{term}", flush=True)
            row = evaluate_one(model, eos_token, ds_name, term, device)
            if row is not None:
                writer.writerow(row)
                f.flush()

    print(f"\nDone. Results → {out_file}", flush=True)


if __name__ == "__main__":
    main()
