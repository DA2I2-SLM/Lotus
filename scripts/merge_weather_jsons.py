"""Merge per-ctx weather eval JSONs into single files expected by make_finetune_comparison.py."""
import json, glob, sys
from pathlib import Path

OUT_DIR = Path("output_job/eval_1b_results")

for mode in ("zeroshot", "finetune"):
    parts = []
    for ctx in (512, 1024, 2048, 3072):
        p = OUT_DIR / f"eval_1b_{mode}_weather_ctx{ctx}.json"
        if not p.exists():
            print(f"MISSING: {p}", file=sys.stderr)
            continue
        parts.extend(json.loads(p.read_text()))
    if parts:
        out = OUT_DIR / f"eval_1b_{mode}_weather.json"
        out.write_text(json.dumps(parts, indent=2))
        print(f"Wrote {len(parts)} entries → {out}")
    else:
        print(f"No parts found for {mode}", file=sys.stderr)
