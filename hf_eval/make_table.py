#!/usr/bin/env python3
"""Render the OpenTSLM-paper-style Accuracy/F1 table from summary.json files.

Usage:
  python make_table.py <results_root>
  (defaults to /scratch/$USER/FoundationalTS/hf_eval_results)
"""
import glob
import getpass
import json
import os
import sys

USER = getpass.getuser()
root = sys.argv[1] if len(sys.argv) > 1 else f"/scratch/{USER}/FoundationalTS/hf_eval_results"

TASK_LABEL = {"har": "HAR-CoT", "sleep": "Sleep-CoT", "ecg": "ECG-QA-CoT"}
TASK_ORDER = ["har", "sleep", "ecg"]

rows = []
for summ in sorted(glob.glob(os.path.join(root, "*", "summary.json"))):
    with open(summ) as f:
        s = json.load(f)
    repo = s.get("repo_id", os.path.basename(os.path.dirname(summ)))
    mtype = s.get("model_type", "?")
    for task in TASK_ORDER:
        m = s.get("tasks", {}).get(task)
        if not m:
            continue
        if "error" in m:
            rows.append((repo, mtype, task, None))
        else:
            rows.append((repo, mtype, task, m))

# Markdown table
print(f"# OpenTSLM eval results (root: {root})\n")
print("| Model (HF repo) | Type | Task | N | Accuracy | Macro-F1 | Weighted-F1 |")
print("|---|---|---|---:|---:|---:|---:|")
for repo, mtype, task, m in rows:
    tl = TASK_LABEL.get(task, task)
    if m is None:
        print(f"| {repo} | {mtype} | {tl} | - | ERROR | ERROR | ERROR |")
    else:
        print(f"| {repo} | {mtype} | {tl} | {m['n']} | {m['accuracy']*100:.2f} | "
              f"{m['macro_f1']*100:.2f} | {m['weighted_f1']*100:.2f} |")

# Also a compact per-model view
print("\n## Per-model (Accuracy / Macro-F1, %)\n")
by_model = {}
for repo, mtype, task, m in rows:
    by_model.setdefault((repo, mtype), {})[task] = m
print("| Model | " + " | ".join(TASK_LABEL[t] for t in TASK_ORDER) + " |")
print("|---|" + "---|" * len(TASK_ORDER))
for (repo, mtype), tasks in by_model.items():
    cells = []
    for t in TASK_ORDER:
        m = tasks.get(t)
        if not m:
            cells.append("—")
        elif m is None:
            cells.append("ERR")
        else:
            cells.append(f"{m['accuracy']*100:.1f} / {m['macro_f1']*100:.1f}")
    print(f"| {repo} | " + " | ".join(cells) + " |")
