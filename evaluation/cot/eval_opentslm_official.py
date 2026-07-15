#!/usr/bin/env python3
"""Evaluate an official OpenTSLM SP checkpoint (from HF Hub) on HAR + Sleep CoT.

Uses OpenTSLM.load_pretrained(repo_id) which:
  - Downloads the official OpenTSLM/gemma-3-270m-{sleep,har,...}-sp checkpoint.
  - Instantiates OpenTSLMSP (their base class, transformer-encoder architecture).
  - Loads the saved encoder/projector/LoRA state.

This is separate from eval_cot_phase2.py which targets our OpenTSLMSPFixPos2D
(MLP-encoder) checkpoints. Reuses the same label-extraction + F1/accuracy logic.
"""

import argparse
import getpass
import json
import os
import re
import sys
from collections import defaultdict
from typing import List

import torch
from tqdm import tqdm

USER = getpass.getuser()
sys.path.insert(0, f"/home/{USER}/FoundationalTS/src")


def setup_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def normalise(s: str) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


_ANSWER_RE = re.compile(r"answer\s*[:\-]\s*([^\n.<]+)", re.IGNORECASE)


def extract_answer_suffix(text: str) -> str:
    if not text:
        return ""
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).strip().rstrip(".").strip()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if lines and len(lines[-1]) <= 40:
        return lines[-1].rstrip(".").strip()
    return ""


def extract_label_from_text(text: str, candidates: List[str]) -> str:
    if not text:
        return ""
    suffix = extract_answer_suffix(text)
    if suffix and candidates:
        n_suffix = normalise(suffix)
        for c in candidates:
            if normalise(c) == n_suffix:
                return c
        for c in candidates:
            if normalise(c) in n_suffix or n_suffix in normalise(c):
                return c
        return suffix
    if suffix:
        return suffix
    t = normalise(text)
    for c in candidates:
        if normalise(c) in t:
            return c
    return ""


def compute_f1(label_golds: List[str], label_preds: List[str]) -> dict:
    classes = sorted({normalise(l) for l in label_golds if l})
    per_class = {}
    tp_total = fp_total = fn_total = 0
    for c in classes:
        tp = sum(1 for g, p in zip(label_golds, label_preds) if normalise(g) == c and normalise(p) == c)
        fp = sum(1 for g, p in zip(label_golds, label_preds) if normalise(g) != c and normalise(p) == c)
        fn = sum(1 for g, p in zip(label_golds, label_preds) if normalise(g) == c and normalise(p) != c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1,
                        "support": sum(1 for g in label_golds if normalise(g) == c)}
        tp_total += tp
        fp_total += fp
        fn_total += fn
    macro_f1 = (sum(v["f1"] for v in per_class.values()) / len(per_class)) if per_class else 0.0
    accuracy = sum(1 for g, p in zip(label_golds, label_preds)
                   if normalise(g) == normalise(p) and g) / max(len(label_golds), 1)
    return {"accuracy": accuracy, "macro_f1": macro_f1, "per_class": per_class}


def _coerce_row(row):
    """OpenTSLMSP.pad_and_apply_batch expects torch tensors for time_series; our
    QADataset rows may contain numpy arrays or Python lists. Convert in-place."""
    import numpy as np
    ts = row.get("time_series", [])
    if isinstance(ts, list):
        new_ts = []
        for t in ts:
            if isinstance(t, np.ndarray):
                new_ts.append(torch.from_numpy(t).float())
            elif isinstance(t, torch.Tensor):
                new_ts.append(t.float())
            else:
                new_ts.append(torch.tensor(t, dtype=torch.float32))
        row["time_series"] = new_ts
    elif isinstance(ts, np.ndarray):
        row["time_series"] = [torch.from_numpy(ts).float()]
    return row


def eval_task(model, dataset, task_name: str, label_candidates: List[str],
              max_samples: int, max_new_tokens: int, batch_size: int = 8) -> dict:
    n = min(len(dataset), max_samples) if max_samples > 0 else len(dataset)
    records = []
    label_golds: List[str] = []
    label_preds: List[str] = []
    with torch.no_grad():
        with tqdm(total=n, desc=f"Eval {task_name}") as pbar:
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_rows = [_coerce_row(dataset[i]) for i in range(start, end)]
                try:
                    # OpenTSLMSP.generate returns List[str] (not a tuple).
                    text_outs = model.generate(batch_rows, max_new_tokens=max_new_tokens)
                    if isinstance(text_outs, tuple):
                        text_outs = text_outs[0]
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    text_outs = []
                    for r in batch_rows:
                        try:
                            t = model.generate([r], max_new_tokens=max_new_tokens)
                            if isinstance(t, tuple):
                                t = t[0]
                            text_outs.append(t[0] if t else "")
                        except Exception as e:
                            text_outs.append(f"<error: {type(e).__name__}: {e}>")
                except Exception as e:
                    text_outs = [f"<error: {type(e).__name__}: {e}>"] * len(batch_rows)
                for j, row in enumerate(batch_rows):
                    pred = text_outs[j] if j < len(text_outs) and text_outs[j] is not None else ""
                    gold = row.get("answer", "")
                    label_gold = (extract_answer_suffix(gold) or row.get("label") or "").strip()
                    label_pred = extract_label_from_text(pred, label_candidates)
                    label_golds.append(label_gold)
                    label_preds.append(label_pred)
                    records.append({
                        "idx": start + j,
                        "gold": gold,
                        "pred": pred,
                        "label_gold": label_gold,
                        "label_pred": label_pred,
                    })
                pbar.update(end - start)
    cls = compute_f1(label_golds, label_preds)
    metrics = {
        "task": task_name,
        "n_evaluated": n,
        "accuracy": cls["accuracy"],
        "macro_f1": cls["macro_f1"],
        "n_classes": len(cls["per_class"]),
        "per_class": cls["per_class"],
    }
    return {"metrics": metrics, "records": records}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", default="OpenTSLM/gemma-3-270m-sleep-sp",
                    help="Official OpenTSLM HF repo (must end with -sp)")
    ap.add_argument("--tasks", nargs="+", default=["har", "sleep"], choices=["har", "sleep"])
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
    ap.add_argument("--max_samples", type=int, default=0,
                    help="Cap per task; 0 = full split.")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out_dir", default=f"/scratch/{USER}/FoundationalTS/eval_results/opentslm_official_sleep_sp")
    args = ap.parse_args()

    device = setup_device()
    print(f"Device: {device}")

    from opentslm.model.llm.OpenTSLM import OpenTSLM
    print(f"Loading official OpenTSLM ckpt: {args.repo_id}")
    model = OpenTSLM.load_pretrained(args.repo_id, device=device, enable_lora=True)
    model.eval()
    EOS = model.get_eos_token() if hasattr(model, "get_eos_token") else ""

    os.makedirs(args.out_dir, exist_ok=True)
    summary = {}

    for task in args.tasks:
        print(f"\n===== Stage: {task} =====")
        if task == "har":
            from opentslm.time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset
            ds = HARCoTQADataset(split=args.split, EOS_TOKEN=EOS)
            label_candidates = HARCoTQADataset.get_labels()
        elif task == "sleep":
            from opentslm.time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset
            ds = SleepEDFCoTQADataset(split=args.split, EOS_TOKEN=EOS)
            label_candidates = SleepEDFCoTQADataset.get_labels()
        else:
            raise ValueError(f"unknown task: {task}")
        print(f"  label candidates ({len(label_candidates)}): {label_candidates}")

        result = eval_task(model, ds, task, label_candidates,
                           args.max_samples, args.max_new_tokens,
                           batch_size=args.batch_size)
        summary[task] = result["metrics"]
        out_path = os.path.join(args.out_dir, f"{task}_{args.split}_predictions.jsonl")
        with open(out_path, "w") as f:
            for rec in result["records"]:
                f.write(json.dumps(rec) + "\n")
        m = result["metrics"]
        print(f"  ✓ {task}: accuracy={m['accuracy']:.3f}  f1={m['macro_f1']:.3f}  (n={m['n_evaluated']})")
        print(f"  predictions → {out_path}")

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n📊 Summary: {summary_path}")
    print(f"\n{'Task':10s}  {'Accuracy':>10s}  {'F1':>8s}")
    print("-" * 32)
    for task, m in summary.items():
        print(f"{task:10s}  {m['accuracy']:>10.3f}  {m['macro_f1']:>8.3f}")


if __name__ == "__main__":
    main()
