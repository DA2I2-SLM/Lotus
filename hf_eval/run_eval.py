#!/usr/bin/env python3
"""Evaluate a pretrained OpenTSLM HuggingFace checkpoint on the CoT
classification/QA benchmarks (HAR-CoT, Sleep-CoT, ECG-QA-CoT) and report
Accuracy + F1 the way the OpenTSLM paper does.

This script does NOT modify any repo code. It uses the public factory
`opentslm.model.llm.OpenTSLM.load_pretrained` exactly like the demo scripts
in demo/huggingface/, then computes metrics over the full test split.

Metric convention (matches evaluation/baseline/evaluate_*.py and
evaluation/opentslm/parse_predictions.py):
  * The model is asked to produce a CoT rationale ending in "Answer: <label>".
  * The predicted label = normalized text after the LAST "Answer:".
  * The gold label = dataset "label" field (HAR/Sleep) or normalized text
    after the last "Answer:" of the gold answer (ECG).
  * Accuracy = exact-match rate of (gold == pred) after normalization.
  * F1       = macro-F1 over the class set (sklearn), reported alongside
               weighted-F1 and micro/accuracy for completeness.

Usage:
  python run_eval.py --repo_id OpenTSLM/gemma-3-270m-pt-sleep-sp \
      --tasks sleep ecg har --max_samples 0 --out_dir <dir>
"""

import argparse
import getpass
import json
import os
import re
import sys
import time
from typing import List, Dict

import torch

USER = getpass.getuser()
REPO_ROOT = f"/home/{USER}/FoundationalTS"
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "hf_eval"))

from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score  # noqa: E402

from opentslm.model.llm.OpenTSLM import OpenTSLM  # noqa: E402
from opentslm.model_config import PATCH_SIZE  # noqa: E402
from opentslm.time_series_datasets.util import (  # noqa: E402
    extend_time_series_to_match_patch_size_and_aggregate,
)


# --------------------------------------------------------------------------- #
# Label extraction / normalization
# --------------------------------------------------------------------------- #
def normalize(s: str) -> str:
    """Lowercase, replace _/- with space, drop other punctuation, collapse ws."""
    if s is None:
        return ""
    s = s.lower().strip()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def text_after_last_answer(text: str) -> str:
    """Return the substring after the last 'Answer:' (case-insensitive).

    Falls back to the whole string if no 'Answer:' marker is present.
    Keeps only the first line and trims trailing punctuation / EOS markers.
    """
    if not text:
        return ""
    low = text.lower()
    idx = low.rfind("answer:")
    seg = text[idx + len("answer:"):] if idx != -1 else text
    # strip common EOS/special tokens
    seg = seg.replace("</s>", " ").replace("<eos>", " ").replace("<end_of_turn>", " ")
    seg = seg.strip()
    seg = seg.split("\n")[0].strip()
    seg = seg.strip().rstrip(".,!?;:").strip()
    return seg


def detect_patch_size(model, default: int) -> tuple:
    """Find the patch_size the checkpoint's encoder was built with.

    The published checkpoints can use a different patch_size than the local
    model_config.PATCH_SIZE (which may have been changed for other work). The
    collate MUST pad to a multiple of the *encoder's* patch_size, otherwise the
    time-series patches are out-of-distribution and the LLM emits garbage.
    """
    cands = [model,
             getattr(model, "encoder", None),
             getattr(getattr(model, "model", None), "vision_encoder", None),
             getattr(model, "vision_encoder", None)]
    for obj in cands:
        if obj is None:
            continue
        ps = getattr(obj, "patch_size", None)
        if isinstance(ps, int) and ps > 0:
            return ps, f"{type(obj).__name__}.patch_size"
        pe = getattr(obj, "patch_embed", None)
        if pe is not None and hasattr(pe, "kernel_size"):
            ks = pe.kernel_size
            ks = ks[0] if isinstance(ks, (tuple, list)) else ks
            return int(ks), f"{type(obj).__name__}.patch_embed.kernel_size"
    return default, "fallback(model_config)"


def match_to_canonical(text: str, canonical: List[str]) -> str:
    """Map a free-text label to one of the canonical class labels.

    1) exact normalized match
    2) containment (longest canonical label that is a substring of the text,
       or whose text the prediction is a substring of)
    3) otherwise return the normalized text unchanged (counts as a wrong class)
    """
    nt = normalize(text)
    if not canonical:
        return nt
    norm_map = {c: normalize(c) for c in canonical}
    for c, nc in norm_map.items():
        if nc == nt:
            return c
    cands = [c for c, nc in norm_map.items() if nc and (nc in nt or nt in nc)]
    if cands:
        return max(cands, key=lambda c: len(norm_map[c]))
    return nt


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #
def load_dataset(task: str, eos: str, cap: int = 0):
    """Build the test dataset. `cap` (>0) limits the number of evaluated samples.

    For ECG the cap is passed straight to the constructor because building the
    full dataset preloads PTB-XL signals for *all* splits (~159k train rows),
    which is the dominant init cost. Capping there makes init fast and bounds
    the (very large) ECG-QA test set to a tractable, fixed first-N slice.
    HAR/Sleep build fast, so they are capped later via a seeded random subset.
    """
    if task == "har":
        from opentslm.time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset
        ds = HARCoTQADataset("test", EOS_TOKEN=eos)
        labels = HARCoTQADataset.get_labels()
        return ds, labels, True, "random_subset"
    if task == "sleep":
        from opentslm.time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset
        ds = SleepEDFCoTQADataset("test", EOS_TOKEN=eos)
        labels = SleepEDFCoTQADataset.get_labels()
        return ds, labels, True, "random_subset"
    if task == "ecg":
        from opentslm.time_series_datasets.ecg_qa.ECGQACoTQADataset import ECGQACoTQADataset
        kw = {"preload_processed_data": False}
        if cap and cap > 0:
            kw["max_samples"] = cap
        ds = ECGQACoTQADataset("test", EOS_TOKEN=eos, **kw)
        try:
            labels = ECGQACoTQADataset.get_labels()
        except Exception:
            labels = []
        return ds, labels, False, "first_n"  # open exact-match; ctor already capped
    raise ValueError(f"unknown task: {task}")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(gold: List[str], pred: List[str], closed: bool,
                    canonical: List[str]) -> Dict:
    n = len(gold)
    acc = accuracy_score(gold, pred) if n else 0.0
    if closed and canonical:
        label_set = list(canonical)
    else:
        label_set = sorted(set(gold))  # classes that actually appear in gold
    macro_f1 = f1_score(gold, pred, labels=label_set, average="macro",
                        zero_division=0) if n else 0.0
    weighted_f1 = f1_score(gold, pred, labels=label_set, average="weighted",
                           zero_division=0) if n else 0.0
    micro_f1 = f1_score(gold, pred, labels=label_set, average="micro",
                        zero_division=0) if n else 0.0
    return {
        "n": n,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "micro_f1": float(micro_f1),
        "n_classes": len(label_set),
    }


# --------------------------------------------------------------------------- #
# Eval loop
# --------------------------------------------------------------------------- #
def eval_task(model, task: str, eos: str, max_samples: int, batch_size: int,
              max_new_tokens: int, out_dir: str, patch_size: int,
              split_seed: int = 0) -> Dict:
    ds, canonical, closed, sampling = load_dataset(task, eos, cap=max_samples)
    total = len(ds)
    if sampling == "first_n":
        # ECG: constructor already applied the cap deterministically.
        n = total
        perm = list(range(total))
    else:
        n = total if max_samples in (0, None) or max_samples >= total else max_samples
        if n < total:
            g = torch.Generator().manual_seed(split_seed)
            perm = torch.randperm(total, generator=g)[:n].tolist()
        else:
            perm = list(range(total))

    subset = torch.utils.data.Subset(ds, perm)
    loader = DataLoader(
        subset,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=lambda b: extend_time_series_to_match_patch_size_and_aggregate(
            b, patch_size=patch_size
        ),
    )

    pred_path = os.path.join(out_dir, f"{task}_predictions.jsonl")
    gold_labels: List[str] = []
    pred_labels: List[str] = []
    done = 0
    t0 = time.time()
    print(f"[{task}] dataset_len={total}  evaluating n={n}  sampling={sampling}  "
          f"closed_label_set={closed}  n_canonical={len(canonical)}", flush=True)

    # Force GREEDY decoding: the gemma-3 generation_config ships with
    # do_sample=True, so OpenTSLM*.generate (which forwards **kwargs to
    # llm.generate) samples by default -> nondeterministic, drifts to garbage
    # on a small model. Greedy is deterministic and markedly more coherent.
    GEN = {"do_sample": False}
    with open(pred_path, "w") as fout, torch.no_grad():
        for bi, batch in enumerate(loader):
            try:
                preds = model.generate(batch, max_new_tokens=max_new_tokens, **GEN)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                preds = []
                for s in batch:
                    try:
                        preds.append(model.generate([s], max_new_tokens=max_new_tokens, **GEN)[0])
                    except Exception as e:
                        preds.append(f"<error:{type(e).__name__}>")
            except Exception as e:
                preds = [f"<error:{type(e).__name__}>"] * len(batch)

            for s, p in zip(batch, preds):
                gold_answer = s.get("answer", "")
                # gold label
                if closed:
                    # Prefer the gold ANSWER text ("...Answer: <full canonical name>"):
                    # the dataset `label` field can be a short code (e.g. "n2")
                    # that won't match the canonical names the model outputs
                    # ("Non-REM stage 2"). Falling back to `label` only if needed.
                    raw_gold = text_after_last_answer(gold_answer) or s.get("label") or ""
                    gl = match_to_canonical(raw_gold, canonical)
                    pl = match_to_canonical(text_after_last_answer(p), canonical)
                else:
                    gl = normalize(text_after_last_answer(gold_answer))
                    pl = normalize(text_after_last_answer(p))
                gold_labels.append(gl)
                pred_labels.append(pl)
                rec = {
                    "task": task,
                    "gold_label": gl,
                    "pred_label": pl,
                    "correct": bool(gl == pl),
                    "gold_answer": gold_answer,
                    "pred_text": p,
                }
                for k in ("template_id", "ecg_id", "question_type"):
                    if isinstance(s, dict) and k in s:
                        rec[k] = s[k]
                fout.write(json.dumps(rec) + "\n")
            done += len(batch)

            if bi % 10 == 0 or done >= n:
                rate = done / max(time.time() - t0, 1e-6)
                running_acc = (sum(int(a == b) for a, b in zip(gold_labels, pred_labels))
                               / max(len(gold_labels), 1))
                eta = (n - done) / max(rate, 1e-6)
                print(f"[{task}] {done}/{n}  acc(running)={running_acc:.3f}  "
                      f"{rate:.2f} samp/s  ETA={eta/60:.1f} min", flush=True)
            fout.flush()

    metrics = compute_metrics(gold_labels, pred_labels, closed, canonical)
    metrics.update({"task": task, "dataset_len": total, "sampling": sampling,
                    "elapsed_sec": round(time.time() - t0, 1),
                    "predictions_file": pred_path})
    print(f"[{task}] DONE  acc={metrics['accuracy']:.4f}  "
          f"macro_f1={metrics['macro_f1']:.4f}  weighted_f1={metrics['weighted_f1']:.4f}  "
          f"(n={metrics['n']}, {metrics['elapsed_sec']}s)", flush=True)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", required=True)
    ap.add_argument("--tasks", nargs="+", default=["sleep", "ecg", "har"],
                    choices=["har", "sleep", "ecg"])
    ap.add_argument("--max_samples", type=int, default=0, help="0 = full test split")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=400)
    ap.add_argument("--patch_size", type=int, default=0,
                    help="0 = auto-detect from the checkpoint encoder (recommended)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attn_impl", default="eager",
                    choices=["eager", "sdpa", "flash_attention_2", "default"],
                    help="override base-LLM attention impl (gemma-3 needs eager)")
    ap.add_argument("--llm_dtype", default="bf16", choices=["bf16", "fp32", "fp16"])
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    enable_lora = args.repo_id.endswith("-sp")

    print(f"==== OpenTSLM HF eval ====", flush=True)
    print(f"repo_id        = {args.repo_id}", flush=True)
    print(f"model_type     = {'SP (LoRA)' if enable_lora else 'Flamingo'}", flush=True)
    print(f"device         = {device}", flush=True)
    print(f"tasks          = {args.tasks}", flush=True)
    print(f"max_samples    = {args.max_samples} (0=full)", flush=True)
    print(f"max_new_tokens = {args.max_new_tokens}", flush=True)
    print(f"batch_size     = {args.batch_size}", flush=True)
    print(f"out_dir        = {args.out_dir}", flush=True)

    # --- no-edit compat shims (applied BEFORE the model is constructed) ---
    if enable_lora:  # SP path: encoder must be pre-norm (norm_first=True)
        import encoder_compat  # noqa: F401
    else:  # Flamingo path: unwrap SimpleNamespace(visual=...) bug
        import flamingo_compat  # noqa: F401
    if args.attn_impl != "default":
        import torch as _t
        import model_compat
        _dt = {"bf16": _t.bfloat16, "fp32": _t.float32, "fp16": _t.float16}[args.llm_dtype]
        model_compat.apply(attn_implementation=args.attn_impl, torch_dtype=_dt)

    model = OpenTSLM.load_pretrained(args.repo_id, enable_lora=enable_lora, device=device)
    model.eval()

    # CRITICAL FIX: the checkpoints were trained with RIGHT-padded tokenization,
    # but the gemma tokenizer defaults to padding_side='left'. pad_and_apply_batch
    # slices each text segment as embeds[:length]; with left padding that grabs
    # PAD positions (mask=0) for every non-longest segment, silently dropping the
    # ts_header and post_prompt text -> degenerate generation. Forcing right
    # padding restores coherence (validated: Sleep coherence 5/16 -> 16/16).
    for tok_attr in ("tokenizer", "text_tokenizer"):
        tok = getattr(model, tok_attr, None)
        if tok is not None and hasattr(tok, "padding_side"):
            tok.padding_side = "right"
            print(f"  set {tok_attr}.padding_side = 'right'", flush=True)

    eos = model.get_eos_token()

    if args.patch_size and args.patch_size > 0:
        patch_size, ps_src = args.patch_size, "cli"
    else:
        patch_size, ps_src = detect_patch_size(model, PATCH_SIZE)
    print(f"patch_size     = {patch_size} (source={ps_src}; model_config.PATCH_SIZE={PATCH_SIZE})",
          flush=True)

    summary = {"repo_id": args.repo_id,
               "model_type": "sp" if enable_lora else "flamingo",
               "max_samples": args.max_samples,
               "max_new_tokens": args.max_new_tokens,
               "patch_size": patch_size,
               "batch_size": args.batch_size,
               "tasks": {}}

    for task in args.tasks:
        print(f"\n{'='*60}\nTASK: {task}\n{'='*60}", flush=True)
        try:
            m = eval_task(model, task, eos, args.max_samples, args.batch_size,
                          args.max_new_tokens, args.out_dir, patch_size,
                          split_seed=args.seed)
        except Exception as e:
            import traceback
            traceback.print_exc()
            m = {"task": task, "error": f"{type(e).__name__}: {e}"}
        summary["tasks"][task] = m
        # write summary incrementally after every task
        with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    # final table
    print(f"\n\n{'='*60}\nRESULT TABLE  ({args.repo_id})\n{'='*60}", flush=True)
    print(f"{'Task':12s} {'N':>6s} {'Accuracy':>10s} {'MacroF1':>10s} {'WeightedF1':>12s}", flush=True)
    print("-" * 54, flush=True)
    for task, m in summary["tasks"].items():
        if "error" in m:
            print(f"{task:12s} {'ERR':>6s}  {m['error']}", flush=True)
        else:
            print(f"{task:12s} {m['n']:>6d} {m['accuracy']:>10.4f} "
                  f"{m['macro_f1']:>10.4f} {m['weighted_f1']:>12.4f}", flush=True)
    print(f"\nSummary JSON -> {os.path.join(args.out_dir, 'summary.json')}", flush=True)


if __name__ == "__main__":
    main()
