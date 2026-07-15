#!/usr/bin/env python3
"""Evaluate OpenTSLMSPFixPos2D best_model.pt on stage 3/4/5 CoT datasets.

Stages:
  3 - HAR CoT (HARCoTQADataset)         → classification accuracy
  4 - Sleep CoT (SleepEDFCoTQADataset)  → classification accuracy
  5 - ECG-QA CoT (ECGQACoTQADataset)    → exact-match + ROUGE-L

Usage:
    python evaluation/cot/eval_cot_phase2.py \
        --ckpt /scratch/$USER/FoundationalTS/results/sincos_patch32_mlp/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase2_downstream_from_pretrained_nofreq/checkpoints/best_model.pt \
        --tasks har sleep ecg \
        --max_samples 500 \
        --out_dir ./eval_results/phase2_cot/
"""

import argparse
import getpass
import json
import os
import re
import sys
from collections import defaultdict
from typing import List, Tuple

import torch
from tqdm import tqdm

USER = getpass.getuser()
sys.path.insert(0, f"/home/{USER}/FoundationalTS/src")


def setup_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(ckpt_path: str, llm_id: str, device: str,
               lora_r: int = 4, lora_alpha: int = 8):
    from opentslm.model.llm.OpenTSLMSPFixPos2D import OpenTSLMSPFixPos2D

    args = argparse.Namespace(
        pe_type="sincos",
        use_only_time_pe=False,
        use_only_channel_pe=False,
        is_pretraining=False,
        gradient_checkpointing=False,
        patch_size=32,
        enc_dec_type="mlp",
        mlp_enc_hidden_dim=64,
        mlp_enc_ff_dim=256,
        mlp_dec_hidden_dim=64,
        mlp_dec_ff_dim=256,
    )
    print(f"Initialising OpenTSLMSPFixPos2D with LLM={llm_id}")
    model = OpenTSLMSPFixPos2D(args=args, llm_id=llm_id, device=device).to(device)
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # CRITICAL: enable LoRA BEFORE loading checkpoint. Without this, the call
    # to load_lora_state_from_checkpoint raises and gets swallowed by the
    # try/except below, leaving the base un-fine-tuned LLM (→ garbage output).
    if ckpt.get("lora_enabled", False) and "lora_state" in ckpt:
        print(f"  enabling LoRA r={lora_r} alpha={lora_alpha}")
        model.enable_lora(lora_r=lora_r, lora_alpha=lora_alpha)

    # Resized LLM token embeddings — load BEFORE LoRA state so PEFT sees
    # the final embedding shape.
    if "llm_input_embeddings" in ckpt:
        model.llm.get_input_embeddings().weight.data.copy_(
            ckpt["llm_input_embeddings"].to(device)
        )
    if "llm_output_embeddings" in ckpt:
        out_emb = model.llm.get_output_embeddings()
        if out_emb is not None:
            out_emb.weight.data.copy_(ckpt["llm_output_embeddings"].to(device))

    # Encoder/projector/decoder/decoder_projector/mse_projector
    for name in (
        "encoder_state", "projector_state", "decoder_state",
        "decoder_projector_state", "mse_projector_state",
    ):
        if name in ckpt:
            getattr(model, name.replace("_state", "")).load_state_dict(ckpt[name])

    # LoRA adapters (will raise if enable_lora wasn't called above).
    model.load_lora_state_from_checkpoint(ckpt, allow_missing=False)

    print(f"  ckpt epoch={ckpt.get('epoch','?')}, val_loss={ckpt.get('val_loss','?')}")
    model.eval()
    return model


def normalise(s: str) -> str:
    """Lowercase + strip + collapse whitespace + remove punctuation."""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


_ANSWER_RE = re.compile(r"answer\s*[:\-]\s*([^\n.<]+)", re.IGNORECASE)


def extract_answer_suffix(text: str) -> str:
    """Pull the post-'Answer:' suffix. Falls back to last short line."""
    if not text:
        return ""
    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).strip().rstrip(".").strip()
    # fallback: last non-empty line, if it's short (likely a label)
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if lines and len(lines[-1]) <= 40:
        return lines[-1].rstrip(".").strip()
    return ""


def extract_label_from_text(text: str, candidates: List[str]) -> str:
    """First try `Answer: <label>` extraction, then fall back to substring match."""
    if not text:
        return ""
    suffix = extract_answer_suffix(text)
    if suffix and candidates:
        # find candidate that best matches the suffix
        n_suffix = normalise(suffix)
        for c in candidates:
            if normalise(c) == n_suffix:
                return c
        for c in candidates:
            if normalise(c) in n_suffix or n_suffix in normalise(c):
                return c
        return suffix  # unmatched but extracted
    if suffix:
        return suffix
    # last resort: any candidate substring in the whole text
    t = normalise(text)
    for c in candidates:
        if normalise(c) in t:
            return c
    return ""


def rouge_l(pred: str, gold: str) -> float:
    """ROUGE-L F1 via LCS over whitespace tokens."""
    a = normalise(pred).split()
    b = normalise(gold).split()
    if not a or not b:
        return 0.0
    # LCS DP
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    lcs = dp[m][n]
    p = lcs / m
    r = lcs / n
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def collect_labels(dataset, label_field_keys=("answer", "rationale")) -> List[str]:
    """Walk a few samples to gather candidate classification labels.
    Heuristic: short, repeated answers ⇒ class names."""
    counts = defaultdict(int)
    for i, row in enumerate(dataset.dataset):
        if i >= 2000:
            break
        for k in label_field_keys:
            v = row.get(k) if isinstance(row, dict) else None
            if isinstance(v, str) and len(v) <= 30:
                counts[v.strip()] += 1
    # Keep candidates that appear at least 3 times (likely class labels).
    cands = sorted([k for k, c in counts.items() if c >= 3], key=lambda k: -counts[k])
    return cands[:30]


def compute_f1(label_golds: List[str], label_preds: List[str]) -> dict:
    """Per-class precision/recall/F1 + macro-F1 + accuracy."""
    classes = sorted({normalise(l) for l in label_golds if l})
    per_class = {}
    tp_total = fp_total = fn_total = 0
    for c in classes:
        tp = sum(1 for g, p in zip(label_golds, label_preds)
                 if normalise(g) == c and normalise(p) == c)
        fp = sum(1 for g, p in zip(label_golds, label_preds)
                 if normalise(g) != c and normalise(p) == c)
        fn = sum(1 for g, p in zip(label_golds, label_preds)
                 if normalise(g) == c and normalise(p) != c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1,
                        "support": sum(1 for g in label_golds if normalise(g) == c)}
        tp_total += tp
        fp_total += fp
        fn_total += fn
    macro_f1 = (sum(v["f1"] for v in per_class.values()) / len(per_class)) if per_class else 0.0
    micro_prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    micro_rec  = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0
    accuracy = sum(1 for g, p in zip(label_golds, label_preds)
                   if normalise(g) == normalise(p) and g) / max(len(label_golds), 1)
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "per_class": per_class,
    }


def eval_task(model, dataset, task_name: str, label_candidates: List[str],
              max_samples: int, max_new_tokens: int, batch_size: int = 8) -> dict:
    n = min(len(dataset), max_samples) if max_samples > 0 else len(dataset)
    records = []
    label_golds: List[str] = []
    label_preds: List[str] = []
    rouge_sum = 0.0
    with torch.no_grad():
        with tqdm(total=n, desc=f"Eval {task_name}") as pbar:
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_rows = [dataset[i] for i in range(start, end)]
                try:
                    text_outs, _ = model.generate(batch_rows, max_new_tokens=max_new_tokens)
                except torch.cuda.OutOfMemoryError:
                    # Fall back to per-sample for this batch only.
                    torch.cuda.empty_cache()
                    text_outs = []
                    for r in batch_rows:
                        try:
                            t, _ = model.generate([r], max_new_tokens=max_new_tokens)
                            text_outs.append(t[0] if t else "")
                        except Exception as e:
                            text_outs.append(f"<error: {type(e).__name__}>")
                except Exception as e:
                    text_outs = [f"<error: {type(e).__name__}>"] * len(batch_rows)
                for j, row in enumerate(batch_rows):
                    pred = text_outs[j] if j < len(text_outs) and text_outs[j] is not None else ""
                    gold = row.get("answer", "")
                    rl = rouge_l(pred, gold)
                    rouge_sum += rl
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
                        "rouge_l": rl,
                    })
                pbar.update(end - start)
    cls_metrics = compute_f1(label_golds, label_preds)
    metrics = {
        "task": task_name,
        "n_evaluated": n,
        "accuracy": cls_metrics["accuracy"],
        "macro_f1": cls_metrics["macro_f1"],
        "micro_f1": cls_metrics["micro_f1"],
        "rouge_l_mean": rouge_sum / max(n, 1),
        "n_classes": len(cls_metrics["per_class"]),
        "n_label_candidates": len(label_candidates),
        "per_class": cls_metrics["per_class"],
    }
    return {"metrics": metrics, "records": records}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to best_model.pt")
    ap.add_argument("--llm_id", default="google/gemma-3-270m")
    ap.add_argument("--lora_r", type=int, default=4)
    ap.add_argument("--lora_alpha", type=int, default=8)
    ap.add_argument("--tasks", nargs="+", default=["har", "sleep", "ecg"],
                    choices=["har", "sleep", "ecg"])
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
    ap.add_argument("--max_samples", type=int, default=500,
                    help="cap per task; 0 = full split (slow)")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--out_dir", default=f"/scratch/{USER}/FoundationalTS/eval_results/phase2_cot")
    args = ap.parse_args()

    device = setup_device()
    print(f"Device: {device}")
    model = load_model(args.ckpt, args.llm_id, device,
                       lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    EOS = model.get_eos_token() if hasattr(model, "get_eos_token") else ""

    os.makedirs(args.out_dir, exist_ok=True)
    summary = {}

    for task in args.tasks:
        print(f"\n===== Stage: {task} =====")
        if task == "har":
            from opentslm.time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset
            ds = HARCoTQADataset(split=args.split, EOS_TOKEN=EOS)
        elif task == "sleep":
            from opentslm.time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset
            ds = SleepEDFCoTQADataset(split=args.split, EOS_TOKEN=EOS)
        elif task == "ecg":
            from opentslm.time_series_datasets.ecg_qa.ECGQACoTQADataset import ECGQACoTQADataset
            ds = ECGQACoTQADataset(split=args.split, EOS_TOKEN=EOS,
                                   preload_processed_data=False)
        else:
            raise ValueError(f"unknown task: {task}")

        # For classification tasks (HAR/Sleep), pull the canonical label list
        # from the dataset class itself. The earlier heuristic `collect_labels`
        # mined short `answer` strings but CoT `answer` is the full rationale
        # (hundreds of chars), yielding zero candidates. The QADataset subclasses
        # expose `get_labels()` returning the canonical N-class label list.
        label_candidates: List[str] = []
        if task == "har":
            from opentslm.time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset
            label_candidates = HARCoTQADataset.get_labels()
        elif task == "sleep":
            from opentslm.time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset
            label_candidates = SleepEDFCoTQADataset.get_labels()
        if task in ("har", "sleep"):
            print(f"  label candidates ({len(label_candidates)}): {label_candidates}")

        result = eval_task(
            model, ds, task, label_candidates,
            args.max_samples, args.max_new_tokens,
        )
        summary[task] = result["metrics"]
        # Persist per-task predictions
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
