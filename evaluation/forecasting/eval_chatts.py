#!/usr/bin/env python3
"""
ChatTS instruction-following evaluation.

Evaluates a joint_phase2_downstream checkpoint on the ChatTS test split
(the held-out 10% from sft + ift JSONL files, same deterministic split
used during training).

Metrics reported
----------------
  ROUGE-1 / ROUGE-2 / ROUGE-L   (overall and per-subset: sft, ift)
  CE loss                        (teacher-forced perplexity proxy)

Output files
------------
  <out_dir>/predictions.jsonl   — one JSON line per sample
  <out_dir>/metrics.json        — aggregated + per-subset metrics

Usage
-----
  python eval_chatts.py \\
    --ckpt $LOTUS_RESULTS_ROOT/.../joint_phase2_downstream/checkpoints/best_model.pt \\
    --llm_id google/gemma-3-270m \\
    --patch_size 32

  # override ChatTS data root:
  python eval_chatts.py --ckpt ... --chatts_dir /path/to/ChatTS
"""

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from types import SimpleNamespace
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from opentslm.model.llm.OpenTSLMSPFixPos2D import OpenTSLMSPFixPos2D
from opentslm.time_series_datasets.Phase2DownstreamDataset import (
    CHATTS_DIR,
    _format_chatts_row,
)
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate

import evaluate as hf_evaluate


# ── Dataset ──────────────────────────────────────────────────────────────────────

class ChatTSEvalDataset(Dataset):
    """ChatTS test split — same deterministic split used during training."""

    def __init__(
        self,
        eos_token: str,
        chatts_dir: str = CHATTS_DIR,
        splits: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
    ):
        rng = random.Random(seed)
        all_rows: List[dict] = []
        for subset in ("sft", "ift"):
            p = os.path.join(chatts_dir, subset, "train.jsonl")
            if not os.path.exists(p):
                print(f"[ChatTSEvalDataset] {subset}: not found at {p}, skipping")
                continue
            with open(p) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            row = json.loads(line)
                            row["_subset"] = subset
                            all_rows.append(row)
                        except Exception:
                            pass
            print(f"[ChatTSEvalDataset] {subset}: {len(all_rows)} rows so far")

        rng.shuffle(all_rows)
        n = len(all_rows)
        test_start = int(n * (splits[0] + splits[1]))
        test_rows  = all_rows[test_start:]

        samples: List[dict] = []
        skipped = 0
        for row in test_rows:
            try:
                s = _format_chatts_row(row, eos_token)
                if s is not None:
                    s["_subset"]   = row.get("_subset", "unknown")
                    s["_gold_raw"] = row.get("output", "")
                    s["_input"]    = row.get("input", "")
                    samples.append(s)
                else:
                    skipped += 1
            except Exception:
                skipped += 1
        self.samples = samples
        print(f"[ChatTSEvalDataset] test: {len(samples)} samples ({skipped} skipped)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


# ── Generation ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_batch(
    model: OpenTSLMSPFixPos2D,
    batch: List[dict],
    max_new_tokens: int,
) -> List[str]:
    """Batched HuggingFace generation — fast, no TS patch output needed for ChatTS."""
    inputs_embeds, attention_mask, _, _, _, _, _ = model.pad_and_apply_batch(
        batch, return_targets=False
    )
    output_ids = model.llm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.3,
        pad_token_id=model.tokenizer.pad_token_id,
        eos_token_id=model.tokenizer.eos_token_id,
    )
    # HuggingFace returns only newly generated tokens when inputs_embeds is provided.
    return model.tokenizer.batch_decode(output_ids, skip_special_tokens=True)


# ── CE loss (teacher-forced) ─────────────────────────────────────────────────────

@torch.no_grad()
def compute_ce_loss_batch(
    model: OpenTSLMSPFixPos2D,
    batch: List[dict],
) -> float:
    """Average teacher-forced CE loss over the batch (answer tokens only)."""
    device = model.device
    (
        inputs_embeds_prompt,
        attention_mask_prompt,
        _,
        _ts_mask,
        _,
        _,
        _,
    ) = model.pad_and_apply_batch(batch, return_targets=True)

    B = len(batch)
    answers = [s["answer"] for s in batch]
    ans_tok = model.tokenizer(
        answers, return_tensors="pt", padding=True,
        truncation=True, max_length=2048,
    )
    ans_ids  = ans_tok.input_ids.to(device)
    ans_mask = ans_tok.attention_mask.to(device)
    ans_emb  = model.llm.get_input_embeddings()(ans_ids)
    bos_id   = model.tokenizer.bos_token_id

    new_embeds, new_masks, valid_prompt_lens = [], [], []
    for i in range(B):
        valid_i   = int(attention_mask_prompt[i].sum().item())
        a_len     = int(ans_mask[i].sum().item())
        ans_start = (1 if (bos_id is not None and a_len > 0
                           and ans_ids[i, 0].item() == bos_id) else 0)
        a_eff = a_len - ans_start
        new_embeds.append(torch.cat([
            inputs_embeds_prompt[i, :valid_i],
            ans_emb[i, ans_start:a_len],
        ], dim=0))
        new_masks.append(torch.cat([
            attention_mask_prompt[i, :valid_i],
            torch.ones(a_eff, device=device, dtype=attention_mask_prompt.dtype),
        ], dim=0))
        valid_prompt_lens.append(valid_i)

    from torch.nn.utils.rnn import pad_sequence
    inputs_embeds_comb  = pad_sequence(new_embeds, batch_first=True)
    attention_mask_comb = pad_sequence(new_masks,  batch_first=True)

    outputs = model.llm(
        inputs_embeds=inputs_embeds_comb,
        attention_mask=attention_mask_comb,
        return_dict=True,
    )
    logits = outputs.logits.float()   # [B, L_comb, V]

    total_loss = 0.0
    total_toks = 0
    for i in range(B):
        p_len = valid_prompt_lens[i]
        a_len = int(ans_mask[i].sum().item())
        ans_start = (1 if (bos_id is not None and a_len > 0
                           and ans_ids[i, 0].item() == bos_id) else 0)
        a_eff = a_len - ans_start
        if a_eff <= 1:
            continue
        # logits at positions [p_len .. p_len+a_eff-2] predict tokens [ans_start+1 .. a_len-1]
        lo = logits[i, p_len : p_len + a_eff - 1, :]
        la = ans_ids[i, ans_start + 1 : a_len].to(device)
        if lo.shape[0] != la.shape[0] or la.shape[0] == 0:
            continue
        total_loss += F.cross_entropy(lo, la, reduction="sum").item()
        total_toks += la.shape[0]

    return total_loss / total_toks if total_toks > 0 else float("nan")


# ── Model loading ─────────────────────────────────────────────────────────────────

def load_model(args) -> OpenTSLMSPFixPos2D:
    cfg = SimpleNamespace(
        is_pretraining=True,
        use_only_time_pe=False,
        use_only_channel_pe=False,
        use_freq_embed=False,
        patch_size=args.patch_size,
        pe_type=getattr(args, "pe_type", "sincos"),
        enc_dec_type=getattr(args, "enc_dec_type", "transformer"),
        mlp_dec_hidden_dim=getattr(args, "mlp_dec_hidden_dim", None),
        mlp_dec_ff_dim=getattr(args, "mlp_dec_ff_dim", None),
        mlp_enc_hidden_dim=getattr(args, "mlp_enc_hidden_dim", None),
        mlp_enc_ff_dim=getattr(args, "mlp_enc_ff_dim", None),
    )
    model_cls = OpenTSLMSPFixPos2D
    model = model_cls(cfg, llm_id=args.llm_id, device=args.device)

    if args.lora_r > 0:
        model.enable_lora(lora_r=args.lora_r, lora_alpha=args.lora_alpha)
        print(f"LoRA enabled: r={args.lora_r}  alpha={args.lora_alpha}")

    model.load_from_file(args.ckpt)
    model.eval()
    print(f"Loaded: {args.ckpt}")
    return model


# ── Argument parsing ──────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="ChatTS instruction-following evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--ckpt",       required=True,
                    help="Path to best_model.pt from joint_phase2_downstream.")
    ap.add_argument("--llm_id",     default="google/gemma-3-270m")
    ap.add_argument("--device",     default="cuda")
    ap.add_argument("--patch_size", type=int, default=32)

    ap.add_argument("--chatts_dir", default=CHATTS_DIR,
                    help="Root dir containing sft/ and ift/ subdirectories.")
    ap.add_argument("--chatts_splits", type=float, nargs=3,
                    default=[0.8, 0.1, 0.1],
                    metavar=("TRAIN", "VAL", "TEST"),
                    help="Train/val/test fractions — must match training.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Shuffle seed — must match training.")

    ap.add_argument("--batch_size",      type=int, default=8)
    ap.add_argument("--max_new_tokens",  type=int, default=256,
                    help="Max tokens to generate per sample.")
    ap.add_argument("--skip_ce",         action="store_true",
                    help="Skip teacher-forced CE loss computation (faster).")
    ap.add_argument("--sample_frac",     type=float, default=1.0,
                    metavar="FRAC",
                    help="Fraction of test samples to evaluate (e.g. 0.01 for 1%%). "
                         "Samples are taken evenly across channel-count groups. "
                         "Enables --verbose automatically.")
    ap.add_argument("--verbose",         action="store_true",
                    help="Print full input / gold / generated for every sample.")

    ap.add_argument("--lora_r",             type=int, default=16)
    ap.add_argument("--lora_alpha",         type=int, default=32)
    ap.add_argument("--mlp_dec_hidden_dim", type=int, default=None)
    ap.add_argument("--mlp_dec_ff_dim",     type=int, default=None)
    ap.add_argument("--mlp_enc_hidden_dim", type=int, default=None)
    ap.add_argument("--mlp_enc_ff_dim",     type=int, default=None)
    ap.add_argument("--enc_dec_type",       default="transformer")
    ap.add_argument("--pe_type",            default="sincos")

    ap.add_argument("--out_dir", default="./eval_results/chatts")
    return ap.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────────

def main():
    args  = parse_args()
    model = load_model(args)
    os.makedirs(args.out_dir, exist_ok=True)

    eos = model.get_eos_token()
    ds = ChatTSEvalDataset(
        eos_token=eos,
        chatts_dir=args.chatts_dir,
        splits=tuple(args.chatts_splits),
        seed=args.seed,
    )
    if len(ds) == 0:
        print("No test samples found — check --chatts_dir.")
        return

    verbose = args.verbose or (args.sample_frac < 1.0)

    # Group samples by channel count — the 2D encoder requires all samples in
    # a batch to have the same n_channels (mirrors group_channels_separately=True
    # used in training via TaskHomogeneousDistributedBatchSampler).
    from collections import defaultdict as _dd
    groups: Dict[int, List[dict]] = _dd(list)
    for s in ds:
        groups[s["n_channels"]].append(s)

    # Subsample evenly within each channel group when --sample_frac < 1.
    if args.sample_frac < 1.0:
        frac = max(0.0, min(1.0, args.sample_frac))
        groups = {
            n_ch: samps[: max(1, math.ceil(len(samps) * frac))]
            for n_ch, samps in groups.items()
        }
        total_kept = sum(len(v) for v in groups.values())
        print(f"[sample_frac={frac:.3f}] using {total_kept} / {len(ds)} samples")

    def _iter_batches():
        for n_ch, samples in sorted(groups.items()):
            for i in range(0, len(samples), args.batch_size):
                raw = samples[i : i + args.batch_size]
                yield extend_time_series_to_match_patch_size_and_aggregate(
                    raw, patch_size=args.patch_size
                )

    total_batches = sum(
        math.ceil(len(v) / args.batch_size) for v in groups.values()
    )

    rouge_metric = hf_evaluate.load("rouge")

    # Accumulate per-subset
    by_subset: Dict[str, Dict[str, List]] = defaultdict(lambda: {"pred": [], "gold": [], "ce": []})
    all_pred, all_gold = [], []

    pred_path = os.path.join(args.out_dir, "predictions.jsonl")
    with open(pred_path, "w") as fout:
        for batch_idx, batch in enumerate(tqdm(_iter_batches(), total=total_batches, desc="Generating")):
            if not batch:
                continue

            generated = generate_batch(model, batch, args.max_new_tokens)

            ce_loss = float("nan")
            if not args.skip_ce:
                ce_loss = compute_ce_loss_batch(model, batch)

            for sample, gen_text in zip(batch, generated):
                gold = sample.get("_gold_raw", sample["answer"].replace(eos, "").strip())
                subset = sample.get("_subset", "unknown")

                all_pred.append(gen_text)
                all_gold.append(gold)
                by_subset[subset]["pred"].append(gen_text)
                by_subset[subset]["gold"].append(gold)
                by_subset[subset]["ce"].append(ce_loss)

                fout.write(json.dumps({
                    "subset":    subset,
                    "input":     sample.get("_input", ""),
                    "gold":      gold,
                    "generated": gen_text,
                    "ce_loss":   ce_loss,
                }) + "\n")

                if verbose:
                    n_ch  = sample.get("n_channels", "?")
                    print(f"\n{'─'*70}")
                    print(f"  subset={subset}  channels={n_ch}  ce={ce_loss:.4f}")
                    print(f"  INPUT : {sample.get('_input', '')}")
                    print(f"  GOLD  : {gold}")
                    print(f"  PRED  : {gen_text}", flush=True)

            if not verbose and (batch_idx + 1) % 50 == 0:
                r = rouge_metric.compute(predictions=all_pred, references=all_gold)
                print(
                    f"\n[step {batch_idx+1}]  n={len(all_pred)}"
                    f"  R1={r['rouge1']:.4f}  R2={r['rouge2']:.4f}  RL={r['rougeL']:.4f}"
                )
                print(f"  GOLD: {all_gold[-1][:120]}")
                print(f"  PRED: {all_pred[-1][:120]}", flush=True)

    print(f"\nSaved {len(all_pred)} predictions → {pred_path}")

    # ── Aggregate metrics ────────────────────────────────────────────────────────
    overall_rouge = rouge_metric.compute(predictions=all_pred, references=all_gold)

    valid_ce = [v for sub in by_subset.values() for v in sub["ce"]
                if v == v]  # filter NaN
    avg_ce = sum(valid_ce) / len(valid_ce) if valid_ce else float("nan")

    metrics: Dict = {
        "overall": {
            "rouge1": overall_rouge["rouge1"],
            "rouge2": overall_rouge["rouge2"],
            "rougeL": overall_rouge["rougeL"],
            "ce_loss": avg_ce,
            "n": len(all_pred),
        },
        "by_subset": {},
    }

    for subset, data in sorted(by_subset.items()):
        r = rouge_metric.compute(predictions=data["pred"], references=data["gold"])
        valid = [v for v in data["ce"] if v == v]
        metrics["by_subset"][subset] = {
            "rouge1": r["rouge1"],
            "rouge2": r["rouge2"],
            "rougeL": r["rougeL"],
            "ce_loss": sum(valid) / len(valid) if valid else float("nan"),
            "n": len(data["pred"]),
        }

    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Print summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  ChatTS Evaluation Results")
    print(f"{'='*60}")
    ov = metrics["overall"]
    print(f"  {'subset':>10}  {'n':>6}  {'R1':>8}  {'R2':>8}  {'RL':>8}  {'CE':>8}")
    print(f"  {'-'*56}")
    print(
        f"  {'overall':>10}  {ov['n']:>6}  "
        f"{ov['rouge1']:>8.4f}  {ov['rouge2']:>8.4f}  "
        f"{ov['rougeL']:>8.4f}  {ov['ce_loss']:>8.4f}"
    )
    for subset, m in sorted(metrics["by_subset"].items()):
        print(
            f"  {subset:>10}  {m['n']:>6}  "
            f"{m['rouge1']:>8.4f}  {m['rouge2']:>8.4f}  "
            f"{m['rougeL']:>8.4f}  {m['ce_loss']:>8.4f}"
        )
    print(f"\n  Metrics → {metrics_path}")


if __name__ == "__main__":
    main()
