#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT
"""
Unified OpenTSLM evaluation — TSQA accuracy, MSE next-patch prediction, M4 captioning.

Usage examples:
  # All three tasks
  python evaluate_tsqa_opentslm.py --ckpt /path/best_model.pt --llm_id meta-llama/Llama-3.2-1B

  # Single task
  python evaluate_tsqa_opentslm.py --ckpt /path/best_model.pt --llm_id ... --tasks tsqa
  python evaluate_tsqa_opentslm.py --ckpt /path/best_model.pt --llm_id ... --tasks mse
  python evaluate_tsqa_opentslm.py --ckpt /path/best_model.pt --llm_id ... --tasks m4

  # Multiple tasks
  python evaluate_tsqa_opentslm.py --ckpt /path/best_model.pt --llm_id ... --tasks tsqa mse
"""

import argparse
import json
import os
import re
import sys
from types import SimpleNamespace
from typing import Dict, Any, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from opentslm.model.llm.OpenTSLMSPFixPos2D import OpenTSLMSPFixPos2D
from opentslm.model.llm.OpenTSLMSPFixPos2DTimerXL import OpenTSLMSPFixPos2DTimerXL
from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(ckpt: str, llm_id: str, patch_size: int, pe_type: str,
               use_only_time_pe: bool, use_only_channel_pe: bool,
               lora_r: int, lora_alpha: int, device: str) -> OpenTSLMSPFixPos2D:
    ckpt_state = torch.load(ckpt, map_location="cpu", weights_only=False)
    encoder_state = ckpt_state.get("encoder_state")
    if encoder_state is not None and "patch_embed.weight" in encoder_state:
        ckpt_patch_size = int(encoder_state["patch_embed.weight"].shape[-1])
        if ckpt_patch_size != patch_size:
            print(
                f"Checkpoint patch_size={ckpt_patch_size} overrides requested "
                f"patch_size={patch_size}."
            )
        patch_size = ckpt_patch_size

    cfg = SimpleNamespace(
        patch_size=patch_size,
        pe_type=pe_type,
        use_only_time_pe=use_only_time_pe,
        use_only_channel_pe=use_only_channel_pe,
        is_pretraining=False,
    )
    if 'stage1_mcq' in ckpt:
        model = OpenTSLMSP(llm_id=llm_id, device=device)
        model.load_from_file(ckpt)
        model.to(device)
        model.eval()
        print(f"Loaded checkpoint: {ckpt}")
        return model


    model_cls = (
        OpenTSLMSPFixPos2DTimerXL
        if "OpenTSLMSPFixPos2DTimerXL" in ckpt
        else OpenTSLMSPFixPos2D
    )
    model = model_cls(args=cfg, llm_id=llm_id, device=device)
    if 'lorallm' in ckpt:
        model.enable_lora(lora_r=lora_r, lora_alpha=lora_alpha)
    model.load_from_file(ckpt)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {ckpt}")
    return model


def collate_fn(patch_size: int):
    return lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
        batch, patch_size=patch_size
    )


# ── TSQA evaluation ────────────────────────────────────────────────────────────

def _tsqa_accuracy(predictions: List[str], gold_answers: List[str]) -> float:
    correct = 0
    for pred, gold in zip(predictions, gold_answers):
        gt = gold.lower().strip()[:3]
        pr = pred.lower().strip()[:3]
        m = re.search(r'answer:\s*(.+)', pr, re.IGNORECASE)
        if m:
            pr = m.group(1).strip()[:3]
        if gt == pr:
            correct += 1
    return correct / len(predictions) if predictions else 0.0


@torch.no_grad()
def evaluate_tsqa(model: OpenTSLMSPFixPos2D, patch_size: int,
                  max_new_tokens: int, max_samples: int | None,
                  out_dir: str) -> Dict[str, Any]:
    from opentslm.time_series_datasets.TSQADataset import TSQADataset

    ds = TSQADataset(split="test", EOS_TOKEN=model.get_eos_token())
    if max_samples:
        ds = torch.utils.data.Subset(ds, range(min(max_samples, len(ds))))
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn(patch_size))

    predictions, gold_answers, detailed = [], [], []
    print(f"\n[TSQA] {len(ds)} samples, max_new_tokens={max_new_tokens}")
    for batch in tqdm(loader, desc="TSQA"):
        if isinstance(model, OpenTSLMSP):
            text_outputs = model.generate(batch, max_new_tokens=max_new_tokens)
        else:
            text_outputs, _ = model.generate(batch, max_new_tokens=max_new_tokens)
        for sample, pred in zip(batch, text_outputs):
            gold = sample["answer"]
            predictions.append(pred)
            gold_answers.append(gold)
            detailed.append({"gold": gold, "prediction": pred})

    accuracy = _tsqa_accuracy(predictions, gold_answers)
    metrics = {"accuracy": accuracy, "n_samples": len(predictions)}

    _save_jsonl(out_dir, "tsqa_predictions.jsonl", detailed)
    _save_json(out_dir, "tsqa_metrics.json", metrics)

    print(f"[TSQA] Accuracy: {accuracy:.4f}  ({accuracy*100:.1f}%)")
    _print_examples(detailed, n=3)
    return metrics


# ── MSE next-patch evaluation ──────────────────────────────────────────────────

def _decode_ts_patch_predictions(
    model,
    last_hidden,
    ts_mask,
    ts_patch_targets_shifted,
    ts_patch_valid,
):
    """
    Rebuild the padded per-series patch layout expected by the decoder.

    `ts_mask` only marks real history patches, while targets/valid masks are
    padded to the batch-wide maximum number of patches.
    """
    N_ts_total, N_patches, _ = ts_patch_targets_shifted.shape
    ts_hidden = last_hidden[ts_mask]

    patch_counts = (ts_patch_valid.sum(dim=1) + 1).tolist()
    expected_flat = int(sum(patch_counts))
    if ts_hidden.size(0) != expected_flat:
        raise RuntimeError(
            f"ts_mask extracted {ts_hidden.size(0)} hidden vectors, "
            f"expected {expected_flat} real patch vectors."
        )

    h_enc = model.decoder_projector(ts_hidden.float())
    h_enc = F.layer_norm(h_enc, [h_enc.size(-1)])

    enc_dim = h_enc.size(-1)
    h_enc_3d = h_enc.new_zeros((N_ts_total, N_patches, enc_dim))

    offset = 0
    for s, count in enumerate(patch_counts):
        h_enc_3d[s, :count, :] = h_enc[offset:offset + count]
        offset += count

    return model.decoder(h_enc_3d)


@torch.no_grad()
def _predict_next_patches(model, batch):
    """
    Returns (pred_patches, tgt_patches, valid_mask) or (None, None, None).
    pred/tgt: [N_ts_total, N_patches, P]
    valid_mask: [N_ts_total, N_patches] bool
    """
    if not isinstance(model, (OpenTSLMSPFixPos2D, OpenTSLMSPFixPos2DTimerXL)):
        return None, None, None

    (
        inputs_embeds,
        attention_mask,
        _,
        ts_mask,
        ts_patch_targets_shifted,
        ts_patch_valid,
        _,           # ts_chan_id — unused by MLP decoder
    ) = model.pad_and_apply_batch(batch, return_targets=True)

    if ts_patch_targets_shifted is None or ts_patch_targets_shifted.numel() == 0:
        return None, None, None

    outputs = model.llm(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )

    N_ts_total, N_patches, P = ts_patch_targets_shifted.shape
    last_hidden = outputs.hidden_states[-1]        # [B, L, H]
    pred_patches = _decode_ts_patch_predictions(
        model=model,
        last_hidden=last_hidden,
        ts_mask=ts_mask,
        ts_patch_targets_shifted=ts_patch_targets_shifted,
        ts_patch_valid=ts_patch_valid,
    )

    return pred_patches, ts_patch_targets_shifted, ts_patch_valid


@torch.no_grad()
def evaluate_mse(model: OpenTSLMSPFixPos2D, patch_size: int,
                 batch_size: int, dataset_name: str, max_batches: int,
                 out_dir: str) -> Dict[str, Any]:
    from opentslm.time_series_datasets.ChronosDataset import ChronosDataset

    eos = model.get_eos_token()
    ds = ChronosDataset(mode="test", EOS_TOKEN=eos, dataset_name=dataset_name)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn(patch_size)
    )

    total_mse = total_mae = total_n = 0
    print(f"\n[MSE] {len(ds)} series, batch_size={batch_size}, dataset={dataset_name}")
    for bi, batch in enumerate(tqdm(loader, desc="MSE")):
        if bi >= max_batches:
            break
        pred, tgt, valid = _predict_next_patches(model, batch)
        if pred is None:
            continue

        pred_v = pred.reshape(-1, pred.size(-1))[valid.reshape(-1)]
        tgt_v  = tgt.reshape(-1, tgt.size(-1))[valid.reshape(-1)]

        n = tgt_v.size(0)
        total_mse += F.mse_loss(pred_v, tgt_v).item() * n
        total_mae += F.l1_loss(pred_v, tgt_v).item() * n
        total_n   += n

    avg_mse = total_mse / max(1, total_n)
    avg_mae = total_mae / max(1, total_n)
    metrics = {"mse": avg_mse, "mae": avg_mae, "n_valid_patches": total_n}

    _save_json(out_dir, "mse_metrics.json", metrics)
    print(f"[MSE] MSE={avg_mse:.6f}  MAE={avg_mae:.6f}  valid_patches={total_n}")
    return metrics


# ── M4 captioning evaluation ───────────────────────────────────────────────────

@torch.no_grad()
def evaluate_m4(model: OpenTSLMSPFixPos2D, patch_size: int,
                batch_size: int, max_new_tokens: int,
                out_dir: str) -> Dict[str, Any]:
    try:
        import evaluate as hf_evaluate
    except ImportError:
        print("[M4] 'evaluate' package not installed. Run: pip install evaluate rouge_score")
        return {}

    from opentslm.time_series_datasets.m4.M4QADataset import M4QADataset

    eos = model.get_eos_token()
    ds = M4QADataset("test", eos)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn(patch_size))

    all_generated, all_gold, detailed = [], [], []
    print(f"\n[M4] {len(ds)} samples, batch_size={batch_size}, max_new_tokens={max_new_tokens}")

    for batch in tqdm(loader, desc="M4 captioning"):
        inputs_embeds, attn_mask, _, _, _, _, _ = model.pad_and_apply_batch(
            batch, return_targets=False
        )
        output_ids = model.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=model.tokenizer.pad_token_id,
            eos_token_id=model.tokenizer.eos_token_id,
        )
        generated_texts = model.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for sample, gen in zip(batch, generated_texts):
            gold = sample["answer"]
            all_generated.append(gen)
            all_gold.append(gold)
            detailed.append({"id": sample.get("id", ""), "gold": gold, "generated": gen})

    rouge = hf_evaluate.load("rouge")
    scores = rouge.compute(predictions=all_generated, references=all_gold)
    scores["n_samples"] = len(all_generated)

    _save_jsonl(out_dir, "m4_predictions.jsonl", detailed)
    _save_json(out_dir, "m4_metrics.json", scores)
    print(f"[M4] ROUGE-1={scores['rouge1']:.4f}  ROUGE-2={scores['rouge2']:.4f}  ROUGE-L={scores['rougeL']:.4f}")
    _print_examples(
        [{"gold": g, "prediction": p} for g, p in zip(all_gold, all_generated)], n=2
    )
    return scores


# ── Helpers ────────────────────────────────────────────────────────────────────

def _save_jsonl(out_dir: str, filename: str, records: list):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(records)} records → {path}")


def _save_json(out_dir: str, filename: str, data: dict):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved metrics → {path}")


def _print_examples(records: list, n: int = 3):
    print(f"\nFirst {min(n, len(records))} examples:")
    print("-" * 60)
    for r in records[:n]:
        gold = r.get("gold", "")
        pred = r.get("prediction", r.get("generated", ""))
        print(f"  Gold      : {gold[:120]}")
        print(f"  Predicted : {pred[:120]}")
        print()


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate OpenTSLM: TSQA / MSE / M4 captioning")
    p.add_argument("--ckpt",        required=True,  help="Path to checkpoint (best_model.pt)")
    p.add_argument("--llm_id",      required=True,  help="HuggingFace LLM ID used during training")
    p.add_argument("--tasks",       nargs="+", default=["tsqa", "m4"],
                   choices=["tsqa", "mse", "m4"],   help="Tasks to run (default: all)")
    p.add_argument("--out_dir",     default="./eval_results", help="Output directory")
    p.add_argument("--device",      default=None,   help="cuda / cpu (default: auto)")
    p.add_argument("--patch_size", required=True, type=int, default=64)
    p.add_argument("--pe_type",     default="sincos")
    p.add_argument("--use_only_time_pe",    action="store_true")
    p.add_argument("--use_only_channel_pe", action="store_true")
    p.add_argument("--lora_r",      type=int, default=4)
    p.add_argument("--lora_alpha",  type=int, default=32)
    # TSQA
    p.add_argument("--tsqa_max_new_tokens", type=int, default=16)
    p.add_argument("--tsqa_max_samples",    type=int, default=None)
    # MSE
    p.add_argument("--mse_batch_size",  type=int, default=16)
    p.add_argument("--mse_dataset",     default="all")
    p.add_argument("--mse_max_batches", type=int, default=10_000_000)
    # M4
    p.add_argument("--m4_batch_size",     type=int, default=128)
    p.add_argument("--m4_max_new_tokens", type=int, default=128)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(
        ckpt=args.ckpt,
        llm_id=args.llm_id,
        patch_size=args.patch_size,
        pe_type=args.pe_type,
        use_only_time_pe=args.use_only_time_pe,
        use_only_channel_pe=args.use_only_channel_pe,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        device=device,
    )
    model_patch_size = int(getattr(model, "patch_size", args.patch_size))

    os.makedirs(args.out_dir, exist_ok=True)
    all_metrics = {}

    if "tsqa" in args.tasks:
        all_metrics["tsqa"] = evaluate_tsqa(
            model=model, patch_size=model_patch_size,
            max_new_tokens=args.tsqa_max_new_tokens,
            max_samples=args.tsqa_max_samples, out_dir=args.out_dir,
        )

    if "mse" in args.tasks:
        all_metrics["mse"] = evaluate_mse(
            model=model, patch_size=model_patch_size,
            batch_size=args.mse_batch_size, dataset_name=args.mse_dataset,
            max_batches=args.mse_max_batches, out_dir=args.out_dir,
        )

    if "m4" in args.tasks:
        all_metrics["m4"] = evaluate_m4(
            model=model, patch_size=model_patch_size,
            batch_size=args.m4_batch_size,
            max_new_tokens=args.m4_max_new_tokens, out_dir=args.out_dir,
        )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Checkpoint : {args.ckpt}")
    print(f"LLM        : {args.llm_id}")
    for task, metrics in all_metrics.items():
        print(f"\n[{task.upper()}]")
        for k, v in metrics.items():
            print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    _save_json(args.out_dir, "all_metrics.json", all_metrics)


if __name__ == "__main__":
    main()
