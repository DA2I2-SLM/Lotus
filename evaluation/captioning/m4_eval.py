#!/usr/bin/env python3
"""
M4 Caption Generation Evaluation
---------------------------------
Loads a trained OpenTSLMSPFixPos2D checkpoint, runs greedy generation on the
M4 test split, and reports ROUGE-1/2/L scores between generated and gold captions.

Uses model.llm.generate() directly with inputs_embeds (HuggingFace batched
generation) instead of the custom token-by-token loop — much faster for
text-only captioning tasks that have no TS patch output.

Results are saved to --out_dir as:
  predictions.jsonl   — one JSON line per sample (id, gold, generated)
  metrics.json        — aggregated ROUGE scores
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

# Same src-path bootstrap as evaluate_tsqa_opentslm.py (line 33) and
# curriculum_learning.py (line 6). Without this, running this file directly
# fails at `from opentslm...` because the package lives under src/ and isn't
# pip-installed.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import evaluate
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from opentslm.model.llm.OpenTSLMSPFixPos2D import OpenTSLMSPFixPos2D
from opentslm.model.llm.OpenTSLMSPFixPos2DTimerXL import OpenTSLMSPFixPos2DTimerXL
from opentslm.time_series_datasets.m4.M4QADataset import M4QADataset
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate M4 captioning")
    parser.add_argument("--ckpt", required=True,
                        help="Path to best_model.pt checkpoint")
    parser.add_argument("--llm_id", default="google/gemma-3-270m",
                        help="HuggingFace LLM ID used during training")
    parser.add_argument("--device", default="cuda",
                        help="Device: cuda or cpu")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size — larger is faster, limited by GPU memory")
    parser.add_argument("--patch_size", type=int, default=16,
                        help="Patch size used during training")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Max tokens to generate per caption")
    parser.add_argument("--pe_type", default="nope",
                        help="Positional encoding type used during training")
    parser.add_argument("--out_dir", default="./m4_eval_results",
                        help="Directory to save predictions and metrics")
    parser.add_argument("--use_freq_embed", action="store_true",
                        help="Rebuild the encoder with the rfft frequency branch (must match training).")
    parser.add_argument("--enc_dec_type", default="transformer",
                        choices=["transformer", "mlp"],
                        help="Encoder/decoder backbone used during training.")
    parser.add_argument("--mlp_enc_hidden_dim", type=int, default=None)
    parser.add_argument("--mlp_enc_ff_dim",     type=int, default=None)
    parser.add_argument("--mlp_dec_hidden_dim", type=int, default=None)
    parser.add_argument("--mlp_dec_ff_dim",     type=int, default=None)
    return parser.parse_args()


def load_model(args):
    # Auto-detect freq branch from checkpoint keys so a forgotten flag doesn't
    # turn into an obscure load_state_dict "Unexpected key(s)" failure.
    use_freq_embed = bool(getattr(args, "use_freq_embed", False))
    try:
        _ckpt_state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        _enc_state = _ckpt_state.get("encoder_state") if isinstance(_ckpt_state, dict) else None
        if _enc_state is not None and (
            "freq_embed.weight" in _enc_state or "concat_proj.weight" in _enc_state
        ):
            if not use_freq_embed:
                print(
                    "Checkpoint contains freq_embed/concat_proj weights — "
                    "auto-enabling use_freq_embed=True."
                )
                use_freq_embed = True
        del _ckpt_state, _enc_state
    except Exception as _exc:
        print(f"[m4_eval] freq-branch auto-detect skipped: {_exc!r}")

    model_args = SimpleNamespace(
        pe_type=args.pe_type,
        patch_size=args.patch_size,
        use_only_time_pe=False,
        use_only_channel_pe=False,
        use_freq_embed=use_freq_embed,
        enc_dec_type=getattr(args, "enc_dec_type", "transformer"),
        mlp_enc_hidden_dim=getattr(args, "mlp_enc_hidden_dim", None),
        mlp_enc_ff_dim=getattr(args, "mlp_enc_ff_dim", None),
        mlp_dec_hidden_dim=getattr(args, "mlp_dec_hidden_dim", None),
        mlp_dec_ff_dim=getattr(args, "mlp_dec_ff_dim", None),
    )
    model_cls = (
        OpenTSLMSPFixPos2DTimerXL
        if "OpenTSLMSPFixPos2DTimerXL" in args.ckpt
        else OpenTSLMSPFixPos2D
    )
    model = model_cls(
        args=model_args,
        llm_id=args.llm_id,
        device=args.device,
    )
    model.load_from_file(args.ckpt)
    model.to(args.device)
    model.eval()
    print(f"Loaded checkpoint: {args.ckpt}")
    return model


def collate(batch):
    return extend_time_series_to_match_patch_size_and_aggregate(batch)


@torch.no_grad()
def generate_batch(model, batch, max_new_tokens):
    """
    Build prompt embeddings via pad_and_apply_batch, then call model.llm.generate()
    directly for batched HuggingFace generation (much faster than the custom loop).
    Returns a list of decoded strings, one per sample.
    """
    inputs_embeds, attention_mask, _, _, _, _, _ = model.pad_and_apply_batch(
        batch, return_targets=False
    )

    output_ids = model.llm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,         # greedy
        pad_token_id=model.tokenizer.pad_token_id,
        eos_token_id=model.tokenizer.eos_token_id,
    )

    # When inputs_embeds is passed (no input_ids), HuggingFace generate() returns
    # ONLY the newly generated token IDs — prompt tokens are not included.
    return model.tokenizer.batch_decode(output_ids, skip_special_tokens=True)


def compute_rouge(predictions, references):
    rouge = evaluate.load("rouge")
    return rouge.compute(predictions=predictions, references=references)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model = load_model(args)

    eos = model.get_eos_token()
    test_ds = M4QADataset("test", eos)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    print(f"Test set size: {len(test_ds)}  |  batch size: {args.batch_size}  |  max_new_tokens: {args.max_new_tokens}")

    predictions_path = os.path.join(args.out_dir, "predictions.jsonl")
    all_generated = []
    all_gold = []

    with open(predictions_path, "w") as fp:
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Generating captions")):
            generated_texts = generate_batch(model, batch, args.max_new_tokens)

            for sample, generated in zip(batch, generated_texts):
                gold = sample["answer"]
                all_generated.append(generated)
                all_gold.append(gold)
                fp.write(json.dumps({
                    "id":        sample.get("id", ""),
                    "gold":      gold,
                    "generated": generated,
                }) + "\n")

            # Every 100 batches: print a qualitative example + running ROUGE
            if (batch_idx + 1) % 100 == 0:
                sample_id = batch[0].get("id", "")
                n_so_far = len(all_generated)
                print(f"\n{'='*70}")
                print(f"[Step {batch_idx + 1}]  samples so far: {n_so_far}  |  id: {sample_id}")
                print(f"  GOLD : {all_gold[-len(batch)][:200]}")
                print(f"  PRED : {all_generated[-len(batch)][:200]}")
                running = compute_rouge(all_generated, all_gold)
                print(f"  ROUGE (running) — "
                      f"R1={running['rouge1']:.4f}  "
                      f"R2={running['rouge2']:.4f}  "
                      f"RL={running['rougeL']:.4f}")
                print(f"{'='*70}", flush=True)

    print(f"Saved {len(all_generated)} predictions to {predictions_path}")

    print("Computing ROUGE scores...")
    scores = compute_rouge(all_generated, all_gold)

    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w") as fp:
        json.dump(scores, fp, indent=2)

    print("\n=== M4 Captioning Results ===")
    for k, v in scores.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nMetrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
