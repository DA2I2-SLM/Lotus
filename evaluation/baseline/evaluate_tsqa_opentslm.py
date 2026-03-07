# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import re
import sys
import json
import argparse
from typing import Dict, Any, List
import itertools

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append("/home/s223540177/FoundationalTS")
from opentslm.time_series_datasets.TSQADataset import TSQADataset
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
from opentslm.model_config import PATCH_SIZE


def evaluate_tsqa_metrics(predictions: List[str], gold_answers: List[str]) -> Dict[str, Any]:
    """
    Compute accuracy for TSQA predictions.
    Compares only the first 3 characters (e.g. "(a)") of each answer.
    """
    correct = 0
    for pred, gold in zip(predictions, gold_answers):
        gt_clean = gold.lower().strip()[:3]
        pred_clean = pred.lower().strip()[:3]

        # Check for "Answer: (a)" style output
        answer_match = re.search(r'answer:\s*(.+)', pred_clean, re.IGNORECASE)
        if answer_match:
            pred_clean = answer_match.group(1).strip()[:3]

        if gt_clean == pred_clean:
            correct += 1

    accuracy = correct / len(predictions) if predictions else 0.0
    return {"accuracy": accuracy}


def load_model(checkpoint_path: str, base_model_id: str, args, device: str):
    """
    Reconstruct OpenTSLMSPFixPos2D and load trained encoder+projector weights
    from a training checkpoint.

    The LLM backbone (e.g. gemma-3-270m) is loaded fresh from HuggingFace —
    only the encoder and projector were saved during training.
    """
    from opentslm.model.llm.OpenTSLMSPFixPos2D import OpenTSLMSPFixPos2D

    print(f"Initialising OpenTSLMSPFixPos2D with LLM: {base_model_id}")
    model = OpenTSLMSPFixPos2D(args=args, llm_id=base_model_id, device=device).to(device)

    print(f"Loading encoder + projector weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.encoder.load_state_dict(checkpoint["encoder_state"])
    model.projector.load_state_dict(checkpoint["projector_state"])

    epoch = checkpoint.get("epoch", "?")
    val_loss = checkpoint.get("val_loss", "?")
    print(f"Checkpoint: epoch={epoch}, val_loss={val_loss}")

    model.eval()
    return model


def run_evaluation(model, dataset, device: str, max_new_tokens: int = 16) -> Dict[str, Any]:
    """
    Run evaluation using the model's own generate() method — the same path
    used during training in curriculum_learning.py's _evaluate_stage().
    """
    loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=PATCH_SIZE
        ),
    )


    predictions = []
    gold_answers = []
    detailed_results = []

    print(f"\nRunning inference on {len(dataset)} samples...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            text_outputs, _ = model.generate(batch, max_new_tokens=max_new_tokens)

            for sample, pred in zip(batch, text_outputs):
                gold = sample["answer"]
                predictions.append(pred)
                gold_answers.append(gold)
                detailed_results.append({
                    "prompt": sample.get("post_prompt", ""),
                    "gold": gold,
                    "prediction": pred,
                })

    metrics = evaluate_tsqa_metrics(predictions, gold_answers)
    return metrics, detailed_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate OpenTSLMSPFixPos2D on the TSQA dataset.")
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to best_model.pt checkpoint from training",
    )
    parser.add_argument(
        "--base_model_id",
        type=str,
        required=True,
        help="HuggingFace model id used during training (e.g. google/gemma-3-270m)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=16,
        help="Max new tokens to generate (default: 16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: cuda / cpu (default: auto)",
    )
    # --- args required by OpenTSLMSPFixPos2D ---
    parser.add_argument("--pe_type", type=str, default="sincos")
    parser.add_argument("--use_only_time_pe", default=False, action="store_true")
    parser.add_argument("--use_only_channel_pe", default=False, action="store_true")
    parser.add_argument("--is_pretraining", default=False, action="store_true")

    args = parser.parse_args()

    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load model
    model = load_model(
        checkpoint_path=args.checkpoint,
        base_model_id=args.base_model_id,
        args=args,
        device=device,
    )

    # Load dataset
    print("Loading TSQA test dataset...")
    dataset = TSQADataset(
        split="test",
        EOS_TOKEN=model.get_eos_token(),
    )
    print(f"Loaded {len(dataset)} test samples")

    # Run evaluation
    metrics, detailed_results = run_evaluation(
        model=model,
        dataset=dataset,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Base model : {args.base_model_id}")
    print(f"Samples    : {len(detailed_results)}")
    for metric, value in metrics.items():
        print(f"{metric:12s}: {value:.4f}  ({value*100:.1f}%)")

    # Show first few examples
    print("\nFirst 5 predictions:")
    print("-" * 60)
    for r in detailed_results[:5]:
        print(f"  Gold      : {r['gold']}")
        print(f"  Predicted : {r['prediction']}")
        print()

    # Save results next to the checkpoint
    import os
    out_dir = os.path.join(os.path.dirname(args.checkpoint), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    predictions_file = os.path.join(out_dir, "test_predictions.jsonl")
    metrics_file = os.path.join(out_dir, "metrics.json")

    with open(predictions_file, "w") as f:
        for r in detailed_results:
            f.write(json.dumps(r) + "\n")

    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Predictions saved to : {predictions_file}")
    print(f"Metrics saved to     : {metrics_file}")


if __name__ == "__main__":
    main()