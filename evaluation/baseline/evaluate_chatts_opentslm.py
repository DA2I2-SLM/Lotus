# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

# Evaluates OpenTSLMSPFixPos2D on the ChatTS benchmark datasets (dataset_a / dataset_b).
# Dataset format mirrors the original ChatTS evaluation (zenodo.org/records/14349206).
# Usage:
#   python evaluate_chatts_opentslm.py

import sys
import json
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger

import getpass
USER_NAME = getpass.getuser()
sys.path.append(f"/home/{USER_NAME}/FoundationalTS/src")
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
from opentslm.model_config import PATCH_SIZE


# ── CONFIG ────────────────────────────────────────────────────────────────────
EXP             = 'chatts_dataset_a'
CHECKPOINT      = f'/scratch/{USER_NAME}/OpenTSLM/results/sincos_patch16/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase1_extend/checkpoints/periodic_checkpoint.pt'                                   # ← set to path for fine-tuned model, or None for base model only
BASE_MODEL_ID   = 'google/gemma-3-270m'                  # ← set this (must match training)
DATASET         = f'/scratch/{USER_NAME}/FoundationalTS/evaluation/chatts/dataset_a.json'
WORKDIR         = f'/home/{USER_NAME}/FoundationalTS'
MAX_NEW_TOKENS  = 512
BATCH_SIZE      = 128
# ─────────────────────────────────────────────────────────────────────────────


def load_model(checkpoint_path: str, base_model_id: str, device: str):
    from opentslm.model.llm.OpenTSLMSPFixPos2D import OpenTSLMSPFixPos2D
    from opentslm.model.llm.OpenTSLMSPFixPos2DTimerXL import OpenTSLMSPFixPos2DTimerXL
    import argparse
    args = argparse.Namespace(
        pe_type="sincos",
        use_only_time_pe=False,
        use_only_channel_pe=False,
        is_pretraining=False,
    )
    model_cls = (
        OpenTSLMSPFixPos2DTimerXL
        if checkpoint_path and "OpenTSLMSPFixPos2DTimerXL" in checkpoint_path
        else OpenTSLMSPFixPos2D
    )
    logger.info(f"Initialising {model_cls.__name__} with LLM: {base_model_id}")
    model = model_cls(args=args, llm_id=base_model_id, device=device).to(device)

    if checkpoint_path is not None:
        logger.info(f"Loading encoder + projector weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.encoder.load_state_dict(checkpoint["encoder_state"])
        model.projector.load_state_dict(checkpoint["projector_state"])
        epoch = checkpoint.get("epoch", "?")
        val_loss = checkpoint.get("val_loss", "?")
        logger.info(f"Checkpoint: epoch={epoch}, val_loss={val_loss}")
    else:
        logger.info("No checkpoint provided — running with base model weights only")

    model.eval()
    return model


def build_samples(dataset: list, eos_token: str) -> list:
    """Convert ChatTS JSON records into the dict format expected by model.generate()."""
    samples = []
    for record in dataset:
        question = record["question"]
        gold = record.get("answer", "")
        raw_ts = record["timeseries"]   # list of lists  [C, L]

        time_series = []
        time_series_text = []

        for i, ts_vals in enumerate(raw_ts):
            series = torch.tensor(ts_vals, dtype=torch.float32)
            mean = series.mean().item()
            std = series.std().item()
            normalized = (series - mean) / (std + 1e-8)

            if len(raw_ts) == 1:
                text = f"This is the time series, it has mean {mean:.4f} and std {std:.4f}"
            else:
                text = f"This is time series {i}, it has mean {mean:.4f} and std {std:.4f}"

            time_series.append(normalized)
            time_series_text.append(text)

        samples.append({
            "pre_prompt":        question + eos_token,
            "post_prompt":       "Answer:",
            "time_series":       time_series,
            "time_series_text":  time_series_text,
            "answer":            gold,
        })

    return samples


def answer_question_list(model, samples: list) -> dict:
    """Run inference — mirrors the role of LLMClient.llm_batch_generate() in the original."""
    loader = DataLoader(
        samples,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
            batch, patch_size=PATCH_SIZE
        ),
    )

    answer_dict = {}
    idx = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating"):
            generated = model.generate(batch, max_new_tokens=MAX_NEW_TOKENS)
            for response in generated:
                answer_dict[idx] = {"response": response}
                idx += 1

    return answer_dict


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load model
    model = load_model(CHECKPOINT, BASE_MODEL_ID, device)

    # Load dataset
    dataset = json.load(open(DATASET))
    logger.info(f"Loaded {len(dataset)} samples from {DATASET}")

    # Build samples
    samples = build_samples(dataset, eos_token=model.get_eos_token())

    # Output dir
    exp_dir = os.path.join(WORKDIR, f"exp/{EXP}")
    logger.info(f"Experiment directory: {exp_dir}")
    os.makedirs(exp_dir, exist_ok=True)

    # Generation
    logger.info("Start Generation...")
    question_list = [s["pre_prompt"] for s in samples]
    answer = answer_question_list(model, samples)

    # Prepare and save results (same format as original ChatTS script)
    generated_answer = []
    for idx, ans in answer.items():
        generated_answer.append({
            "idx":           idx,
            "question_text": question_list[idx],
            "response":      ans["response"],
            "gold":          samples[idx]["answer"],
        })

    output_file = os.path.join(exp_dir, "generated_answer.json")
    json.dump(generated_answer, open(output_file, "wt"), ensure_ascii=False, indent=4)
    logger.info(f"Results saved to {output_file}")
