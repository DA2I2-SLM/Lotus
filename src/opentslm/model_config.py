# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

# ---------------------------
# Hyper‑parameters
# ---------------------------

BATCH_SIZE = 16

PRED_LEN = 96  # must match Stage2AllDataset.PRED_LEN

WINDOW_LENS = [2048] # NOTE: condition: WINDOW_LENS // PATCH_SIZE == 0

PATCH_SIZE = 32
NUM_EPOCHS = 20  # allow many but we will early‑stop
EARLY_STOP_PAT = 5  # stop if val loss hasn’t improved for this many epochs
LR_ENCODER = 1e-4
LR_ENCODER_JOINT = 1e-4  # low encoder LR for joint training — preserves Chronos representations while projector+LLM adapt for text tasks
LR_DECODER = 1e-4
LR_PROJECTOR = 1e-4
LR_DECODER_PROJECTOR = 1e-4
WEIGHT_DECAY = 1e-2
GRAD_CLIP_NORM = 1.0
WARMUP_FRAC = 0.03
MAX_SAMPLES = None  # set to an int for quick experiments
JOINT_VAL_FRACTION = 0.25  # fraction of each source's val split used in joint_phase1/joint_phase1_extend (preserves task ratio)
RESULTS_FILE = "test_predictions.jsonl"
EMBED_DIM = 768
ENCODER_OUTPUT_DIM = EMBED_DIM
TRANSFORMER_INPUT_DIM = EMBED_DIM