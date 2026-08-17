# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT
import sys
sys.path.append("./src")  # Ensure src is in the path for imports


import gc
import json
import math
import os
import argparse
import contextlib
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List, Optional, Dict, Any, Callable
from opentslm.time_series_datasets.TSQADataset import TSQADataset
from opentslm.time_series_datasets.m4.M4QADataset import M4QADataset
from opentslm.time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset
from opentslm.time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset
from opentslm.time_series_datasets.ecg_qa.ECGQACoTQADataset import ECGQACoTQADataset
from opentslm.time_series_datasets.ChatTSDataset import ChatTSDataset
from opentslm.time_series_datasets.Stage2AllDataset import Stage2AllDataset
from opentslm.time_series_datasets.Phase1JointDataset import (
    Phase1JointDataset,
    Phase1JointExtendDataset,
    Phase1JointTsqaM4Dataset,
    Phase1JointTsqaOnlyDataset,
    Phase1ChronosOnlyDataset,
    Phase1ChronosFuturePatchDataset,
    Phase1ChronosSeq2SeqDataset,
)
print('import Phase1JointDataset')
from opentslm.time_series_datasets.Phase2JointDataset import Phase2JointDataset
from opentslm.time_series_datasets.Phase2DownstreamDataset import Phase2DownstreamDataset
from opentslm.time_series_datasets.TaskHomogeneousBatchSampler import (
    TaskHomogeneousDistributedBatchSampler,
    TaskWeightedDistributedBatchSampler,
)

from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)
from opentslm.time_series_datasets.processor import OpenTSLMProcessor
import torch
import torch.distributed as dist
# Switch tensor-sharing from the default 'file_descriptor' (memfd) strategy
# to 'file_system' (files in /dev/shm). The default strategy hands one FD
# per shared tensor between dataloader workers and the main rank; with
# persistent_workers=True across many epochs and several workers per rank,
# FDs leak/accumulate and eventually trip
#   RuntimeError: unable to mmap N bytes from file </torch_...>: Cannot allocate memory (12)
# which deadlocks DDP (one rank stuck in NCCL collective waiting for a worker
# that died). 'file_system' uses real files in /dev/shm (we have 126 GB);
# this is the standard PyTorch recommendation for long-running training jobs
# with many workers (see https://pytorch.org/docs/stable/multiprocessing.html
# under "Sharing strategies").
import torch.multiprocessing as _torch_mp
try:
    _torch_mp.set_sharing_strategy("file_system")
except RuntimeError:
    # Already set elsewhere — fine.
    pass
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    FullStateDictConfig,
    StateDictType,
)
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
from opentslm.model.llm.OpenTSLMFlamingoFixPos2D import OpenTSLMFlamingoFixPos2D
from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP
from opentslm.model.llm.OpenTSLMSPFixPos2D import OpenTSLMSPFixPos2D
from opentslm.model.llm.OpenTSLMSPFixPos2DTimerXL import OpenTSLMSPFixPos2DTimerXL
from opentslm.model.llm.OpenTSLMSPMomentSoftPrompt import OpenTSLMSPMomentSoftPrompt

import datetime
from opentslm.logger import get_logger, set_global_verbose
import wandb

from opentslm.model_config import (
    BATCH_SIZE,
    EARLY_STOP_PAT,
    GRAD_CLIP_NORM,
    JOINT_VAL_FRACTION,
    LR_ENCODER,
    LR_ENCODER_JOINT,
    LR_PROJECTOR,
    NUM_EPOCHS,
    PATCH_SIZE,
    WARMUP_FRAC,
    WEIGHT_DECAY,
    LR_DECODER,
    LR_DECODER_PROJECTOR,
    PRED_LEN,
)
# Per-run early-stop patience override (default = model_config EARLY_STOP_PAT)
import os as _os_es



# Warm-start source for `--warm_start_from pretrained`: load weights directly from
# this explicit checkpoint path, bypassing the results/<stage> convention + symlink.
# Override with the WARM_START_PRETRAINED_CKPT env var.
PRETRAINED_WARM_START_CKPT = os.environ.get(
    "WARM_START_PRETRAINED_CKPT",
    "/scratch/s223540177/FoundationalTS/pretrained_weights/"
    "joint_phase1_extend_balanced_lora_nofreq/checkpoints/best_model.pt",
)


# Global stage configuration - users can modify this to mix and match stages
CURRICULUM_STAGES = [
    "stage1_mcq",
    "stage2_captioning",
    "stage3_cot",
    "stage4_sleep_cot",
    "stage5_ecg_cot",
    "stage2_all",
    "joint_phase1",
    "joint_phase1_extend",
    "joint_phase1_extend_lorallm",
    "phase1_mse_then_ce",
    "joint_phase2",
    "joint_phase2_open",
]

AVAILABLE_STAGES = CURRICULUM_STAGES + [
    "joint_phase1_tsqa_m4",
    "joint_phase1_tsqa",
    "phase1_chronos_mse",
    "phase1_chronos_mse_lora",
    "phase1_chronos_mse_full",
    "phase1_chronos_joint_mse",
    "phase1_chronos_seq2seq",
    "phase1_chronos_seq2seq_separate_mse",
    "joint_phase1_extended_separate_mse",
    "joint_phase1_extend_balanced",
    "joint_phase1_extend_balanced_lora",
    "joint_phase1_extend_balanced_m4_by_size",
    "joint_phase1_extend_m4_w05",
    "joint_phase1_extend_m4_w15",
    "joint_phase2_downstream",
    "joint_phase1_flamingo",
    "joint_phase1_enc_mse",
    "joint_phase1_enc_mse_detach",
]

JOINT_PHASE1_STAGES = {
    "joint_phase1",
    "joint_phase1_extend",
    "joint_phase1_extend_lorallm",
    "phase1_chronos_joint_mse",
    "joint_phase1_extended_separate_mse",
    "joint_phase1_extend_balanced",
    "joint_phase1_extend_balanced_lora",
    "joint_phase1_extend_balanced_m4_by_size",
    "joint_phase1_extend_m4_w05",
    "joint_phase1_extend_m4_w15",
    "joint_phase1_tsqa_m4",
    "joint_phase1_tsqa",
    "joint_phase1_flamingo",
    "joint_phase1_enc_mse",
    "joint_phase1_enc_mse_detach",
}
PHASE1_MSE_ONLY_STAGES = {
    "phase1_chronos_mse",
    "phase1_chronos_mse_lora",
    "phase1_chronos_mse_full",
    "phase1_chronos_seq2seq",
    "phase1_chronos_seq2seq_separate_mse",
}
# Stages that unfreeze the full LLM (full fine-tuning, no LoRA). The LLM
# parameters get a dedicated optimizer group with lr=lr_base.
FULL_LLM_FINETUNE_STAGES = {
    "phase1_chronos_mse_full",
}


def _save_llm_state_if_full_ft(checkpoint: dict, model, stage_name: str) -> None:
    """Persist the backbone when it is being fully fine-tuned.

    The SP checkpoint format stores encoder/projector/decoder plus a LoRA
    adapter. That is complete for a FROZEN backbone, but silently loses
    everything a full fine-tune learns -- `llm_state` appears nowhere in the
    save, load or eval paths, so a full-FT run would train ~1B parameters for
    days and write a checkpoint from which only the encoder survives. Costs
    ~2 GB (bf16) and is written only on best-model saves, not periodic ones.
    """
    if not _is_full_ft(stage_name):
        return
    checkpoint["llm_state"] = {k: v.detach().cpu()
                               for k, v in model.llm.state_dict().items()}
    checkpoint["llm_full_ft"] = True


def _is_full_ft(stage_name: str) -> bool:
    """Whether this stage trains the whole LLM rather than a LoRA adapter.

    FORCE_FULL_LLM_FT=1 promotes the CURRENT stage into the full-fine-tune path
    without editing the stage tables, mirroring the existing FORCE_STAGE_LORA
    escape hatch. It exists so joint_phase2_downstream can be run both ways —
    the LoRA arm and the full-FT arm of the capacity ablation — from the same
    code, with no risk of disturbing any other stage's defaults.

    Full FT and LoRA are mutually exclusive (see _unfreeze_llm_if_needed's
    docstring); the LoRA paths below check this and stand down.
    """
    return (stage_name in FULL_LLM_FINETUNE_STAGES
            or os.environ.get("FORCE_FULL_LLM_FT") == "1")
PHASE1_CE_ONLY_STAGES = {"phase1_tsqa_m4_ce"}
JOINT_PHASE1_WITH_M4_STAGES = {
    "joint_phase1_extend",
    "joint_phase1_extend_lorallm",
    "joint_phase1_extended_separate_mse",
    "joint_phase1_extend_balanced",
    "joint_phase1_extend_balanced_lora",
    "joint_phase1_extend_balanced_m4_by_size",
    "joint_phase1_extend_m4_w05",
    "joint_phase1_enc_mse",
    "joint_phase1_extend_m4_w15",
    "joint_phase1_tsqa_m4",
    "joint_phase1_flamingo",
}
JOINT_PHASE2_STAGES = {"joint_phase2", "joint_phase2_open", "joint_phase2_downstream"}
JOINT_STAGES = JOINT_PHASE1_STAGES | JOINT_PHASE2_STAGES
JOINT_CE_MSE_STAGES = {
    "joint_phase1",
    "joint_phase1_extend",
    "joint_phase1_extend_lorallm",
    "joint_phase1_extended_separate_mse",
    "joint_phase2",
    "joint_phase2_open",
    "joint_phase1_flamingo",
    "joint_phase1_enc_mse",
}
JOINT_PHASE1_DETACH_LLM_MSE_STAGES = {
    "joint_phase1_extend_lorallm",
    "joint_phase1_extended_separate_mse",
    # NOTE: joint_phase1_extend_balanced used to be here to "fix" the CE↔MSE
    # gradient conflict on the LLM, but doing so trained the decoder on the
    # PRE-LLM projector output while eval (fixed_inference.py:790-820) always
    # routes through the LLM. The resulting train/eval representation mismatch
    # produced wildly wrong forecasts at inference time (flat/uncorrelated).
    # Removed — the decoder must train on LLM hidden states so eval works.
    # The original CE↔MSE conflict needs a different mitigation if it returns
    # (loss-weight rebalancing, GradNorm/PCGrad, or lower task_weight_synthetic).
}
# Stages where encoder is detached from CE path — only MSE trains encoder.
JOINT_PHASE1_DETACH_ENC_CE_STAGES = {
    "joint_phase1_enc_mse_detach",
}


def print_trainable(model, topk=80):
    if hasattr(model, "module"):
        model = model.module

    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    total = sum(p.numel() for _, p in model.named_parameters())
    trn = sum(k for _, k in trainable)

    print(f"Trainable params: {trn:,}/{total:,} ({100*trn/total:.6f}%)")
    for n, k in sorted(trainable, key=lambda x: -x[1])[:topk]:
        print(f"{k:12,d}  {n}")


class CurriculumTrainer:
    """
    Curriculum learning trainer for OpenTSLM models.
    Trains models stage by stage with shared training logic.
    While this may look like a lot of code, it's actually quite modular.
    We simply train either OpenTSLMSP or OpenTSLMFlamingo, both using the same training loop.
    We train across different stages:
    - stage1_mcq: Trains the model on a time-series MCQ dataset (TSQA)
    - stage2_captioning: Trains the model on a time-series captioning dataset (M4 time series captioning)
    - stage3_cot: Trains the model on a chain-of-thought reasoning dataset (HAR CoT)
    - stage4_sleep_cot: Trains the model on sleep stage classification with chain-of-thought reasoning
    - stage5_ecg_cot: Trains the model on ECG QA with chain-of-thought reasoning

    Features:
    - Automatic loss history tracking saved to loss_history.txt in each stage's checkpoints directory
    - Loss history is appended to when resuming training, preserving all previous epochs
    - Displays previous loss history when resuming training

    If you run this script, you should be able to reproduce our results from the paper.
    All datasets are automatically downloaded and processed.
    """

    def _sanitize_llm_id(self, llm_id: str) -> str:
        """Sanitize llm_id for use in directory names (e.g., meta-llama/Llama-3.2-1B -> Llama3_2_1B)"""
        if not llm_id:
            return "unknown_llm"
        # Take last part after /, replace . and - with _
        name = llm_id.split("/")[-1]
        name = name.replace(".", "_").replace("-", "_")
        # Optionally, remove duplicate underscores
        while "__" in name:
            name = name.replace("__", "_")
        return name

    def __init__(
        self,
        model_type: str,
        device: str = None,
        gradient_checkpointing: bool = False,
        dist_url: str = "env://",
        dist_backend: str = "nccl",
        local_rank: int = int(os.environ.get("LOCAL_RANK", 0)),
        llm_id: str = None,
        args = None
    ):
        """
        Initialize the curriculum trainer.

        Args:
            model_type: Either 'OpenTSLMSP' or 'OpenTSLMFlamingo'
            device: Device to use for training ('cuda', 'mps', or 'cpu')
            gradient_checkpointing: Enable gradient checkpointing
            dist_url: URL used to set up distributed training
            dist_backend: Distributed backend
            local_rank: Local GPU rank
            llm_id: LLM model ID (e.g., 'google/medgemma-2b', 'meta-llama/Llama-3.2-1B')
        """
        self.model_type = model_type
        self.device = device or self._get_device()
        if self.device == "mps":
            print(
                "🚨 Warning: Using MPS, might not be fully compatible with the model. Use CUDA for best results."
            )
        self.llm_id = llm_id
        self.llm_id_safe = self._sanitize_llm_id(llm_id)
        self.args = args

        # Distributed training parameters
        self.gradient_checkpointing = gradient_checkpointing
        self.dist_url = dist_url
        self.dist_backend = dist_backend
        self.local_rank = local_rank

        # Initialize distributed training if needed
        self.rank = 0
        self.world_size = 1
        if self._should_use_distributed():
            self._init_distributed()

        self.model = self._initialize_model()

        # Worker-side preprocessor: bundles tokenization + per-sample TS prep
        # into the DataLoader collate so the GPU rank only does encoder/LLM work.
        # Falls back to extend_time_series_to_match_patch_size_and_aggregate
        # when --use_processor is not set.
        self.processor = None
        if getattr(args, "use_processor", False):
            # DDP doesn't forward arbitrary attributes — reach the underlying module.
            inner = self.model.module if hasattr(self.model, "module") else self.model
            tokenizer = getattr(inner, "tokenizer", None)
            patch_size = getattr(inner, "patch_size", PATCH_SIZE)
            if tokenizer is None:
                # Hard-fail rather than silently falling back to the legacy
                # collate: a silent fallback wasted hours of training before.
                raise RuntimeError(
                    f"--use_processor was set but model {type(inner).__name__} "
                    f"has no `tokenizer` attribute. Either implement it on the "
                    f"model or drop --use_processor."
                )
            self.processor = OpenTSLMProcessor(
                tokenizer=tokenizer, patch_size=int(patch_size)
            )
            if self.rank == 0:
                print(
                    f"⚡ OpenTSLMProcessor enabled (patch_size={int(patch_size)}); "
                    "tokenization + TS prep moved into DataLoader workers."
                )

        patch_suffix = f"_patch{PATCH_SIZE}"
        weight_suffix = "_dynw" if getattr(self.args, "dynamic_loss_weighting", False) else ""
        # Append an enc/dec backbone suffix so MLP runs live in their own tree
        # and don't clobber transformer checkpoints (and vice versa).
        _enc_dec_type = str(getattr(self.args, "enc_dec_type", "transformer")).lower()
        if _enc_dec_type != "transformer":
            patch_suffix = f"{patch_suffix}_{_enc_dec_type}"
        # Frequency-embed runs USED to get an extra "_seed_<data_seed>_freq" on
        # the parent results dir; that created sibling top-level trees per
        # variant. Now both freq and nofreq runs share the same parent — the
        # distinction moves to a per-stage suffix below ("_freq" / "_nofreq"),
        # so freq/nofreq checkpoints for the same stage land side-by-side
        # under the same model dir.
        _results_base = os.environ.get("RESULTS_BASE_DIR", "results")
        if self.args.pe_type is None:
            self.results_dir = os.path.join(f"{_results_base}/base_opentslm", self.llm_id_safe, self.model_type)
        elif self.args.use_only_time_pe:
            self.results_dir = os.path.join(f"{_results_base}/{self.args.pe_type}_time_only{patch_suffix}{weight_suffix}", self.llm_id_safe, self.model_type)
        elif self.args.use_only_channel_pe:
            self.results_dir = os.path.join(f"{_results_base}/{self.args.pe_type}_channel_only{patch_suffix}{weight_suffix}", self.llm_id_safe, self.model_type)
        else:
            self.results_dir = os.path.join(f"{_results_base}/{self.args.pe_type}{patch_suffix}{weight_suffix}", self.llm_id_safe, self.model_type)
        # Per-stage suffix so freq/nofreq variants of the same stage are kept
        # apart. Use _stage_dir(stage_name) below to build a stage-specific
        # path (= self.results_dir / stage_name + _stage_suffix).
        self._stage_suffix = "_freq" if bool(getattr(self.args, "use_freq_embed", False)) else "_nofreq"
        self._create_results_dir()

    def _stage_dir(self, stage_name: str) -> str:
        """Per-stage subdir: ``<results_dir>/<stage_name>{_freq|_nofreq}``.

        All checkpoint and per-stage metrics paths should be built from this so
        a single results parent dir cleanly separates freq vs nofreq variants
        of the same stage.
        """
        return os.path.join(self.results_dir, str(stage_name) + self._stage_suffix)

    def _get_device(self) -> str:
        """Get the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _initialize_model(self):
        """Initialize the specified model type."""
        if self.model_type == "OpenTSLMSP":
            model = OpenTSLMSP(llm_id=self.llm_id, device=self.device).to(self.device)
        elif self.model_type == "OpenTSLMSPFixPos2D":
            model = OpenTSLMSPFixPos2D(args=self.args, llm_id=self.llm_id, device=self.device).to(self.device)
        elif self.model_type == "OpenTSLMSPFixPos2DTimerXL":
            model = OpenTSLMSPFixPos2DTimerXL(args=self.args, llm_id=self.llm_id, device=self.device).to(self.device)
        elif self.model_type == "OpenTSLMSPMomentSoftPrompt":  # ADD THIS BRANCH
            model = OpenTSLMSPMomentSoftPrompt(
                args=self.args,
                llm_id=self.llm_id,
                device=self.device,
            ).to(self.device)

        elif self.model_type == "OpenTSLMFlamingo":
            model = OpenTSLMFlamingo(
                cross_attn_every_n_layers=1,
                gradient_checkpointing=self.gradient_checkpointing,
                llm_id=self.llm_id,
                device=self.device,
                patch_size=getattr(self.args, "patch_size", PATCH_SIZE),
            ).to(self.device)
        elif self.model_type == "OpenTSLMFlamingoFixPos2D":
            model = OpenTSLMFlamingoFixPos2D(
                args=self.args,
                cross_attn_every_n_layers=1,
                gradient_checkpointing=self.gradient_checkpointing,
                llm_id=self.llm_id,
                device=self.device,
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Use DDP for multi-GPU training (simpler than FSDP).
        # find_unused_parameters=True is required because different task types
        # (Chronos MSE vs CE) use different subsets of trainable parameters:
        # - Chronos: decoder_projector + decoder get gradients
        # - TSQA/M4: decoder_projector + decoder do NOT get gradients
        # Without this flag DDP hangs waiting for all-reduce on unused params.
        if self.world_size > 1:
            model = DDP(
                model,
                device_ids=[self.local_rank] if torch.cuda.is_available() else None,
                find_unused_parameters=True,
                static_graph=True,  # stop the per-iter graph re-tracing that
                                    # leaks ~18 layer-output tensors per step
                                    # (Gemma3 has 18 layers) under find_unused.
            )
            if self.rank == 0:
                print(f"Wrapped {self.model_type} with DDP for distributed training")

        return model

    def _get_cast_dtype(self, precision: str):
        """Get cast dtype for mixed precision."""
        if precision == "bf16":
            return torch.bfloat16
        elif precision == "fp16":
            return torch.float16
        else:
            return None

    def _create_results_dir(self):
        """Create the results directory structure.

        Only creates subdirectories for the stages actually being run in this
        invocation (``self.args.stages``). Previously this iterated the full
        ``CURRICULUM_STAGES`` list, so a single ``--stages phase1_chronos_mse``
        run would litter the results tree with 14 empty stage dirs.
        """
        os.makedirs(self.results_dir, exist_ok=True)
        model_dir = self.results_dir
        os.makedirs(model_dir, exist_ok=True)

        active_stages = getattr(self.args, "stages", None) or CURRICULUM_STAGES
        for stage in active_stages:
            stage_dir = self._stage_dir(stage)
            os.makedirs(stage_dir, exist_ok=True)
            os.makedirs(os.path.join(stage_dir, "checkpoints"), exist_ok=True)
            os.makedirs(os.path.join(stage_dir, "results"), exist_ok=True)


    def _get_optimizer(
        self,
        batch_size: int = None,
        lr_encoder: float = None,
        lr_projector: float = None,
        lr_decoder: float = None,
        le_decoder_projector: float = None,
        lr_base: float = None,
        stage_name: str = None,
    ):
        """Get optimizer for the model with configurable learning rates."""
        model = self._get_model()

        if self.model_type in ["OpenTSLMSPFixPos", "OpenTSLMSPFixPos2D", "OpenTSLMSPFixPos2DTimerXL"]:
            # joint_phase2 / joint_phase2_open freeze encoder + projector (CE gradients
            # don't reach them; only decoder head and LoRA train). joint_phase2_downstream
            # mirrors stage3/4/5 single-task optimization within joint training, so it
            # KEEPS enc + proj trainable — each single-task batch then computes its loss
            # exactly like the corresponding standalone stage (stage3_cot for HAR, etc.).
            freeze_enc_and_proj = (
                stage_name in JOINT_PHASE2_STAGES
                and stage_name != "joint_phase2_downstream"
            )

            # First restore all params to trainable, then selectively freeze.
            # This ensures a clean state when stages are chained.
            for p in model.encoder.parameters():
                p.requires_grad = not freeze_enc_and_proj
            for p in model.projector.parameters():
                p.requires_grad = not freeze_enc_and_proj
            if freeze_enc_and_proj and self.rank == 0:
                print(f"🔒 {stage_name}: encoder and projector are frozen — only decoder head and LoRA train.")

            decoder_proj_lr = le_decoder_projector if le_decoder_projector is not None else LR_PROJECTOR

            if freeze_enc_and_proj:
                # Encoder + encoder projector frozen — only decoder head trains (+ LoRA for LLM)
                decoder_lr   = lr_decoder   if lr_decoder   is not None else LR_DECODER
                projector_lr = lr_projector if lr_projector is not None else LR_PROJECTOR  # unused but kept for LoRA LR fallback
                param_groups = [
                    {"params": list(model.decoder.parameters()),           "lr": decoder_lr,      "weight_decay": WEIGHT_DECAY},
                    {"params": list(model.decoder_projector.parameters()), "lr": decoder_proj_lr, "weight_decay": WEIGHT_DECAY},
                ]
            else:
                encoder_lr   = lr_encoder   if lr_encoder   is not None else LR_ENCODER
                projector_lr = lr_projector if lr_projector is not None else LR_PROJECTOR
                decoder_lr   = lr_decoder   if lr_decoder   is not None else LR_DECODER
                param_groups = [
                    {"params": list(model.encoder.parameters()),           "lr": encoder_lr,      "weight_decay": WEIGHT_DECAY},
                    {"params": list(model.projector.parameters()),         "lr": projector_lr,    "weight_decay": WEIGHT_DECAY},
                    {"params": list(model.decoder.parameters()),           "lr": decoder_lr,      "weight_decay": WEIGHT_DECAY},
                    {"params": list(model.decoder_projector.parameters()), "lr": decoder_proj_lr, "weight_decay": WEIGHT_DECAY},
                    {"params": list(model.mse_projector.parameters()),     "lr": decoder_proj_lr, "weight_decay": WEIGHT_DECAY},
                ]

            # Add LoRA parameters if enabled
            if hasattr(model, "lora_enabled") and model.lora_enabled:
                lora_params = model.get_lora_parameters()
                if lora_params:
                    # Use lr_base for LoRA when provided (it's the intended LoRA LR);
                    # fall back to projector_lr only if lr_base was not set.
                    lora_lr = lr_base if lr_base is not None else projector_lr
                    param_groups.append(
                        {"params": lora_params, "lr": lora_lr, "weight_decay": WEIGHT_DECAY}
                    )
                    if self.rank == 0:
                        print(f"📊 Learning rates for {self.model_type} (with LoRA):")
                        if not freeze_enc_and_proj:
                            print(f"   Encoder LR:          {encoder_lr:.2e}")
                            print(f"   Projector LR:        {projector_lr:.2e}")
                        print(f"   Decoder LR:          {decoder_lr:.2e}")
                        print(f"   Decoder projector LR:{decoder_proj_lr:.2e}")
                        print(f"   LoRA LR:             {lora_lr:.2e} ({len(lora_params)} parameters)")
                else:
                    raise RuntimeError(
                        "LoRA is enabled but no trainable LoRA parameters found. "
                        "This indicates a LoRA configuration issue."
                    )
            elif _is_full_ft(stage_name):
                # Full LLM fine-tuning: collect every now-trainable LLM param
                # (unfrozen in _unfreeze_llm_if_needed) into a dedicated group
                # at lr_base — typically 5e-5 to keep the LLM close to its
                # pretrained weights while encoder/decoder train at LR_ENCODER.
                # lr_base is the *LoRA* learning rate for this stage (2e-4 for
                # joint_phase2_downstream, see :561). Inheriting it for full FT
                # would train a 1B backbone 20x faster than the comment two lines
                # up recommends, and 20x faster than the tokenized full-FT
                # baseline this arm is meant to be compared against (--lr 1e-5).
                # FULL_FT_LLM_LR pins it explicitly.
                _env_lr = os.environ.get("FULL_FT_LLM_LR")
                if _env_lr:
                    llm_lr = float(_env_lr)
                else:
                    llm_lr = lr_base if lr_base is not None else 5e-5
                llm_params = [p for p in model.llm.parameters() if p.requires_grad]
                if not llm_params:
                    raise RuntimeError(
                        f"{stage_name} is in FULL_LLM_FINETUNE_STAGES but no LLM "
                        "params are trainable. _unfreeze_llm_if_needed did not run."
                    )
                param_groups.append(
                    {"params": llm_params, "lr": llm_lr, "weight_decay": WEIGHT_DECAY}
                )
                if self.rank == 0:
                    n_llm = sum(p.numel() for p in llm_params)
                    print(f"📊 Learning rates for {self.model_type} (full LLM FT):")
                    if not freeze_enc_and_proj:
                        print(f"   Encoder LR:          {encoder_lr:.2e}")
                        print(f"   Projector LR:        {projector_lr:.2e}")
                    print(f"   Decoder LR:          {decoder_lr:.2e}")
                    print(f"   Decoder projector LR:{decoder_proj_lr:.2e}")
                    print(f"   LLM LR (full FT):    {llm_lr:.2e} ({n_llm:,} parameters)")
            else:
                if self.rank == 0:
                    print(f"📊 Learning rates for {self.model_type}:")
                    if not freeze_enc_and_proj:
                        print(f"   Encoder LR:          {encoder_lr:.2e}")
                        print(f"   Projector LR:        {projector_lr:.2e}")
                    print(f"   Decoder projector LR:{decoder_proj_lr:.2e}")

            # Print the total trainable parameters for transparency
            if self.rank == 0:
                total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"Total trainable parameters: {total_trainable:,}")
                # be specific about which parameters are trainable

            return AdamW(param_groups)
        

        elif self.model_type == "OpenTSLMSP":
            # Parameter groups with different learning rates for SP
            enc_params = list(model.encoder.parameters())
            proj_params = list(model.projector.projector.parameters())

            # Use provided learning rates or defaults
            encoder_lr = lr_encoder if lr_encoder is not None else LR_ENCODER
            projector_lr = lr_projector if lr_projector is not None else LR_PROJECTOR

            param_groups = [
                {"params": enc_params, "lr": encoder_lr, "weight_decay": WEIGHT_DECAY},
                {
                    "params": proj_params,
                    "lr": projector_lr,
                    "weight_decay": WEIGHT_DECAY,
                },
            ]

            # Add LoRA parameters if enabled
            if hasattr(model, "lora_enabled") and model.lora_enabled:
                lora_params = model.get_lora_parameters()
                if lora_params:
                    # Use projector LR for LoRA parameters (similar fine-tuning nature)
                    param_groups.append(
                        {
                            "params": lora_params,
                            "lr": projector_lr,
                            "weight_decay": WEIGHT_DECAY,
                        }
                    )
                    if self.rank == 0:
                        print(f"📊 Learning rates for {self.model_type} (with LoRA):")
                        print(f"   Encoder LR: {encoder_lr:.2e}")
                        print(f"   Projector LR: {projector_lr:.2e}")
                        print(
                            f"   LoRA LR: {projector_lr:.2e} ({len(lora_params)} parameters)"
                        )
                else:
                    raise RuntimeError(
                        "LoRA is enabled but no trainable LoRA parameters found. This indicates a LoRA configuration issue."
                    )

            return AdamW(param_groups)

        else:
            # For Flamingo joint phase1: use component-specific learning rates
            if self.model_type == "OpenTSLMFlamingo" and stage_name == "joint_phase1_flamingo":
                enc_lr        = lr_encoder   if lr_encoder   is not None else LR_ENCODER_JOINT
                dec_lr        = lr_decoder   if lr_decoder   is not None else LR_DECODER
                dec_proj_lr   = le_decoder_projector if le_decoder_projector is not None else LR_DECODER_PROJECTOR
                perceiver_lr  = lr_projector if lr_projector is not None else 1e-4
                base_lr       = lr_base      if lr_base      is not None else 2e-4

                # Explicit per-component param groups
                known_ids: set = set()

                def _params(module):
                    ps = [p for p in module.parameters() if p.requires_grad]
                    known_ids.update(id(p) for p in ps)
                    return ps

                param_groups = [
                    {"params": _params(model.ts_encoder),
                     "lr": enc_lr,       "weight_decay": WEIGHT_DECAY},
                    {"params": _params(model.decoder),
                     "lr": dec_lr,       "weight_decay": WEIGHT_DECAY},
                    {"params": _params(model.decoder_projector),
                     "lr": dec_proj_lr,  "weight_decay": WEIGHT_DECAY},
                    {"params": _params(model.model.perceiver),
                     "lr": perceiver_lr, "weight_decay": 0.0},
                    {"params": _params(model.model.lang_encoder.gated_cross_attn_layers),
                     "lr": base_lr,      "weight_decay": 0.1},
                ]

                # Catch remaining trainable params (e.g. input embeddings) not yet covered
                remaining = [p for p in model.parameters()
                             if p.requires_grad and id(p) not in known_ids]
                if remaining:
                    param_groups.append(
                        {"params": remaining, "lr": base_lr, "weight_decay": 0.0}
                    )

                if self.rank == 0:
                    total_trainable = sum(
                        p.numel() for g in param_groups for p in g["params"]
                    )
                    print(f"📊 Learning rates for {self.model_type} ({stage_name}):")
                    print(f"   Encoder LR:           {enc_lr:.2e}")
                    print(f"   Decoder LR:           {dec_lr:.2e}")
                    print(f"   Decoder projector LR: {dec_proj_lr:.2e}")
                    print(f"   Perceiver LR:         {perceiver_lr:.2e}")
                    print(f"   Gated cross-attn LR:  {base_lr:.2e}")
                    print(f"   Total trainable:      {total_trainable:,}")

                return AdamW(param_groups)

            # For other Flamingo stages: all trainable params at a single LR
            params_to_optimize = model.named_parameters()
            params_to_optimize = list(
                filter(
                    lambda x: x[1].requires_grad
                    and not getattr(x[1], "exclude_from_optimizer", False),
                    params_to_optimize,
                )
            )

            # Group parameters for weight decay
            params_with_wd, params_without_wd = [], []
            for n, p in params_to_optimize:
                if "gated_cross_attn" in n:
                    params_with_wd.append(p)
                else:
                    params_without_wd.append(p)

            # Use provided base learning rate or default
            base_lr = lr_base if lr_base is not None else 2e-4

            if self.rank == 0:
                print(f"📊 Learning rate for {self.model_type}:")
                print(f"   Base LR: {base_lr:.2e}")

            return torch.optim.AdamW(
                [
                    {"params": params_with_wd, "weight_decay": 0.1},
                    {"params": params_without_wd, "weight_decay": 0.0},
                ],
                lr=base_lr,
            )

    def _make_collate(self, patch_size: int):
        """Return the active DataLoader collate.

        Uses :class:`OpenTSLMProcessor` (worker-side tokenizer + TS prep) when
        ``--use_processor`` is set; otherwise the legacy patch-pad collate.
        """
        if self.processor is not None:
            return self.processor
        return partial(
            extend_time_series_to_match_patch_size_and_aggregate,
            patch_size=patch_size,
        )

    def _merge_data_loaders(
        self,
        datasets: List[Dataset],
        shuffle: bool,
        batch_size: int,
        patch_size: int,
        distribute_data: bool = False,
        drop_last: bool = False,
    ) -> DataLoader:
        """Create a merged data loader from multiple datasets."""
        merged_ds = ConcatDataset(datasets)

        _nw = getattr(self.args, "dataloader_workers", 4)
        _pin = torch.cuda.is_available() and _nw > 0
        _worker_kwargs = dict(
            num_workers=_nw,
            pin_memory=_pin,
            persistent_workers=(_nw > 0),
            prefetch_factor=(2 if _nw > 0 else None),
        )
        _collate = self._make_collate(patch_size)

        # Use distributed sampler if distributed training is enabled
        if distribute_data and dist.is_initialized():
            sampler = DistributedSampler(
                merged_ds, num_replicas=self.world_size, rank=self.rank, shuffle=shuffle
            )
            return DataLoader(
                merged_ds,
                sampler=sampler,
                batch_size=batch_size,
                drop_last=drop_last,
                collate_fn=_collate,
                **_worker_kwargs,
            )
        else:
            return DataLoader(
                merged_ds,
                shuffle=shuffle,
                batch_size=batch_size,
                collate_fn=_collate,
                **_worker_kwargs,
            )

    def _assert_distributed_loader_lengths_match(
        self, loader: DataLoader, stage_name: str
    ) -> None:
        """Fail fast if DDP ranks see different train loader lengths."""
        if not dist.is_initialized():
            return

        local_batches = len(loader)
        local_dataset_size = len(loader.dataset) if hasattr(loader, "dataset") else -1

        batches_t = torch.tensor([local_batches], device=self.device, dtype=torch.long)
        dataset_t = torch.tensor([local_dataset_size], device=self.device, dtype=torch.long)
        gathered_batches = [torch.zeros_like(batches_t) for _ in range(self.world_size)]
        gathered_dataset = [torch.zeros_like(dataset_t) for _ in range(self.world_size)]
        dist.all_gather(gathered_batches, batches_t)
        dist.all_gather(gathered_dataset, dataset_t)

        batch_lens = [int(t.item()) for t in gathered_batches]
        dataset_lens = [int(t.item()) for t in gathered_dataset]
        if len(set(batch_lens)) != 1:
            msg = (
                f"DDP train_loader length mismatch for stage '{stage_name}': "
                f"batches_per_rank={batch_lens}, dataset_sizes={dataset_lens}. "
                "This causes distributed all_reduce deadlocks near epoch end. "
                "Ensure dataset construction is deterministic across ranks."
            )
            if self.rank == 0:
                print(f"❌ {msg}")
            raise RuntimeError(msg)

    def _restore_llm_embeddings_from_checkpoint(self, model, checkpoint):
        """Copy resized LLM token embeddings back from a checkpoint.

        resize_token_embeddings randomly initialises new rows (for <ts>, </ts>)
        every time the model is constructed. The LLM is otherwise frozen, so
        without restoring those rows the encoder/decoder see different inputs
        than they saw during training.
        """
        if not hasattr(model, "llm"):
            return
        in_emb = model.llm.get_input_embeddings()
        if "llm_input_embeddings" in checkpoint:
            saved_in = checkpoint["llm_input_embeddings"].to(
                device=in_emb.weight.device, dtype=in_emb.weight.dtype
            )
            if saved_in.shape == in_emb.weight.shape:
                with torch.no_grad():
                    in_emb.weight.data.copy_(saved_in)
            elif self.rank == 0:
                print(
                    f"⚠️  llm_input_embeddings shape mismatch "
                    f"({tuple(saved_in.shape)} vs {tuple(in_emb.weight.shape)}); skipping."
                )
            out_emb = model.llm.get_output_embeddings()
            if "llm_output_embeddings" in checkpoint and out_emb is not None:
                saved_out = checkpoint["llm_output_embeddings"].to(
                    device=out_emb.weight.device, dtype=out_emb.weight.dtype
                )
                if saved_out.shape == out_emb.weight.shape:
                    with torch.no_grad():
                        out_emb.weight.data.copy_(saved_out)
        elif self.rank == 0:
            print(
                "⚠️  Checkpoint has no llm_input_embeddings — <ts>/</ts> rows "
                "will differ from the ones used at training time."
            )

    def _save_checkpoint(
        self, stage: str, epoch: int, val_loss: float, optimizer, scheduler, extra_state=None
    ):
        """Save model checkpoint for a specific stage."""
        checkpoint_dir = os.path.join(self._stage_dir(stage), "checkpoints")

        # Only save on rank 0 for distributed training
        if dist.is_initialized() and self.rank != 0:
            return

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Periodic checkpoints are only used as an in-flight safety net.
        # Remove them before saving a new best model so we do not keep two
        # multi-GB checkpoints in the same stage and trip filesystem quotas.
        periodic_checkpoint_path = os.path.join(
            checkpoint_dir, "periodic_checkpoint.pt"
        )
        if os.path.exists(periodic_checkpoint_path):
            try:
                os.remove(periodic_checkpoint_path)
                if self.rank == 0:
                    print(
                        "🧹 Removed periodic checkpoint before saving the new best "
                        f"model to free disk quota: {periodic_checkpoint_path}"
                    )
            except OSError as e:
                if self.rank == 0:
                    print(
                        "⚠️  Could not remove periodic checkpoint before saving the "
                        f"best model: {e}"
                    )

        # Get the underlying model (handles DDP wrapping)
        model = self._get_model()

        if self.model_type == "OpenTSLMSP":
            checkpoint = {
                "encoder_state": model.encoder.state_dict(),
                "projector_state": model.projector.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_loss": val_loss,
                "epoch": epoch,
            }

            # Add LoRA state to checkpoint
            model.save_lora_state_to_checkpoint(checkpoint)
            _save_llm_state_if_full_ft(checkpoint, model, stage)
        elif self.model_type in ["OpenTSLMSPFixPos", "OpenTSLMSPFixPos2D", "OpenTSLMSPFixPos2DTimerXL"]:
            checkpoint = {
                "encoder_state": model.encoder.state_dict(),
                "projector_state": model.projector.state_dict(),
                "decoder_state": model.decoder.state_dict(),
                "decoder_projector_state": model.decoder_projector.state_dict(),
                "mse_projector_state": model.mse_projector.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_loss": val_loss,
                "epoch": epoch,
            }

            # Persist the resized LLM token embeddings so <ts>/</ts> rows match
            # training at eval time. resize_token_embeddings randomises these on
            # every construction and the LLM is otherwise frozen, so without
            # saving them the frozen encoder/decoder would see different inputs
            # whenever the model is rebuilt.
            _in_emb = model.llm.get_input_embeddings()
            checkpoint["llm_input_embeddings"] = _in_emb.weight.data.detach().cpu().clone()
            _out_emb = model.llm.get_output_embeddings()
            if _out_emb is not None and _out_emb.weight.data_ptr() != _in_emb.weight.data_ptr():
                checkpoint["llm_output_embeddings"] = _out_emb.weight.data.detach().cpu().clone()

            # Add LoRA state to checkpoint
            model.save_lora_state_to_checkpoint(checkpoint)
            _save_llm_state_if_full_ft(checkpoint, model, stage)
        elif self.model_type == "OpenTSLMSPMomentSoftPrompt":  # ADD THIS
            checkpoint = {
                "encoder_projector_state": model.encoder_projector.state_dict(),
                "decoder_projector_state": model.decoder_projector.state_dict(),
                "decoder_head_state": model.decoder_head.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_loss": val_loss,
                "epoch": epoch,
            }

            # OPTIONAL: if LoRA enabled, save only LoRA weights (lightweight)
            if getattr(model, "lora_enabled", False):
                lora_sd = {k: v for k, v in model.llm.state_dict().items() if "lora_" in k}
                checkpoint["lora_state"] = lora_sd

        else:
            # Handle DDP or single GPU case for OpenTSLMFlamingo
            model_state = model.state_dict()
            if hasattr(self.model, "module"):
                # Remove 'module.' prefix for DDP
                model_state = {
                    k.replace("module.", ""): v for k, v in model_state.items()
                }
            checkpoint = {
                "model_state": model_state,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_loss": val_loss,
                "epoch": epoch,
            }

        # Record the encoder geometry that a warm start must match. These are not
        # weights, but silently changing them across phases re-interprets the
        # weights that ARE saved (see _warn_on_encoder_geometry_drift).
        checkpoint["encoder_geometry"] = {
            "pe_type": getattr(self.args, "pe_type", None),
            "pe_base_t": getattr(self.args, "pe_base_t", None),
            "pe_base_c": getattr(self.args, "pe_base_c", None),
            "patch_size": getattr(self.args, "patch_size", None),
            "use_freq_embed": bool(getattr(self.args, "use_freq_embed", False)),
            "freq_mode": getattr(self.args, "freq_mode", "linear"),
        }

        if extra_state:
            checkpoint.update(extra_state)

        checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")

        # Check disk space before saving
        if self.rank == 0:
            import shutil

            total, used, free = shutil.disk_usage(checkpoint_dir)
            free_gb = free / (1024**3)
            print(f"💾 Disk space: {free_gb:.2f} GB free in {checkpoint_dir}")

            # Estimate checkpoint size (rough estimate)
            estimated_size_gb = sum(
                p.numel() * p.element_size() for p in self._get_model().parameters()
            ) / (1024**3)
            if (
                free_gb < estimated_size_gb * 2
            ):  # Need at least 2x the size for safe writing
                print(
                    f"⚠️  Warning: Low disk space. Need ~{estimated_size_gb:.2f} GB, have {free_gb:.2f} GB free"
                )

        # Try to save with error handling
        try:
            torch.save(checkpoint, checkpoint_path)
        except Exception as e:
            quota_probe_error = None
            quota_probe_path = os.path.join(checkpoint_dir, ".checkpoint_write_probe")
            try:
                with open(quota_probe_path, "wb") as probe_file:
                    probe_file.write(b"0")
                os.remove(quota_probe_path)
            except OSError as probe_error:
                quota_probe_error = probe_error

            if self.rank == 0:
                print(f"❌ Failed to save checkpoint: {e}")
                print(f"   Checkpoint path: {checkpoint_path}")
                print(
                    f"   Checkpoint size: {sum(p.numel() * p.element_size() for p in self._get_model().parameters()) / 1024**3:.2f} GB"
                )
                if getattr(quota_probe_error, "errno", None) == 122:
                    print(
                        "   Small-file write probe also failed with disk quota "
                        "exceeded. Remove older checkpoints or use a larger "
                        "filesystem for results."
                    )

                raise RuntimeError(f"Failed to save checkpoint: {e}")

    def _save_periodic_checkpoint(self, stage: str, epoch: int, step: int, optimizer, scheduler):
        """Save a mid-epoch checkpoint to periodic_checkpoint.pt.

        Deletes the previous file before writing so only one copy ever exists.
        Only rank 0 writes; all other ranks skip.
        """
        if dist.is_initialized() and self.rank != 0:
            return

        checkpoint_dir = os.path.join(self._stage_dir(stage), "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, "periodic_checkpoint.pt")

        # Delete old checkpoint first so we never have two copies on disk
        if os.path.exists(path):
            os.remove(path)

        model = self._get_model()

        if self.model_type == "OpenTSLMSP":
            checkpoint = {
                "encoder_state": model.encoder.state_dict(),
                "projector_state": model.projector.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "epoch": epoch,
                "step": step,
                "val_loss": float("inf"),  # not yet validated
            }
            model.save_lora_state_to_checkpoint(checkpoint)
        elif self.model_type in ["OpenTSLMSPFixPos", "OpenTSLMSPFixPos2D", "OpenTSLMSPFixPos2DTimerXL"]:
            checkpoint = {
                "encoder_state": model.encoder.state_dict(),
                "projector_state": model.projector.state_dict(),
                "decoder_state": model.decoder.state_dict(),
                "decoder_projector_state": model.decoder_projector.state_dict(),
                "mse_projector_state": model.mse_projector.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "epoch": epoch,
                "step": step,
                "val_loss": float("inf"),  # not yet validated
            }

            # Persist resized LLM token embeddings (see _save_checkpoint).
            _in_emb = model.llm.get_input_embeddings()
            checkpoint["llm_input_embeddings"] = _in_emb.weight.data.detach().cpu().clone()
            _out_emb = model.llm.get_output_embeddings()
            if _out_emb is not None and _out_emb.weight.data_ptr() != _in_emb.weight.data_ptr():
                checkpoint["llm_output_embeddings"] = _out_emb.weight.data.detach().cpu().clone()

            model.save_lora_state_to_checkpoint(checkpoint)
        else:
            checkpoint = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "epoch": epoch,
                "step": step,
                "val_loss": float("inf"),
            }

        try:
            torch.save(checkpoint, path)
            print(f"[rank0] 💾 Periodic checkpoint saved — epoch {epoch}, step {step}")
        except Exception as e:
            print(f"[rank0] ⚠️  Failed to save periodic checkpoint: {e}")

    def _save_loss_history(
        self, stage: str, epoch: int, train_loss: float, val_loss: float
    ):
        """Save loss history to a file for tracking training progress."""
        if dist.is_initialized() and self.rank != 0:
            return  # Only save on rank 0 for distributed training

        checkpoint_dir = os.path.join(self._stage_dir(stage), "checkpoints")
        loss_history_file = os.path.join(checkpoint_dir, "loss_history.txt")

        # Ensure the directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Create the file with header if it doesn't exist
        if not os.path.exists(loss_history_file):
            with open(loss_history_file, "w") as f:
                f.write("Epoch\tTrain_Loss\tVal_Loss\n")
                f.write("-" * 30 + "\n")

        # Append the current epoch's losses
        with open(loss_history_file, "a") as f:
            f.write(f"{epoch}\t{train_loss:.6f}\t{val_loss:.6f}\n")

    def _display_loss_history(self, stage: str):
        """Display the loss history for a stage if available."""
        if dist.is_initialized() and self.rank != 0:
            return  # Only display on rank 0 for distributed training

        checkpoint_dir = os.path.join(self._stage_dir(stage), "checkpoints")
        loss_history_file = os.path.join(checkpoint_dir, "loss_history.txt")

        if os.path.exists(loss_history_file):
            try:
                with open(loss_history_file, "r") as f:
                    lines = f.readlines()

                if len(lines) > 2:  # More than just header
                    print(f"📊 Previous loss history for {stage}:")
                    print("   Epoch\tTrain_Loss\tVal_Loss")
                    print("   " + "-" * 30)

                    # Show last 5 epochs (or all if less than 5)
                    start_idx = max(2, len(lines) - 5)  # Skip header lines
                    for line in lines[start_idx:]:
                        if line.strip() and not line.startswith("-"):
                            parts = line.strip().split("\t")
                            if len(parts) == 3:
                                epoch, train_loss, val_loss = parts
                                print(f"   {epoch}\t{train_loss}\t{val_loss}")

                    if len(lines) > 7:  # More than 5 epochs
                        print(f"   ... and {len(lines) - 7} more epochs")
                    print()
            except Exception as e:
                print(f"⚠️  Could not read loss history: {e}")

    def _load_state_dict_flexible(self, module, state_dict, name="module"):
        """Load state dict, skipping keys whose tensor shape doesn't match the
        current model (e.g. patch_embed when patch_size changes between runs).
        Warns on rank 0 about skipped and genuinely missing keys.
        """
        model_sd = module.state_dict()
        compatible, skipped_keys = {}, set()
        for k, v in state_dict.items():
            if k in model_sd and model_sd[k].shape != v.shape:
                skipped_keys.add(k)
                if self.rank == 0:
                    print(
                        f"⚠️  [{name}] skipping '{k}': "
                        f"ckpt={tuple(v.shape)} vs model={tuple(model_sd[k].shape)}"
                    )
            else:
                compatible[k] = v
        missing, _ = module.load_state_dict(compatible, strict=False)
        truly_missing = [k for k in missing if k not in skipped_keys]
        if truly_missing and self.rank == 0:
            print(f"⚠️  [{name}] {len(truly_missing)} keys missing from checkpoint: {truly_missing[:10]}")

    def _load_checkpoint(
        self, stage: str, optimizer, scheduler, eval_only: bool = False, scaler=None
    ):
        """Load model checkpoint for a specific stage."""
        checkpoint_path = os.path.join(self._stage_dir(stage), "checkpoints", "best_model.pt"
        )
        self._last_loaded_checkpoint_meta = {}

        if os.path.exists(checkpoint_path):
            # Always load checkpoint to CPU first to avoid GPU OOM spikes
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            self._last_loaded_checkpoint_meta = {
                "joint_init_mse": checkpoint.get("joint_init_mse"),
                "joint_init_ce": checkpoint.get("joint_init_ce"),
            }

            # Get the underlying model (handles DDP wrapping)
            model = self._get_model()

            if self.model_type == "OpenTSLMSP":
                self._load_state_dict_flexible(model.encoder, checkpoint["encoder_state"], "encoder")
                self._load_state_dict_flexible(model.projector, checkpoint["projector_state"], "projector")

                # Load LoRA state using the OpenTSLMSP method (allow missing for backward compatibility)
                try:
                    model.load_lora_state_from_checkpoint(
                        checkpoint, allow_missing=True
                    )
                except RuntimeError as e:
                    if self.rank == 0:
                        print(f"❌ Failed to load LoRA state from checkpoint: {e}")
                    raise
            elif self.model_type in ["OpenTSLMSPFixPos", "OpenTSLMSPFixPos2D", "OpenTSLMSPFixPos2DTimerXL"]:
                self._load_state_dict_flexible(model.encoder, checkpoint["encoder_state"], "encoder")
                self._load_state_dict_flexible(model.projector, checkpoint["projector_state"], "projector")
                self._load_state_dict_flexible(model.decoder, checkpoint["decoder_state"], "decoder")
                self._load_state_dict_flexible(model.decoder_projector, checkpoint["decoder_projector_state"], "decoder_projector")
                if "mse_projector_state" in checkpoint:
                    self._load_state_dict_flexible(model.mse_projector, checkpoint["mse_projector_state"], "mse_projector")
                if checkpoint.get("llm_full_ft") and "llm_state" in checkpoint:
                    # Full-FT arm: the backbone itself was trained, so restoring
                    # encoder/projector alone would silently revert it to the
                    # pretrained weights on every resume.
                    _miss = model.llm.load_state_dict(checkpoint["llm_state"], strict=False)
                    if self.rank == 0:
                        print(f"🔓 restored fully fine-tuned LLM from checkpoint "
                              f"(missing={len(_miss.missing_keys)}, "
                              f"unexpected={len(_miss.unexpected_keys)})")

                self._restore_llm_embeddings_from_checkpoint(model, checkpoint)

                # Load LoRA state using the OpenTSLMSP method (allow missing for backward compatibility)
                try:
                    model.load_lora_state_from_checkpoint(
                        checkpoint, allow_missing=True
                    )
                except RuntimeError as e:
                    if self.rank == 0:
                        print(f"❌ Failed to load LoRA state from checkpoint: {e}")
                    raise

                # Only load optimizer state when training
                if (
                    not eval_only
                    and optimizer is not None
                    and "optimizer_state" in checkpoint
                ):
                    optimizer.load_state_dict(checkpoint["optimizer_state"])
            elif self.model_type == "OpenTSLMSPMomentSoftPrompt":  # ADD THIS
                model.encoder_projector.load_state_dict(checkpoint["encoder_projector_state"])
                model.decoder_projector.load_state_dict(checkpoint["decoder_projector_state"])
                model.decoder_head.load_state_dict(checkpoint["decoder_head_state"])

                if getattr(model, "lora_enabled", False) and ("lora_state" in checkpoint):
                    # load whatever keys exist
                    sd = model.llm.state_dict()
                    sd.update(checkpoint["lora_state"])
                    model.llm.load_state_dict(sd, strict=False)

                if not eval_only and optimizer is not None and "optimizer_state" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer_state"])
            else:
                # Handle DDP or single GPU case for OpenTSLMFlamingo
                model_state = checkpoint["model_state"]
                if hasattr(self.model, "module"):
                    # Add 'module.' prefix for DDP
                    model_state = {f"module.{k}": v for k, v in model_state.items()}

                # Load state dict with strict=False to handle missing keys
                try:
                    missing_keys, unexpected_keys = self.model.load_state_dict(
                        model_state, strict=False
                    )
                    if missing_keys and self.rank == 0:
                        print(
                            f"⚠️  Warning: Missing keys when loading checkpoint for {stage}:"
                        )
                        for key in missing_keys[:10]:  # Show first 10 missing keys
                            print(f"   - {key}")
                        if len(missing_keys) > 10:
                            print(f"   ... and {len(missing_keys) - 10} more keys")
                    if unexpected_keys and self.rank == 0:
                        print(
                            f"⚠️  Warning: Unexpected keys when loading checkpoint for {stage}:"
                        )
                        for key in unexpected_keys[
                            :10
                        ]:  # Show first 10 unexpected keys
                            print(f"   - {key}")
                        if len(unexpected_keys) > 10:
                            print(f"   ... and {len(unexpected_keys) - 10} more keys")
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load model state from checkpoint for {stage}: {e}"
                    )

                # Only load optimizer state when training
                if (
                    not eval_only
                    and optimizer is not None
                    and "optimizer_state" in checkpoint
                ):
                    optimizer.load_state_dict(checkpoint["optimizer_state"])

            # Only load scheduler state when training
            if (
                not eval_only
                and scheduler is not None
                and "scheduler_state" in checkpoint
            ):
                scheduler.load_state_dict(checkpoint["scheduler_state"])

            if scaler is not None and "scaler_state" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state"])

            return checkpoint.get("epoch", "?"), checkpoint.get(
                "val_loss", float("inf")
            )
        return None, float("inf")

    def _load_previous_stage_model(
        self, current_stage: str
    ) -> Optional[Dict[str, Any]]:
        """Load the best model from the previous stage and return its metrics."""
        try:
            if current_stage not in CURRICULUM_STAGES:
                # Sub-stages (e.g. phase1_chronos_mse) are not in CURRICULUM_STAGES;
                # treat them as first-in-sequence — no previous stage to load.
                return None
            current_idx = CURRICULUM_STAGES.index(current_stage)
            if current_idx == 0:
                # First stage, no previous model to load
                return None
            previous_stage = CURRICULUM_STAGES[current_idx - 1]
            metrics_file = os.path.join(self._stage_dir(previous_stage), "results", "metrics.json"
            )
            if not os.path.exists(metrics_file):
                # PATCH: If running stage2_captioning and previous stage metrics are missing, skip loading
                if current_stage == "stage2_captioning":
                    if self.rank == 0:
                        print(
                            f"⚠️  Skipping previous stage {previous_stage} because metrics file not found: {metrics_file}"
                        )
                    return None
                raise RuntimeError(
                    f"Previous stage {previous_stage} metrics file not found: {metrics_file}"
                )
            # Be robust to malformed JSON (e.g., concurrent writes or concatenated JSON)
            try:
                with open(metrics_file, "r") as f:
                    metrics = json.load(f)
            except Exception as e:
                if self.rank == 0:
                    print(
                        f"⚠️  Warning: Could not parse metrics file for {previous_stage} ({metrics_file}): {e}"
                    )
                    print("   Proceeding without previous metrics.")
                metrics = {}
            # Load the model weights from previous stage
            checkpoint_path = os.path.join(self._stage_dir(previous_stage), "checkpoints", "best_model.pt"
            )
            if not os.path.exists(checkpoint_path):
                # PATCH: If running stage2_captioning and previous stage checkpoint is missing, skip loading
                if current_stage == "stage2_captioning":
                    if self.rank == 0:
                        print(
                            f"⚠️  Skipping previous stage {previous_stage} because checkpoint not found: {checkpoint_path}"
                        )
                    return None
                raise RuntimeError(
                    f"Previous stage {previous_stage} checkpoint not found: {checkpoint_path}"
                )
            print(
                "Loading checkpoint from previous stage: ",
                checkpoint_path,
                "and model type: ",
                self.model_type,
                "and llm_id: ",
                self.llm_id,
            )
            print("This might take a while")
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            # Get the underlying model (handles DDP wrapping)
            model = self._get_model()
            if self.model_type == "OpenTSLMSP":
                self._load_state_dict_flexible(model.encoder, checkpoint["encoder_state"], "encoder")
                self._load_state_dict_flexible(model.projector, checkpoint["projector_state"], "projector")

                # Load LoRA state from previous stage (allow missing for stage transitions)
                try:
                    loaded_count = model.load_lora_state_from_checkpoint(
                        checkpoint, allow_missing=True
                    )
                    if loaded_count > 0 and self.rank == 0:
                        print(
                            f"📥 Loaded LoRA adapters from previous stage: {loaded_count} parameters"
                        )
                except RuntimeError as e:
                    if self.rank == 0:
                        print(f"❌ Failed to load LoRA state from previous stage: {e}")
                    # For previous stage loading, we can be more tolerant of LoRA mismatches
                    # as stages might have different LoRA configurations
            elif self.model_type in ["OpenTSLMSPFixPos", "OpenTSLMSPFixPos2D", "OpenTSLMSPFixPos2DTimerXL"]:
                self._load_state_dict_flexible(model.encoder, checkpoint["encoder_state"], "encoder")
                self._load_state_dict_flexible(model.projector, checkpoint["projector_state"], "projector")
                self._load_state_dict_flexible(model.decoder, checkpoint["decoder_state"], "decoder")
                self._load_state_dict_flexible(model.decoder_projector, checkpoint["decoder_projector_state"], "decoder_projector")
                if "mse_projector_state" in checkpoint:
                    self._load_state_dict_flexible(model.mse_projector, checkpoint["mse_projector_state"], "mse_projector")
                if checkpoint.get("llm_full_ft") and "llm_state" in checkpoint:
                    # Full-FT arm: the backbone itself was trained, so restoring
                    # encoder/projector alone would silently revert it to the
                    # pretrained weights on every resume.
                    _miss = model.llm.load_state_dict(checkpoint["llm_state"], strict=False)
                    if self.rank == 0:
                        print(f"🔓 restored fully fine-tuned LLM from checkpoint "
                              f"(missing={len(_miss.missing_keys)}, "
                              f"unexpected={len(_miss.unexpected_keys)})")

                self._restore_llm_embeddings_from_checkpoint(model, checkpoint)

                # Load LoRA state from previous stage (allow missing for stage transitions)
                try:
                    loaded_count = model.load_lora_state_from_checkpoint(
                        checkpoint, allow_missing=True
                    )
                    if loaded_count > 0 and self.rank == 0:
                        print(
                            f"📥 Loaded LoRA adapters from previous stage: {loaded_count} parameters"
                        )
                except RuntimeError as e:
                    if self.rank == 0:
                        print(f"❌ Failed to load LoRA state from previous stage: {e}")
                    # For previous stage loading, we can be more tolerant of LoRA mismatches
                    # as stages might have different LoRA configurations
            else:
                # Handle OpenTSLMFlamingo with graceful loading
                model_state = checkpoint["model_state"]
                if hasattr(self.model, "module"):
                    # Add 'module.' prefix for DDP
                    model_state = {f"module.{k}": v for k, v in model_state.items()}
                # Load state dict with strict=False to handle missing keys
                try:
                    missing_keys, unexpected_keys = self.model.load_state_dict(
                        model_state, strict=False
                    )
                    if missing_keys and self.rank == 0:
                        print(
                            f"⚠️  Warning: Missing keys when loading previous stage {previous_stage}:"
                        )
                        for key in missing_keys[:5]:  # Show first 5 missing keys
                            print(f"   - {key}")
                        if len(missing_keys) > 5:
                            print(f"   ... and {len(missing_keys) - 5} more keys")
                        print(
                            f"   This is normal when transitioning between stages with different model configurations."
                        )
                    if unexpected_keys and self.rank == 0:
                        print(
                            f"⚠️  Warning: Unexpected keys when loading previous stage {previous_stage}:"
                        )
                        for key in unexpected_keys[:5]:  # Show first 5 unexpected keys
                            print(f"   - {key}")
                        if len(unexpected_keys) > 5:
                            print(f"   ... and {len(unexpected_keys) - 5} more keys")
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load model state from previous stage {previous_stage}: {e}"
                    )
            return {
                "stage": previous_stage,
                "metrics": metrics,
                "epoch": checkpoint.get("epoch", "?"),
                "val_loss": checkpoint.get("val_loss", "?"),
            }
        except Exception as e:
            raise RuntimeError(f"Failed to load previous stage model: {e}")

    def _warn_on_encoder_geometry_drift(self, checkpoint, checkpoint_path: str) -> None:
        """Loudly flag a warm start whose encoder geometry differs from the checkpoint's.

        Parameters like `pe_base_c` and `freq_mode` do not change any tensor shape,
        so a mismatched value loads silently and simply re-interprets the weights
        being restored. This has bitten us: phase 1 trained with `--pe_base_c 10`
        while phase 2 omitted the flag and defaulted to 1e4, giving the warm-started
        encoder a ~40% less separated channel signature than it was pretrained with.
        """
        if self.rank != 0:
            return
        saved = checkpoint.get("encoder_geometry")
        if not saved:
            print(
                "⚠️  Warm-start checkpoint predates encoder-geometry recording "
                f"({os.path.basename(checkpoint_path)}); cannot verify that "
                "--pe_base_t/--pe_base_c/--patch_size/--freq_mode match the values it "
                "was trained with. Check them by hand."
            )
            return
        drift = {
            k: (v, getattr(self.args, k, None))
            for k, v in saved.items()
            if v is not None and getattr(self.args, k, None) != v
        }
        if drift:
            print("🚨 ENCODER GEOMETRY DRIFT vs warm-start checkpoint:")
            for k, (was, now) in drift.items():
                print(f"     {k}: checkpoint={was!r}  current={now!r}")
            print(
                "     These do not change tensor shapes, so the weights loaded "
                "without error but are being interpreted differently than they were "
                "trained. Pass the matching flags unless this is deliberate."
            )

    def _load_weights_from_stage(self, stage_name: str) -> bool:
        """Load model weights from any stage checkpoint, without requiring metrics.

        Returns True if the checkpoint was found and loaded, False otherwise.
        """
        if stage_name == "pretrained":
            # Special source: load directly from an explicit pretrained checkpoint
            # path (PRETRAINED_WARM_START_CKPT / $WARM_START_PRETRAINED_CKPT) — no
            # results/<stage> convention dir or symlink involved.
            checkpoint_path = PRETRAINED_WARM_START_CKPT
        else:
            checkpoint_path = os.path.join(
                self._stage_dir(stage_name), "checkpoints", "best_model.pt"
            )
        if not os.path.exists(checkpoint_path):
            if self.rank == 0:
                print(f"⚠️  No checkpoint found for stage '{stage_name}' at {checkpoint_path}")
            return False

        if self.rank == 0:
            print(f"📥 Loading weights from stage '{stage_name}': {checkpoint_path}")
        # (geometry check happens after the load, once we can read the dict)

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self._warn_on_encoder_geometry_drift(checkpoint, checkpoint_path)
        model = self._get_model()

        if self.model_type in ["OpenTSLMSP", "OpenTSLMSPFixPos", "OpenTSLMSPFixPos2D", "OpenTSLMSPFixPos2DTimerXL"]:
            _enc_sd = checkpoint["encoder_state"]
            if os.environ.get("ABLATION_DROP_FREQ_KEYS") == "1":
                # Frequency-branch ablation: the warm-start checkpoint was
                # trained WITH the rfft branch, so its encoder_state carries
                # freq_embed/concat_proj tensors that an encoder built without
                # --use_freq_embed has no slots for. Drop exactly those keys and
                # keep the load STRICT for everything else -- silently loading
                # non-strict would also mask genuine geometry mistakes.
                _drop = {k for k in _enc_sd if k.startswith(("freq_embed", "concat_proj"))}
                _enc_sd = {k: v for k, v in _enc_sd.items() if k not in _drop}
                if self.rank == 0:
                    print(f"🔬 ABLATION: dropped {len(_drop)} freq-branch tensors from the "
                          f"warm start ({sorted(_drop)}); remaining keys load strict")
            model.encoder.load_state_dict(_enc_sd)
            model.projector.load_state_dict(checkpoint["projector_state"])
            # Vanilla OpenTSLMSP has no decoder/decoder_projector, and its
            # checkpoints (incl. the official OpenTSLM HF releases) lack those
            # keys — guard so warm-start works across all SP variants.
            if hasattr(model, "decoder") and "decoder_state" in checkpoint:
                model.decoder.load_state_dict(checkpoint["decoder_state"])
            if hasattr(model, "decoder_projector") and "decoder_projector_state" in checkpoint:
                model.decoder_projector.load_state_dict(checkpoint["decoder_projector_state"])
            self._restore_llm_embeddings_from_checkpoint(model, checkpoint)
            # If the warm-start checkpoint carried a LoRA adapter, enable LoRA at the
            # SAME rank/alpha (read from the saved lora_config) BEFORE loading it, so
            # the phase-1 adapter is actually inherited. Otherwise
            # load_lora_state_from_checkpoint() raises ("LoRA not enabled"), and
            # _enable_lora_if_needed() would later init a fresh adapter at the stage's
            # default rank — discarding the phase-1 LoRA.
            if checkpoint.get("lora_enabled", False) and not getattr(model, "lora_enabled", False):
                _lcfg = checkpoint.get("lora_config")
                _pcfg = next(iter(_lcfg.values())) if isinstance(_lcfg, dict) and _lcfg else _lcfg
                _r = int(getattr(_pcfg, "r", 16)) if _pcfg is not None else 16
                _alpha = int(getattr(_pcfg, "lora_alpha", 32)) if _pcfg is not None else 32
                _tmods = getattr(_pcfg, "target_modules", None) if _pcfg is not None else None
                _tmods = sorted(_tmods) if _tmods is not None else None
                if self.rank == 0:
                    print(f"🔧 Warm-start: enabling LoRA from checkpoint config "
                          f"(r={_r}, alpha={_alpha}) so the phase-1 adapter transfers")
                model.enable_lora(lora_r=_r, lora_alpha=_alpha, lora_dropout=0.0,
                                  target_modules=_tmods)
                # LoRA params are added after the initial DDP wrap, so re-wrap (mirrors
                # _enable_lora_if_needed) to register AllReduce hooks for lora_A/lora_B.
                if self.world_size > 1:
                    self.model = DDP(
                        model,
                        device_ids=[self.local_rank] if torch.cuda.is_available() else None,
                        find_unused_parameters=True,
                    )
            model.load_lora_state_from_checkpoint(checkpoint, allow_missing=True)
            if _is_full_ft("") and getattr(model, "lora_enabled", False):
                # Merge rather than drop: the phase-1 adapter is part of the
                # pretrained model. Discarding it would start the full-FT arm
                # from a different (worse) initialisation than the LoRA arm and
                # confound the capacity ablation with a warm-start difference.
                try:
                    model.llm = model.llm.merge_and_unload()
                    model.lora_enabled = False
                    for _p in model.llm.parameters():
                        _p.requires_grad = False
                    if self.rank == 0:
                        print("🔗 Full-FT arm: merged the phase-1 LoRA adapter into the "
                              "base weights (merge_and_unload) and dropped the adapter; "
                              "the LLM is now unfrozen wholesale by _unfreeze_llm_if_needed.")
                except Exception as _e:
                    if self.rank == 0:
                        print(f"❌ could not merge phase-1 LoRA for full FT: {_e}")
                    raise
        else:
            if "model_state" in checkpoint:
                model_state = checkpoint["model_state"]
                if hasattr(self.model, "module"):
                    model_state = {f"module.{k}": v for k, v in model_state.items()}
                missing_keys, unexpected_keys = self.model.load_state_dict(
                    model_state, strict=False
                )
                if self.rank == 0 and missing_keys:
                    print(
                        f"⚠️  Warm-start missing keys for '{stage_name}': {missing_keys[:10]}"
                    )
                if self.rank == 0 and unexpected_keys:
                    print(
                        f"⚠️  Warm-start unexpected keys for '{stage_name}': {unexpected_keys[:10]}"
                    )
            else:
                # Support partial transfer from structured soft-prompt checkpoints
                # into the Flamingo forecasting variant when module names align.
                if hasattr(model, "encoder") and "encoder_state" in checkpoint:
                    self._load_state_dict_flexible(model.encoder, checkpoint["encoder_state"], "encoder")
                if hasattr(model, "decoder") and "decoder_state" in checkpoint:
                    self._load_state_dict_flexible(model.decoder, checkpoint["decoder_state"], "decoder")
                if hasattr(model, "decoder_projector") and "decoder_projector_state" in checkpoint:
                    self._load_state_dict_flexible(
                        model.decoder_projector,
                        checkpoint["decoder_projector_state"],
                        "decoder_projector",
                    )

        if self.rank == 0:
            print(f"✅ Weights loaded from '{stage_name}' (epoch {checkpoint.get('epoch', '?')}, val_loss {checkpoint.get('val_loss', '?')})")

        return True

    def _calculate_accuracy(
        self, predictions: List[str], gold_answers: List[str]
    ) -> float:
        """Calculate accuracy for MCQ tasks."""
        correct = 0
        total = len(predictions)

        for pred, gold in zip(predictions, gold_answers):
            # Clean up predictions and gold answers
            pred_clean = pred.strip()
            gold_clean = gold.strip()

            # Check if gold starts with the cleaned prediction (more robust matching)
            if gold_clean.startswith(pred_clean) or pred_clean == gold_clean:
                correct += 1

        return correct / total if total > 0 else 0.0

    def _evaluate_stage(
        self,
        stage: str,
        test_loader: DataLoader,
        stage_name: str,
        metric_func: Callable = None,
        epoch: int = None,
    ) -> Dict[str, Any]:
        """Evaluate model on test set for a specific stage."""
        # Enable eval mode for all ranks
        self.model.eval()
        results = []
        test_loss = 0.0

        # Set higher max_tokens for generation during evaluation
        max_new_tokens = 2000

        # Prepare per-rank streaming writer for test predictions
        results_file_rank = os.path.join(self._stage_dir(stage_name),
            "results",
            f"test_predictions_rank_{self.rank if dist.is_initialized() else 0}.jsonl",
        )
        final_results_file = os.path.join(self._stage_dir(stage_name), "results", "test_predictions.jsonl"
        )
        results_fp = None
        # Ensure directory exists (defensive)
        os.makedirs(os.path.dirname(results_file_rank), exist_ok=True)
        if self.rank == 0:
            print(f"[Eval] rank={self.rank}, world_size={self.world_size}")
            print(f"Saving per-rank test predictions to: {results_file_rank}")
            if dist.is_initialized():
                print(
                    f"Final merged predictions will be saved to: {final_results_file}"
                )
        # # Open per-rank file in write mode to start fresh, then append per-sample
        # results_fp = open(results_file_rank, "w", encoding="utf-8")
        # if not results_fp:
        #     raise RuntimeError(
        #         f"Failed to open per-rank results file: {results_file_rank}"
        #     )
        # try:
        #     with torch.no_grad():
        #         for batch in tqdm(
        #             test_loader, desc=f"Evaluating {stage_name}", disable=self.rank != 0
        #         ):
        #             # Generate predictions with higher max_tokens (skip separate loss computation)
        #             predictions = self._get_model().generate(
        #                 batch, max_new_tokens=max_new_tokens
        #             )

        #             # Collect results
        #             for sample, pred in zip(batch, predictions):
        #                 result = {
        #                     "pre_prompt": sample["pre_prompt"],
        #                     "time_series_text": sample["time_series_text"],
        #                     "post_prompt": sample["post_prompt"],
        #                     "generated": pred,
        #                     "gold": sample["answer"],
        #                 }

        #                 # Add time series ID for stage2 captioning
        #                 if stage == "stage2_captioning" and "id" in sample:
        #                     result["time_series_id"] = sample["id"]

        #                 # Add template_id and ecg_id for stage5_ecg_cot
        #                 if stage == "stage5_ecg_cot":
        #                     if "template_id" in sample:
        #                         result["template_id"] = sample["template_id"]
        #                     if "ecg_id" in sample:
        #                         result["ecg_id"] = sample["ecg_id"]
        #                     if "correct_answer" in sample:
        #                         result["correct_answer"] = sample["correct_answer"]
        #                 results.append(result)
        #                 # Stream write each result immediately to per-rank file
        #                 results_fp.write(json.dumps(result, ensure_ascii=False) + "\n")
        #                 results_fp.flush()
        #                 try:
        #                     os.fsync(results_fp.fileno())
        #                 except Exception:
        #                     pass
        # finally:
        #     if results_fp is not None:
        #         results_fp.close()

        # Synchronize all ranks before merging
        if dist.is_initialized():
            dist.barrier()

        # Rank 0 merges per-rank files into final results file
        if (not dist.is_initialized()) or (self.rank == 0):
            try:
                # Overwrite final file each evaluation
                with open(final_results_file, "w", encoding="utf-8") as merged_fp:
                    if dist.is_initialized():
                        num_ranks = self.world_size
                    else:
                        num_ranks = 1
                    for r in range(num_ranks):
                        part_file = os.path.join(self._stage_dir(stage_name),
                            "results",
                            f"test_predictions_rank_{r}.jsonl",
                        )
                        if os.path.exists(part_file):
                            with open(part_file, "r", encoding="utf-8") as pf:
                                for line in pf:
                                    merged_fp.write(line)
                if self.rank == 0:
                    print(f"Merged per-rank predictions into: {final_results_file}")
            finally:
                pass
        avg_test_loss = float("nan")
        # Calculate stage-specific metrics
        metrics = {"test_loss": avg_test_loss}
        if epoch is not None:
            metrics["epoch"] = epoch
        if metric_func:
            # Compute metrics on rank 0 after merging, else minimal metrics
            if (not dist.is_initialized()) or (self.rank == 0):
                predictions = []
                gold_answers = []
                # Read from final merged file
                merged_path = final_results_file
                with open(merged_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            obj = json.loads(line)
                            predictions.append(obj.get("generated", ""))
                            gold_answers.append(obj.get("gold", ""))
                        except Exception:
                            continue
                additional_metrics = metric_func(predictions, gold_answers)
                metrics.update(additional_metrics)
        # Save results only on rank 0 (or when not distributed)
        if (not dist.is_initialized()) or (self.rank == 0):
            # Save metrics
            metrics_file = os.path.join(self._stage_dir(stage_name), "results", "metrics.json"
            )
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)

            print(f"✅ {stage_name} evaluation complete:")
            print(f"   Test predictions saved to: {final_results_file}")
            print(f"   Metrics saved to: {metrics_file}")
            print(f"   Max tokens used for generation: {max_new_tokens}")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"   {metric}: {value:.4f}")
                else:
                    print(f"   {metric}: {value}")

        # Signal other ranks that evaluation is complete
        if dist.is_initialized():
            dist.barrier()

        return metrics

    def _is_evaluation_completed(self, stage: str) -> bool:
        """Check if evaluation was completed for a stage by looking for test predictions file."""
        test_predictions_file = os.path.join(self._stage_dir(stage), "results", "test_predictions.jsonl"
        )
        metrics_file = os.path.join(self._stage_dir(stage), "results", "metrics.json")

        # Check if both files exist
        if not os.path.exists(test_predictions_file) or not os.path.exists(
            metrics_file
        ):
            return False

        # Also check if metrics file has evaluation results
        try:
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
            return "test_loss" in metrics
        except:
            return False

    def _uses_pretrain_loss(self, stage_name: str) -> bool:
        """Return True when a stage should route through compute_pretrain_loss."""
        if stage_name in PHASE1_CE_ONLY_STAGES:
            return False
        return self.args.is_pretraining or stage_name in PHASE1_MSE_ONLY_STAGES

    def _uses_phase1_dataset_kwargs(self, stage_name: str) -> bool:
        """Return True when a stage needs Phase-1 dataset kwargs like patch_size/val_fraction."""
        _excluded = {"joint_phase2_downstream"}
        return stage_name in ((JOINT_STAGES | PHASE1_MSE_ONLY_STAGES | PHASE1_CE_ONLY_STAGES) - _excluded)

    def _uses_task_homogeneous_sampler(self, stage_name: str) -> bool:
        """Keep TSQA/M4 batches task-homogeneous for CE-only/joint Phase-1 stages."""
        return (
            stage_name in (JOINT_PHASE1_STAGES | PHASE1_CE_ONLY_STAGES)
            and not self._uses_pretrain_loss(stage_name)
        )

    def _train_stage(
        self,
        stage_name: str,
        dataset_class,
        num_epochs: int,
        lr_encoder: float,
        lr_projector: float,
        lr_base: float,
        metric_func: Callable = None,
        batch_size: int = None,
        val_batch_size: int = None,
        eval_only: bool = False,
        sampler=None,
        warm_start_from: str = None,
    ) -> Dict[str, Any]:
        """Generic training function for any stage."""
        # ── --build_cache_only short-circuit ──────────────────────────────
        # Construct the three split datasets (which triggers _load_splits →
        # builds Chronos arrow parts, the combined split cache, and the stub
        # cache on disk), then return immediately without training. Used by
        # CPU-only sbatch jobs to pre-build the data caches so subsequent GPU
        # jobs start training in seconds instead of hours.
        if getattr(self.args, "build_cache_only", False):
            if self.rank == 0:
                print(f"📦 build_cache_only — constructing datasets for stage "
                      f"{stage_name!r} to populate the on-disk caches.")
            ds_kwargs = dict(
                EOS_TOKEN=self._get_model().get_eos_token(),
                data_seed=self.args.data_seed,
                patch_size=PATCH_SIZE,
            )
            # joint_phase1_extend_balanced uses JOINT_VAL_FRACTION explicitly,
            # the other stages don't take val_fraction. Pass conservatively.
            if stage_name in JOINT_PHASE1_STAGES:
                ds_kwargs["val_fraction"] = JOINT_VAL_FRACTION
            for split in ("train", "validation", "test"):
                try:
                    ds = dataset_class(split, **ds_kwargs)
                    if self.rank == 0:
                        size = len(ds.rows) if hasattr(ds, "rows") else len(ds)
                        print(f"  -> [{split}] built, {size} rows")
                    del ds
                except Exception as exc:
                    if self.rank == 0:
                        print(f"  -> [{split}] FAILED to build: {exc!r}")
            if self.rank == 0:
                print(f"✅ {stage_name} caches built.")
            return {"status": "cache_built", "stage": stage_name}

        epoch = None
        # Use provided batch_size or default to global BATCH_SIZE
        if batch_size is None:
            batch_size = BATCH_SIZE
        if val_batch_size is None:
            val_batch_size = batch_size

        # stage_dir is used for ALL path-building (checkpoints, results, loss history).
        # When warm_start_from is set, encode the origin in the directory name so
        # different warm-start runs never overwrite each other.
        # stage_name is kept unchanged for all logic/set-membership checks.
        stage_dir = (
            f"{stage_name}_from_{warm_start_from}"
            if warm_start_from is not None
            else stage_name
        )

        if self.rank == 0:
            print(f"\n🚀 Starting {stage_name} Training with {self.model_type}")
            if eval_only:
                print("🔍 EVAL-ONLY MODE: Skipping training, only running evaluation")
            print("=" * 60)
            print(f"📊 Stage Configuration:")
            print(f"   Epochs: {num_epochs}")
            if self.model_type in ["OpenTSLMSP", "OpenTSLMSPFixPos", "OpenTSLMSPFixPos2D", "OpenTSLMSPFixPos2DTimerXL"]:
                print(f"   Encoder LR: {lr_encoder:.2e}")
                print(f"   Projector LR: {lr_projector:.2e}")
            else:
                print(f"   Base LR: {lr_base:.2e}")
            print(f"   Batch size per GPU: {batch_size}")
            print(f"   Validation batch size per GPU: {val_batch_size}")
            if self.world_size > 1:
                print(f"   Effective batch size: {batch_size * self.world_size}")
            print()

        # Check if checkpoint exists when in eval_only mode
        if eval_only and not self._checkpoint_exists(stage_dir):
            raise RuntimeError(
                f"Eval-only mode requires a checkpoint for {stage_dir}, but none found at {os.path.join(self._stage_dir(stage_dir), 'checkpoints', 'best_model.pt')}"
            )

        # Warm-start: load weights from a specific stage before anything else.
        # When warm_start_from is set, skip the automatic curriculum-chain loading.
        if warm_start_from is not None:
            if not self._checkpoint_exists(stage_dir):
                loaded = self._load_weights_from_stage(warm_start_from)
                if self.rank == 0:
                    if loaded:
                        print(f"🔄🔄🔄🔄 Warm Warm-started from stage '{warm_start_from}'")
                    else:
                        print(f"⚠️  Warm-start from '{warm_start_from}' failed — starting fresh")
        elif stage_name in ({"foundation_stage1_chatts"} | JOINT_STAGES):
            # These stages do not require a previous stage.
            # They either start fresh or resume their own existing checkpoint.
            if self.rank == 0:
                if self._checkpoint_exists(stage_dir):
                    print(f"🔄 Resuming {stage_dir} from its own checkpoint")
                else:
                    print(f"🆕 Starting {stage_dir} fresh (no previous stage required)")
            previous_stage_info = None
        else:
            # Load previous stage model and display metrics
            previous_stage_info = self._load_previous_stage_model(stage_name)
            if previous_stage_info:
                if self.rank == 0:
                    print(f"📂 Loading best model from {previous_stage_info['stage']}:")
                    print(f"   Achieved at epoch: {previous_stage_info['epoch']}")
                    val_loss = previous_stage_info["val_loss"]
                    if isinstance(val_loss, (int, float)):
                        print(f"   Validation loss: {val_loss:.4f}")
                    else:
                        print(f"   Validation loss: {val_loss}")
                    for metric, value in previous_stage_info["metrics"].items():
                        if isinstance(value, (int, float)):
                            print(f"   {metric}: {value:.4f}")
                        else:
                            print(f"   {metric}: {value}")
                    print()
            else:
                # Only allow fresh model for first stage or stages that explicitly don't need a previous stage
                stages_without_dependency = {
                    CURRICULUM_STAGES[0],
                } | JOINT_STAGES | PHASE1_MSE_ONLY_STAGES
                if stage_name not in stages_without_dependency:
                    prev_stage = (
                        CURRICULUM_STAGES[CURRICULUM_STAGES.index(stage_name) - 1]
                        if stage_name in CURRICULUM_STAGES
                        else "<unknown>"
                    )
                    raise RuntimeError(
                        f"Cannot start {stage_name} with fresh model. Previous stage {prev_stage} must be completed first."
                    )
                if self.rank == 0:
                    print("🆕 Starting with fresh model (first stage)")

        # Check if evaluation was already completed
        evaluation_completed = self._is_evaluation_completed(stage_dir)
        if evaluation_completed and self.rank == 0:
            print(
                f"✅ Evaluation already completed for {stage_dir}, skipping training and evaluation"
            )
            print(f"📂 Loading existing metrics...")

            # Load and return existing metrics
            metrics_file = os.path.join(self._stage_dir(stage_dir), "results", "metrics.json"
            )
            with open(metrics_file, "r") as f:
                metrics = json.load(f)

            print(f"📊 Existing results for {stage_name}:")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"   {metric}: {value:.4f}")
                else:
                    print(f"   {metric}: {value}")

            return metrics

        # Enable LoRA if needed for this stage. Pass stage_dir too: on a RESUME
        # the adapter's rank must be read from the dir this run actually saved
        # to, which is stage_name + "_from_<warm_start_from>" when warm-starting.
        self._enable_lora_if_needed(stage_name, stage_dir)
        # Unfreeze full LLM if this stage is a full fine-tune stage
        self._unfreeze_llm_if_needed(stage_name)

        # Initialize optimizer and scheduler        
        optimizer = self._get_optimizer(
            batch_size=batch_size,
            lr_encoder=lr_encoder,
            lr_projector=lr_projector,
            lr_base=lr_base,
            stage_name=stage_name,
        )

        # print_trainable(self.model)
        # Create data loaders
        dataset_kwargs = {"EOS_TOKEN": self._get_model().get_eos_token()}
        if self._uses_phase1_dataset_kwargs(stage_name):
            dataset_kwargs["data_seed"] = self.args.data_seed
            # Keep dataset patch semantics aligned with runtime/model config.
            dataset_kwargs["patch_size"] = PATCH_SIZE
            # Use a fraction of each source's val split (preserves task ratio).
            dataset_kwargs["val_fraction"] = JOINT_VAL_FRACTION
            if self.args.max_samples_per_source is not None:
                dataset_kwargs["max_samples_per_source"] = self.args.max_samples_per_source
        if stage_name in {
            "phase1_chronos_seq2seq",
            "phase1_chronos_seq2seq_separate_mse",
        }:
            dataset_kwargs["pred_len"] = PRED_LEN
        if stage_name == "joint_phase2_downstream":
            dataset_kwargs["patch_size"] = PATCH_SIZE
            _wpc = getattr(self.args, "windows_per_col_train", None)
            if _wpc is not None:
                dataset_kwargs["windows_per_col_train"] = _wpc

        if self._uses_task_homogeneous_sampler(stage_name) and sampler is None:
            train_dataset = dataset_class("train", **dataset_kwargs)
            _mtpb = getattr(self.args, "max_tokens_per_batch", None)
            task_batch_sampler = TaskHomogeneousDistributedBatchSampler(
                train_dataset,
                batch_size=batch_size,
                max_tokens_per_batch=_mtpb,
                num_replicas=self.world_size if dist.is_initialized() else 1,
                rank=self.rank if dist.is_initialized() else 0,
                shuffle=True,
                drop_last=True,
                seed=self.args.data_seed,
            )
            _nw = getattr(self.args, "dataloader_workers", 4)
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=task_batch_sampler,
                collate_fn=self._make_collate(PATCH_SIZE),
                num_workers=_nw,
                pin_memory=torch.cuda.is_available() and _nw > 0,
                persistent_workers=(_nw > 0),
                prefetch_factor=(2 if _nw > 0 else None),
            )
            if self.rank == 0:
                print(
                    "🧩 Using task-homogeneous batch sampler for "
                    f"{stage_name} (no TSQA/Chronos mixing inside a batch)."
                )
        elif sampler is not None:
            if isinstance(sampler, TaskHomogeneousDistributedBatchSampler):
                # TaskWeightedDistributedBatchSampler (and any other
                # TaskHomogeneous subclass) is already DDP-aware — use it
                # directly on the dataset it was built with, via the fast
                # path (workers + pin_memory, same as the auto-created
                # task-homogeneous sampler above).
                _nw = getattr(self.args, "dataloader_workers", 4)
                train_loader = DataLoader(
                    sampler.dataset,
                    batch_sampler=sampler,
                    collate_fn=self._make_collate(PATCH_SIZE),
                    num_workers=_nw,
                    pin_memory=torch.cuda.is_available() and _nw > 0,
                    persistent_workers=(_nw > 0),
                    prefetch_factor=(2 if _nw > 0 else None),
                )
                if self.rank == 0:
                    print(
                        f"🧩 Using task-weighted batch sampler for "
                        f"{stage_name} (DDP-aware, task-homogeneous batches)."
                    )
            elif self.world_size > 1:
                get_logger().warning(
                    "BalancedBatchSampler was provided, but distributed training (DDP) is enabled. BalancedBatchSampler will NOT be used. Data will be sharded using DistributedSampler instead. Typically for stage3_cot it is better to use BalancedBatchSampler, if dataset is imbalanced."
                )
                train_loader = self._merge_data_loaders(
                    [
                        dataset_class("train", **dataset_kwargs)
                    ],
                    shuffle=True,
                    batch_size=batch_size,
                    patch_size=PATCH_SIZE,
                    distribute_data=True,
                    drop_last=True,
                )
            else:
                train_dataset = dataset_class("train", **dataset_kwargs)
                train_loader = DataLoader(
                    train_dataset,
                    batch_sampler=sampler,
                    collate_fn=self._make_collate(PATCH_SIZE),
                )
        else:
            if self.args.is_pretraining:
                train_loader = self._merge_data_loaders(
                    [dataset_class("train", **dataset_kwargs)],
                    shuffle=True,
                    batch_size=batch_size,
                    patch_size=PATCH_SIZE,
                    distribute_data=self.world_size > 1,
                    drop_last=self.world_size > 1,
                )
            else:
                train_loader = self._merge_data_loaders(
                    [dataset_class("train", **dataset_kwargs)],
                    shuffle=True,
                    batch_size=batch_size,
                    patch_size=PATCH_SIZE,
                    distribute_data=self.world_size > 1,
                    drop_last=self.world_size > 1,
                )
        if self.args.is_pretraining:
            val_loader = self._merge_data_loaders(
                [dataset_class("validation", **dataset_kwargs)],
                shuffle=False,
                batch_size=val_batch_size,
                patch_size=PATCH_SIZE,
                distribute_data=self.world_size > 1,
            )

            test_loader = self._merge_data_loaders(
                [dataset_class("test", **dataset_kwargs)],
                shuffle=False,
                batch_size=1,
                patch_size=PATCH_SIZE,
                distribute_data=self.world_size > 1,
            )
        else:
            val_loader = self._merge_data_loaders(
                [dataset_class("validation", **dataset_kwargs)],
                shuffle=False,
                batch_size=val_batch_size,
                patch_size=PATCH_SIZE,
                distribute_data=self.world_size > 1,
            )

            test_loader = self._merge_data_loaders(
                [dataset_class("test", **dataset_kwargs)],
                shuffle=False,
                batch_size=1,
                patch_size=PATCH_SIZE,
                distribute_data=self.world_size > 1,
            )

        self._assert_distributed_loader_lengths_match(train_loader, stage_name)

        # Scheduler
        grad_accum = self.args.grad_accum_steps
        total_steps = num_epochs * len(train_loader) // grad_accum
        warmup_steps = int(WARMUP_FRAC * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # Mixed precision: bfloat16 on A100+, float16 on V100/other GPUs
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))

        if self.rank == 0:
            print(f"📈 Total training steps: {total_steps}")
            print(f"🔥 Warmup steps: {warmup_steps}")

        # Load previous checkpoint if exists (for resuming current stage)
        best_epoch, best_val_loss = self._load_checkpoint(
            stage_dir, optimizer, scheduler, eval_only=eval_only, scaler=scaler
        )
        if best_epoch is not None:
            print(
                f"📂 Resuming {stage_dir} from epoch {best_epoch} (val_loss: {best_val_loss:.4f})"
            )
            # Display previous loss history if available
            self._display_loss_history(stage_dir)
        else:
            print(f"🆕 Starting fresh training for {stage_dir}")
            best_val_loss = float("inf")  # Ensure proper initialization

        # For joint stages: normalise MSE and CE val losses by their epoch-1
        # values so both contribute equally to the early-stopping criterion.
        # Separate best/patience trackers for joint stages
        best_val_mse: float = float("inf")
        best_val_ce:  float = float("inf")
        epochs_no_improve_mse: int = 0
        epochs_no_improve_ce:  int = 0
        # Phase-2 CoT per-task early stopping (PHASE2_PERTASK_EARLYSTOP=1): 3 separate
        # CE patience counters (HAR/Sleep/ECG). Stop if ANY task stagnates for
        # EARLY_STOP_PAT epochs; best_model saved on the EQUAL-weighted mean of the 3
        # (not the ECG-dominated avg_val_loss). Default off -> existing behaviour.
        # PHASE2_SELECT_BY_GEN=1 uses per-task GENERATION exact-match (not teacher-forced
        # CE) as the per-task metric — catches generation collapse that CE can't see.
        # Implies per-task early stopping. Metric is stored as (1 - EM) so lower=better
        # and the same best-select + 3-way patience machinery below works unchanged.
        _select_by_gen: bool = os.environ.get("PHASE2_SELECT_BY_GEN") == "1"
        _pertask_es: bool = _select_by_gen or os.environ.get("PHASE2_PERTASK_EARLYSTOP") == "1"
        import opentslm.model_config as _mc_es
        EARLY_STOP_PAT = int(_os_es.environ.get("PHASE2_EARLY_STOP_PAT", _mc_es.EARLY_STOP_PAT))
        best_val_cot: dict = {"har": float("inf"), "sleep": float("inf"), "ecg": float("inf"), "pamap2": float("inf"), "tsqa": float("inf")}
        best_val_cot_mean: float = float("inf")
        epochs_no_improve_cot: dict = {"har": 0, "sleep": 0, "ecg": 0, "pamap2": 0, "tsqa": 0}
        cot_improved: dict = {"har": False, "sleep": False, "ecg": False, "pamap2": False, "tsqa": False}
        _gen_em_cache: dict = {}   # {task_type: [sample rows]}, built once
        _gen_em_warned: set = set()  # tasks whose generate() has already been reported broken
        _joint_init_mse = None
        _joint_init_ce = None
        if stage_name in JOINT_CE_MSE_STAGES:
            loaded_joint_meta = getattr(self, "_last_loaded_checkpoint_meta", {}) or {}
            if loaded_joint_meta.get("joint_init_mse") is not None:
                _joint_init_mse = float(loaded_joint_meta["joint_init_mse"])
            if loaded_joint_meta.get("joint_init_ce") is not None:
                _joint_init_ce = float(loaded_joint_meta["joint_init_ce"])

        def _should_skip_empty_batch(
            batch, split: str, batch_idx: int, sync_across_ranks: bool = True
        ) -> bool:
            local_has_batch = int(isinstance(batch, list) and len(batch) > 0)
            if not dist.is_initialized() or not sync_across_ranks:
                if local_has_batch == 0 and self.rank == 0:
                    tqdm.write(
                        f"[WARN] Skipping empty {split} batch at index {batch_idx}."
                    )
                return local_has_batch == 0

            min_has_batch = torch.tensor(local_has_batch, device=self.device, dtype=torch.int32)
            max_has_batch = torch.tensor(local_has_batch, device=self.device, dtype=torch.int32)
            dist.all_reduce(min_has_batch, op=dist.ReduceOp.MIN)
            dist.all_reduce(max_has_batch, op=dist.ReduceOp.MAX)

            should_skip = bool(min_has_batch.item() == 0)
            if should_skip and self.rank == 0:
                if max_has_batch.item() == 0:
                    tqdm.write(
                        f"[WARN] Skipping {split} batch {batch_idx}: all ranks filtered out every sample."
                    )
                else:
                    tqdm.write(
                        f"[WARN] Skipping {split} batch {batch_idx}: at least one rank filtered out every sample."
                    )
            return should_skip

        # Skip training loop if eval_only is True
        if eval_only:
            if self.rank == 0:
                print(f"⏭️  Skipping training loop (eval_only mode)")
                print(f"📂 Using existing checkpoint for evaluation")
            epoch = best_epoch
            epochs_no_improve = 0
        else:
            # Training loop
            epochs_no_improve = 0
            start_epoch = best_epoch + 1 if best_epoch is not None else 1
            train_log_every = max(1, getattr(self.args, "train_log_every_n_steps", 1))
            joint_monitor_every = max(
                1, getattr(self.args, "joint_monitor_every_n_epochs", 1)
            )
            joint_generate_every = max(
                0, getattr(self.args, "joint_generate_every_n_epochs", 2)
            )
            # Initialize once before training starts (e.g., in main() or __init__)
            if self.rank == 0:
                wandb_project = (
                    self.args.wandb_project
                    if getattr(self.args, "wandb_project", None)
                    else f"opentslm-pretrain-{self.args.stages}"
                )
                wandb_name = (
                    self.args.wandb_run_name
                    if getattr(self.args, "wandb_run_name", None)
                    else f"pretrain-{self.args.stages}_kl{self.args.kl_beta}"
                )
                wandb.init(
                    project=wandb_project,
                    config=vars(self.args),
                    name=wandb_name,
                )

            # Global optimizer step counter — persists across epochs and accounts
            # for resumed training so wandb x-axis is monotonically correct.
            steps_per_epoch = max(1, len(train_loader) // grad_accum)
            global_optimizer_step = (start_epoch - 1) * steps_per_epoch


            # ── Periodic validation state ─────────────────────────────────
            # Validation runs every --eval_every_n_iters optimizer steps (and
            # once at each epoch tail). Early stopping stays epoch-level and is
            # driven by the improved_this_epoch flags set inside _run_validation.
            eval_every_n_iters = max(1, getattr(self.args, "eval_every_n_iters", 500))
            eval_max_samples_per_task = getattr(self.args, "eval_max_samples_per_task", 1000)
            last_eval_step = global_optimizer_step
            _eval_count = 0
            improved_this_epoch = False
            mse_improved_this_epoch = False
            ce_improved_this_epoch = False

            def _run_validation(epoch, global_optimizer_step, avg_train_loss,
                                max_samples_per_task=None):
                """Validation + wandb logging + best-checkpoint save.

                Called every --eval_every_n_iters optimizer steps and at each
                epoch tail. Sets the nonlocal improved_this_epoch flags; the
                epoch loop reads them for early-stopping decisions.

                max_samples_per_task: global per-task sample cap (chronos / tsqa
                / m4), split across ranks. None / <=0 = evaluate the full val
                set (used by the epoch-tail call).
                """
                nonlocal best_val_loss, best_val_mse, best_val_ce
                nonlocal _joint_init_mse, _joint_init_ce, _eval_count
                nonlocal improved_this_epoch, mse_improved_this_epoch, ce_improved_this_epoch
                nonlocal best_val_cot_mean  # per-task CoT early stop (dicts mutate in place)

                # Controlled experiment: when SKIP_VAL_FOR_LEAK_TEST=1, no-op
                # the entire val function. Used to test whether validation is
                # the trigger for the epoch-2+ autograd graph leak.
                if os.environ.get("SKIP_VAL_FOR_LEAK_TEST", "0") == "1":
                    if self.rank == 0:
                        tqdm.write(f"[SKIP_VAL_FOR_LEAK_TEST=1] skipping val for epoch {epoch}")
                    return

                _eval_count += 1

                # Per-task sample cap. The arg is a global count; divide by the
                # number of ranks so the union across ranks ≈ the requested cap.
                _per_rank_cap = None
                if max_samples_per_task and max_samples_per_task > 0:
                    _per_rank_cap = max(1, int(max_samples_per_task) // max(1, self.world_size))
                _relevant_tasks = (
                    {"chronos"} if stage_name in PHASE1_MSE_ONLY_STAGES
                    else {"chronos", "tsqa", "m4"}
                )
                _task_seen = {"chronos": 0, "tsqa": 0, "m4": 0}

                # Validation
                val_loss = 0.0
                val_mse_loss  = 0.0  # tracked separately for joint stages
                val_ce_loss   = 0.0  # tracked separately for joint stages
                val_tsqa_loss = 0.0  # TSQA CE only (for plotting)
                val_m4_loss   = 0.0  # M4 CE only (for plotting)
                val_next_patch_mse = 0.0
                val_final_patch_mse = 0.0
                num_val_steps = 0
                # Per-task batch counts so we can average task-specific losses
                # only over batches that actually contained that task. Without
                # this, sparse tasks (e.g. a Chronos corpus of 1 row) get their
                # MSE massively diluted by zero contributions from CE-only batches.
                num_val_chronos_batches = 0
                num_val_tsqa_batches    = 0
                num_val_m4_batches      = 0
                num_val_ce_batches      = 0
                # Per-source chronos MSE so wandb shows chronos_uv vs synthetic
                # separately. We re-run the loss on each per-source sub-batch
                # rather than touching the model's loss internals.
                val_mse_chronos_uv      = 0.0
                val_mse_synthetic       = 0.0
                num_val_chronos_uv_batches  = 0
                num_val_synthetic_batches   = 0
                # Per-task tracking for joint_phase2_downstream. Single-task batches
                # (TaskHomogeneousDistributedBatchSampler) let us bucket each batch's
                # loss by batch[0]["type"]. Without this, val_loss is an aggregate
                # that hides which sub-task actually learned.
                val_har_cot_ce          = 0.0
                val_sleep_cot_ce        = 0.0
                val_ecg_cot_ce          = 0.0
                val_chatts_ce           = 0.0
                val_downstream_mse      = 0.0
                num_val_har_cot_batches    = 0
                num_val_sleep_cot_batches  = 0
                num_val_ecg_cot_batches    = 0
                num_val_chatts_batches     = 0
                num_val_downstream_batches = 0
                monitor_batch = []
                _log_gpu_mem(f"epoch{epoch} train END (pre empty_cache)")
                torch.cuda.empty_cache()
                _log_gpu_mem(f"epoch{epoch} train END (post empty_cache)")
                self.model.eval()
                with torch.no_grad(), torch.amp.autocast('cuda', dtype=amp_dtype):
                    for j, batch in enumerate(tqdm(
                        val_loader,
                        desc=f"Validating {stage_name}",
                        disable=self.rank != 0,
                    )):
                        if _should_skip_empty_batch(batch, "validation", j):
                            continue

                        # Per-task sample cap (periodic evals only). Filter the
                        # batch so each task contributes at most _per_rank_cap
                        # samples; once every relevant task is full, all ranks
                        # break together (the done check is all-reduced so the
                        # loop length stays in sync across ranks).
                        if _per_rank_cap is not None:
                            _local_done = all(
                                _task_seen[t] >= _per_rank_cap for t in _relevant_tasks
                            )
                            if dist.is_initialized():
                                _d = torch.tensor(
                                    1 if _local_done else 0, device=self.device
                                )
                                dist.all_reduce(_d, op=dist.ReduceOp.MIN)
                                _local_done = _d.item() > 0
                            if _local_done:
                                break
                            _kept = []
                            for _s in batch:
                                _t = _s.get("type")
                                _t = "m4" if _t in ("m4", "m4_captioning") else _t
                                if _t not in _task_seen:
                                    _kept.append(_s)          # unknown type: keep, uncapped
                                elif _task_seen[_t] < _per_rank_cap:
                                    _kept.append(_s)
                                    _task_seen[_t] += 1
                            batch = _kept
                            if not batch:
                                continue

                        if (
                            stage_name in JOINT_STAGES
                            and self.rank == 0
                            and len(monitor_batch) < 16
                        ):
                            monitor_batch.extend(batch[: 16 - len(monitor_batch)])

                        if self._uses_pretrain_loss(stage_name):
                            if stage_name == "phase1_chronos_seq2seq":
                                loss, ntp_loss, recon_loss = self._get_model().compute_chronos_seq2seq_loss(batch)
                            elif stage_name == "phase1_chronos_seq2seq_separate_mse":
                                loss, ntp_loss, recon_loss = self._get_model().compute_chronos_seq2seq_enc_mse_loss(batch)
                            else:
                                loss, ntp_loss, recon_loss = self._get_model().compute_pretrain_loss(
                                    batch, mse_only=stage_name in PHASE1_MSE_ONLY_STAGES
                                )
                            if stage_name in PHASE1_MSE_ONLY_STAGES:
                                val_loss += recon_loss.item()  # raw MSE, independent of recon_alpha scaling
                                model_ref = self._get_model()
                                val_next_patch_mse += float(
                                    getattr(model_ref, "_last_pretrain_next_patch_mse", loss.new_zeros(())).item()
                                )
                                val_final_patch_mse += float(
                                    getattr(model_ref, "_last_pretrain_final_patch_mse", loss.new_zeros(())).item()
                                )
                                # Per-source chronos MSE: re-run on filtered sub-batches.
                                # Cheap because val is small and only runs once per epoch.
                                chr_uv_sub  = [s for s in batch if s.get("source") == "chronos_uv"]
                                chr_syn_sub = [s for s in batch if s.get("source") == "synthetic"]
                                if chr_uv_sub:
                                    _, _, rec_uv = self._get_model().compute_pretrain_loss(
                                        chr_uv_sub, mse_only=True,
                                    )
                                    val_mse_chronos_uv += float(rec_uv.item())
                                    num_val_chronos_uv_batches += 1
                                if chr_syn_sub:
                                    _, _, rec_syn = self._get_model().compute_pretrain_loss(
                                        chr_syn_sub, mse_only=True,
                                    )
                                    val_mse_synthetic += float(rec_syn.item())
                                    num_val_synthetic_batches += 1
                            else:
                                val_loss += loss.item()
                        else:
                            if stage_name in ("stage2_all"):
                                val_loss += self._get_model().compute_phase2_loss(batch, pred_len=PRED_LEN, patch_size=PATCH_SIZE)[0].item()
                            elif stage_name in JOINT_PHASE1_STAGES:
                                _val_loss_fn = (
                                    self._get_model().compute_joint_phase1_enc_mse_loss
                                    if stage_name == "joint_phase1_enc_mse"
                                    else self._get_model().compute_joint_phase1_loss
                                )
                                loss, loss_ce, loss_tsqa, loss_m4, loss_mse = _val_loss_fn(
                                    batch,
                                    detach_llm_for_mse=(stage_name in JOINT_PHASE1_DETACH_LLM_MSE_STAGES or getattr(self.args, "detach_llm_for_mse", False)),
                                )
                                val_loss      += loss.item()
                                val_mse_loss  += loss_mse.item()
                                val_ce_loss   += loss_ce.item()
                                val_tsqa_loss += loss_tsqa.item()
                                val_m4_loss   += loss_m4.item()
                                # Track which tasks this batch contained so we can
                                # divide task-specific losses by the right counts.
                                has_chronos = any(s.get("type") == "chronos" for s in batch)
                                has_tsqa    = any(s.get("type") == "tsqa"    for s in batch)
                                has_m4      = any(s.get("type") in ("m4", "m4_captioning") for s in batch)
                                if has_chronos:
                                    num_val_chronos_batches += 1
                                if has_tsqa:
                                    num_val_tsqa_batches += 1
                                if has_m4:
                                    num_val_m4_batches += 1
                                if has_tsqa or has_m4:
                                    num_val_ce_batches += 1
                                # Per-source chronos MSE: re-run joint loss on
                                # chronos-only sub-batches filtered by source.
                                # Cheap because val is small and only runs once per epoch.
                                if has_chronos:
                                    chr_uv_sub  = [s for s in batch if s.get("type")=="chronos" and s.get("source")=="chronos_uv"]
                                    chr_syn_sub = [s for s in batch if s.get("type")=="chronos" and s.get("source")=="synthetic"]
                                    if chr_uv_sub:
                                        _, _, _, _, mse_uv = _val_loss_fn(
                                            chr_uv_sub,
                                            detach_llm_for_mse=(stage_name in JOINT_PHASE1_DETACH_LLM_MSE_STAGES or getattr(self.args, "detach_llm_for_mse", False)),
                                        )
                                        val_mse_chronos_uv += float(mse_uv.item())
                                        num_val_chronos_uv_batches += 1
                                    if chr_syn_sub:
                                        _, _, _, _, mse_syn = _val_loss_fn(
                                            chr_syn_sub,
                                            detach_llm_for_mse=(stage_name in JOINT_PHASE1_DETACH_LLM_MSE_STAGES or getattr(self.args, "detach_llm_for_mse", False)),
                                        )
                                        val_mse_synthetic += float(mse_syn.item())
                                        num_val_synthetic_batches += 1
                            elif stage_name in JOINT_PHASE2_STAGES:
                                try:
                                    loss, loss_ce, loss_mse = self._get_model().compute_joint_phase2_loss(
                                        batch,
                                        alpha=getattr(self.args, "mse_alpha", 1.0),
                                    )
                                    val_loss     += loss.item()
                                    val_mse_loss += loss_mse.item()
                                    val_ce_loss  += loss_ce.item()
                                    # Bucket per task type (batches are single-task).
                                    _task_type = batch[0].get("type", "") if batch else ""
                                    _ce_item = loss_ce.item()
                                    _mse_item = loss_mse.item()
                                    if _task_type == "har_cot":
                                        val_har_cot_ce += _ce_item
                                        num_val_har_cot_batches += 1
                                    elif _task_type == "sleep_cot":
                                        val_sleep_cot_ce += _ce_item
                                        num_val_sleep_cot_batches += 1
                                    elif _task_type == "ecg_cot":
                                        val_ecg_cot_ce += _ce_item
                                        num_val_ecg_cot_batches += 1
                                    elif _task_type == "chatts":
                                        val_chatts_ce += _ce_item
                                        num_val_chatts_batches += 1
                                    elif _task_type == "chronos":
                                        val_downstream_mse += _mse_item
                                        num_val_downstream_batches += 1
                                except torch.cuda.OutOfMemoryError:
                                    torch.cuda.empty_cache()
                                    if self.rank == 0:
                                        tqdm.write(f"⚠️  Val OOM at batch {j}, skipping")
                                    continue
                            else:
                                val_loss += self._get_model().compute_loss(batch).item()
                        num_val_steps += 1

                    # The main val loop draws batches from a uniform sampler
                    # over a pool dominated by synthetic (~500k stubs vs ~5
                    # chronos_uv stubs). The per-task cap fires before any uv
                    # batch is ever drawn, so val_mse_chronos_uv stays NaN.
                    # Iterate the uv subset explicitly here. Trivial cost
                    # (~5 stubs per rank), guarantees the metric is populated.
                    if stage_name in PHASE1_MSE_ONLY_STAGES or stage_name in JOINT_PHASE1_STAGES:
                        try:
                            from torch.utils.data import Subset
                            _inner_ds = val_loader.dataset.datasets[0]
                            _rows = getattr(_inner_ds, "rows", None) or []
                            _uv_idx = [
                                i for i, r in enumerate(_rows)
                                if isinstance(r, dict) and r.get("source") == "chronos_uv"
                            ]
                            if _uv_idx:
                                _uv_loader = DataLoader(
                                    Subset(_inner_ds, _uv_idx),
                                    batch_size=self.args.val_batch_size,
                                    shuffle=False,
                                    num_workers=0,
                                    collate_fn=val_loader.collate_fn,
                                )
                                for _b in _uv_loader:
                                    if stage_name in PHASE1_MSE_ONLY_STAGES:
                                        _, _, _r = self._get_model().compute_pretrain_loss(
                                            _b, mse_only=True,
                                        )
                                    else:
                                        _loss_fn = (
                                            self._get_model().compute_joint_phase1_enc_mse_loss
                                            if stage_name == "joint_phase1_enc_mse"
                                            else self._get_model().compute_joint_phase1_loss
                                        )
                                        _, _, _, _, _r = _loss_fn(
                                            _b,
                                            detach_llm_for_mse=(
                                                stage_name in JOINT_PHASE1_DETACH_LLM_MSE_STAGES
                                                or getattr(self.args, "detach_llm_for_mse", False)
                                            ),
                                        )
                                    val_mse_chronos_uv += float(_r.item())
                                    num_val_chronos_uv_batches += 1
                        except Exception as _e:
                            if self.rank == 0:
                                print(f"[val] dedicated chronos_uv pass skipped: {_e!r}")

                # Save before AllReduce so training isn't lost if a rank dies
                # during val metric reduction.
                self._save_periodic_checkpoint(stage_dir, epoch, 0, optimizer, scheduler)

                val_metrics = torch.tensor(
                    [
                        val_loss,
                        val_mse_loss,
                        val_ce_loss,
                        val_tsqa_loss,
                        val_m4_loss,
                        val_next_patch_mse,
                        val_final_patch_mse,
                        float(num_val_steps),
                        float(num_val_chronos_batches),
                        float(num_val_tsqa_batches),
                        float(num_val_m4_batches),
                        float(num_val_ce_batches),
                        val_mse_chronos_uv,
                        val_mse_synthetic,
                        float(num_val_chronos_uv_batches),
                        float(num_val_synthetic_batches),
                        # joint_phase2_downstream per-task tracking (sums + counts)
                        val_har_cot_ce,
                        val_sleep_cot_ce,
                        val_ecg_cot_ce,
                        val_chatts_ce,
                        val_downstream_mse,
                        float(num_val_har_cot_batches),
                        float(num_val_sleep_cot_batches),
                        float(num_val_ecg_cot_batches),
                        float(num_val_chatts_batches),
                        float(num_val_downstream_batches),
                    ],
                    device=self.device,
                    dtype=torch.float32,
                )
                if dist.is_initialized():
                    dist.all_reduce(val_metrics, op=dist.ReduceOp.SUM)
                (
                    tot_val_loss,
                    tot_val_mse_loss,
                    tot_val_ce_loss,
                    tot_val_tsqa_loss,
                    tot_val_m4_loss,
                    tot_val_next_patch_mse,
                    tot_val_final_patch_mse,
                    tot_val_steps,
                    tot_chronos_batches,
                    tot_tsqa_batches,
                    tot_m4_batches,
                    tot_ce_batches,
                    tot_mse_chronos_uv,
                    tot_mse_synthetic,
                    tot_chronos_uv_batches,
                    tot_synthetic_batches,
                    tot_val_har_cot_ce,
                    tot_val_sleep_cot_ce,
                    tot_val_ecg_cot_ce,
                    tot_val_chatts_ce,
                    tot_val_downstream_mse,
                    tot_har_cot_batches,
                    tot_sleep_cot_batches,
                    tot_ecg_cot_batches,
                    tot_chatts_batches,
                    tot_downstream_batches,
                ) = [float(v) for v in val_metrics.tolist()]
                _log_gpu_mem(f"epoch{epoch} val END (pre empty_cache)")
                torch.cuda.empty_cache()
                _log_gpu_mem(f"epoch{epoch} val END (post empty_cache)")
                total_val_steps = max(1.0, tot_val_steps)
                # Per-task losses are averaged only over batches that contained
                # that task; batches where the task was absent contributed 0 to
                # the sum (see compute_joint_phase1_loss) and would otherwise
                # dilute the reported metric.
                avg_val_loss            = tot_val_loss / total_val_steps
                avg_val_mse_loss        = tot_val_mse_loss        / max(1.0, tot_chronos_batches)
                avg_val_ce_loss         = tot_val_ce_loss         / max(1.0, tot_ce_batches)
                avg_val_tsqa_loss       = tot_val_tsqa_loss       / max(1.0, tot_tsqa_batches)
                avg_val_m4_loss         = tot_val_m4_loss         / max(1.0, tot_m4_batches)
                avg_val_next_patch_mse  = tot_val_next_patch_mse  / total_val_steps
                avg_val_final_patch_mse = tot_val_final_patch_mse / total_val_steps
                # Per-source chronos MSE: NaN if no batches contained that source
                # (e.g. chronos_uv with 5 train rows may sometimes appear in 0
                # val batches). NaN signals "no data" in wandb, distinct from 0.
                avg_val_mse_chronos_uv  = (
                    tot_mse_chronos_uv / tot_chronos_uv_batches
                    if tot_chronos_uv_batches > 0 else float("nan")
                )
                avg_val_mse_synthetic   = (
                    tot_mse_synthetic / tot_synthetic_batches
                    if tot_synthetic_batches > 0 else float("nan")
                )
                avg_val_har_cot_ce = (
                    tot_val_har_cot_ce / tot_har_cot_batches
                    if tot_har_cot_batches > 0 else float("nan")
                )
                avg_val_sleep_cot_ce = (
                    tot_val_sleep_cot_ce / tot_sleep_cot_batches
                    if tot_sleep_cot_batches > 0 else float("nan")
                )
                avg_val_ecg_cot_ce = (
                    tot_val_ecg_cot_ce / tot_ecg_cot_batches
                    if tot_ecg_cot_batches > 0 else float("nan")
                )
                avg_val_chatts_ce = (
                    tot_val_chatts_ce / tot_chatts_batches
                    if tot_chatts_batches > 0 else float("nan")
                )
                avg_val_downstream_mse = (
                    tot_val_downstream_mse / tot_downstream_batches
                    if tot_downstream_batches > 0 else float("nan")
                )

                # For joint stages: normalise MSE and CE by their epoch-1 values so
                # both contribute equally regardless of absolute scale, then average.
                # This prevents CE (numerically larger) from dominating early stopping.
                if stage_name in JOINT_CE_MSE_STAGES:
                    if _joint_init_mse is None or _joint_init_ce is None:
                        _joint_init_mse = avg_val_mse_loss if avg_val_mse_loss > 0 else 1.0
                        _joint_init_ce  = avg_val_ce_loss  if avg_val_ce_loss  > 0 else 1.0
                    norm_mse = avg_val_mse_loss / _joint_init_mse
                    norm_ce  = avg_val_ce_loss  / _joint_init_ce
                    early_stop_loss = 0.5 * norm_mse + 0.5 * norm_ce
                else:
                    early_stop_loss = avg_val_loss

                if self.rank == 0:
                    tqdm.write(f"Epoch {epoch} — val   loss: {avg_val_loss:.4f}")
                    if stage_name in PHASE1_MSE_ONLY_STAGES:
                        tqdm.write(
                            f"Epoch {epoch} — val   next_patch_mse={avg_val_next_patch_mse:.4f}  "
                            f"final_patch_mse={avg_val_final_patch_mse:.4f}"
                        )
                    if stage_name in JOINT_CE_MSE_STAGES:
                        tqdm.write(
                            f"Epoch {epoch} — val   mse={avg_val_mse_loss:.4f}  "
                            f"ce={avg_val_ce_loss:.4f}  "
                            f"early-stop={early_stop_loss:.4f} (norm avg)"
                        )
                    tqdm.write(
                        f"Epoch {epoch} — val   chronos_uv_mse="
                        f"{avg_val_mse_chronos_uv:.4f}  "
                        f"synthetic_mse={avg_val_mse_synthetic:.4f}  "
                        f"tsqa_ce={avg_val_tsqa_loss:.4f}  "
                        f"m4_ce={avg_val_m4_loss:.4f}"
                    )
                    if stage_name == "joint_phase2_downstream":
                        tqdm.write(
                            f"Epoch {epoch} — val   "
                            f"har_cot_ce={avg_val_har_cot_ce:.4f}  "
                            f"sleep_cot_ce={avg_val_sleep_cot_ce:.4f}  "
                            f"ecg_cot_ce={avg_val_ecg_cot_ce:.4f}  "
                            f"chatts_ce={avg_val_chatts_ce:.4f}  "
                            f"downstream_mse={avg_val_downstream_mse:.4f}"
                        )
                    tqdm.write(f"Epoch {epoch} — best  loss: {best_val_loss:.4f}")
                    if self._uses_pretrain_loss(stage_name):
                        if stage_name in PHASE1_MSE_ONLY_STAGES:
                            wandb.log(
                                {
                                    "val/next_patch_mse": avg_val_next_patch_mse,
                                    "val/final_patch_mse": avg_val_final_patch_mse,
                                    "val/loss": avg_val_loss,
                                    "val/mse_loss_chronos_uv": avg_val_mse_chronos_uv,
                                    "val/mse_loss_synthetic":  avg_val_mse_synthetic,
                                },
                                step=global_optimizer_step,
                            )
                        else:
                            wandb.log({"pretrain/val_loss": avg_val_loss}, step=global_optimizer_step)
                    elif stage_name in JOINT_STAGES:
                        wandb.log(
                            {
                                "val/loss": avg_val_loss,
                                "val/mse_loss": avg_val_mse_loss,
                                "val/ce_loss": avg_val_ce_loss,
                                "val/tsqa_loss": avg_val_tsqa_loss,
                                "val/m4_loss": avg_val_m4_loss,
                                "val/mse_loss_chronos_uv": avg_val_mse_chronos_uv,
                                "val/mse_loss_synthetic":  avg_val_mse_synthetic,
                            },
                            step=global_optimizer_step,
                        )
                    else:
                        wandb.log({"val/loss": avg_val_loss}, step=global_optimizer_step)

                # ── Phase-1/2 deep monitoring ────────────────────────────────
                # PERMANENTLY DISABLED: compute_loss_per_task and
                # generate_phase1_samples leak ~25 MB/step of autograd graph
                # in subsequent training (Gemma3 layer outputs retained).
                # Verified by Option-C ablation: with these skipped, alloc
                # stays flat at 0.57 GB through entire epochs. Main val loop
                # alone (above) is clean. Per-task metrics no longer logged
                # but best-ckpt selection still uses avg_val_loss, so no
                # training quality impact.
                if False and (
                    stage_name in JOINT_STAGES
                    and self.rank == 0
                    and monitor_batch
                    and _eval_count % joint_monitor_every == 0
                ):
                    with torch.no_grad(), torch.amp.autocast('cuda', dtype=amp_dtype):
                        per_task = self._get_model().compute_loss_per_task(monitor_batch)
                    wandb.log({f"val/{k}": v for k, v in per_task.items()}, step=global_optimizer_step)
                    tqdm.write(
                        "  Per-task val — " +
                        "  ".join(f"{k}={v:.4f}" for k, v in per_task.items())
                    )
                    if joint_generate_every > 0 and _eval_count % joint_generate_every == 0:
                        self._get_model().generate_phase1_samples(monitor_batch)

                # Save loss history for this epoch
                self._save_loss_history(stage_dir, epoch, avg_train_loss, avg_val_loss)


                # ── Save best checkpoint (early stopping is epoch-level) ──
                is_joint = stage_name in JOINT_CE_MSE_STAGES
                _cot_mean = float("inf")
                if _pertask_es:
                    # Per-task CoT: track each task's own best metric + improvement, and
                    # select best_model on the EQUAL-weighted mean of present tasks
                    # (so ECG can't dominate and mask HAR/Sleep degradation).
                    mse_improved = ce_improved = False
                    if _select_by_gen:
                        _em = _gen_em()   # {task: EM} or None on failure
                        if _em is not None:
                            # store (1 - EM) so lower=better (same as CE machinery)
                            _vals = {t: (1.0 - v if v == v else float("nan")) for t, v in _em.items()}
                            if self.rank == 0:
                                tqdm.write("  gen-EM — " + "  ".join(
                                    f"{t}={v:.3f}" for t, v in _em.items() if v == v))
                        else:   # generation failed -> fall back to CE this epoch
                            _vals = {"har": avg_val_har_cot_ce, "sleep": avg_val_sleep_cot_ce,
                                     "ecg": avg_val_ecg_cot_ce}
                    else:
                        _vals = {"har": avg_val_har_cot_ce, "sleep": avg_val_sleep_cot_ce,
                                 "ecg": avg_val_ecg_cot_ce}
                    for _t, _v in _vals.items():
                        cot_improved[_t] = (_v == _v and _v + 1e-4 < best_val_cot[_t])
                        if cot_improved[_t]:
                            best_val_cot[_t] = _v
                    _present = [_v for _v in _vals.values() if _v == _v]
                    _cot_mean = sum(_present) / len(_present) if _present else float("inf")
                    should_save = _cot_mean + 1e-4 < best_val_cot_mean
                elif is_joint:
                    mse_improved = avg_val_mse_loss + 1e-4 < best_val_mse
                    ce_improved  = avg_val_ce_loss  + 1e-4 < best_val_ce
                    should_save  = avg_val_loss + 1e-4 < best_val_loss
                else:
                    mse_improved = ce_improved = False
                    should_save  = avg_val_loss + 1e-4 < best_val_loss

                if dist.is_initialized():
                    save_tensor = torch.tensor(1 if should_save else 0, device=self.device)
                    dist.all_reduce(save_tensor, op=dist.ReduceOp.SUM)
                    should_save = save_tensor.item() > 0

                if should_save:
                    best_val_loss = avg_val_loss
                    if _pertask_es:
                        best_val_cot_mean = _cot_mean
                    improved_this_epoch = True
                    checkpoint_extra = {"scaler_state": scaler.state_dict()}
                    if is_joint:
                        checkpoint_extra["joint_init_mse"] = _joint_init_mse
                        checkpoint_extra["joint_init_ce"] = _joint_init_ce
                    self._save_checkpoint(
                        stage_dir, epoch, avg_val_loss, optimizer, scheduler,
                        extra_state=checkpoint_extra,
                    )
                    if self.rank == 0:
                        tqdm.write("\u2714\ufe0f  New best model saved.\n")

                if is_joint:
                    if mse_improved:
                        best_val_mse = avg_val_mse_loss
                        mse_improved_this_epoch = True
                    if ce_improved:
                        best_val_ce = avg_val_ce_loss
                        ce_improved_this_epoch = True

            # ── Diagnostic: GPU memory probe ────────────────────────────────
            # Tracks alloc / reserved / free over training to find fragmentation
            # patterns that cause epoch-2+ OOMs. Lightweight (one mem_get_info
            # call per probe).
            def _log_gpu_mem(tag: str):
                if not torch.cuda.is_available() or self.rank != 0:
                    return
                free, total = torch.cuda.mem_get_info()
                alloc       = torch.cuda.memory_allocated()
                reserved    = torch.cuda.memory_reserved()
                max_alloc   = torch.cuda.max_memory_allocated()
                max_reserv  = torch.cuda.max_memory_reserved()
                frag        = 1.0 - (alloc / max(reserved, 1))
                print(
                    f"[mem {tag}] alloc={alloc/1024**3:.2f}G "
                    f"reserved={reserved/1024**3:.2f}G "
                    f"free={free/1024**3:.2f}G/{total/1024**3:.2f}G "
                    f"max_alloc={max_alloc/1024**3:.2f}G "
                    f"max_reserved={max_reserv/1024**3:.2f}G "
                    f"frag={frag:.1%}",
                    flush=True,
                )

            # ── Diagnostic: enable allocation history recording so we can
            # dump snapshots later and trace which Python lines own the
            # leaked tensors. Rank 0 only (high memory overhead per rank).
            _mem_history_enabled = False
            if self.rank == 0 and torch.cuda.is_available():
                try:
                    torch.cuda.memory._record_memory_history(max_entries=200000, stacks='python')
                    _mem_history_enabled = True
                    print("[mem snapshot] allocation history recording ENABLED", flush=True)
                except Exception as _exc:
                    print(f"[mem snapshot] could not enable: {_exc}", flush=True)

            def _log_top_allocations(tag: str, top_n: int = 20):
                """Inspect current alive allocations: sum sizes by allocation
                site (last frame of traceback) and print the top-N largest
                source-line buckets. Reveals which code paths are holding
                the leaked memory."""
                if not _mem_history_enabled:
                    return
                try:
                    snap = torch.cuda.memory._snapshot()
                except Exception as _exc:
                    print(f"[mem-top {tag}] snapshot failed: {_exc}", flush=True)
                    return
                from collections import defaultdict
                by_site: "defaultdict[str, int]" = defaultdict(int)
                by_site_count: "defaultdict[str, int]" = defaultdict(int)
                total_active = 0
                for seg in snap.get("segments", []):
                    for blk in seg.get("blocks", []):
                        if blk.get("state") != "active_allocated":
                            continue
                        sz = int(blk.get("size", 0))
                        total_active += sz
                        frames = blk.get("frames", [])
                        # Pick the deepest non-torch frame for clarity.
                        site = "unknown"
                        for fr in frames:
                            fn = str(fr.get("filename", ""))
                            if ("/torch/" in fn) or ("/site-packages/torch" in fn):
                                continue
                            site = f"{fn.split('/')[-1]}:{fr.get('line', '?')}"
                            break
                        else:
                            if frames:
                                fr = frames[0]
                                site = f"{str(fr.get('filename','?')).split('/')[-1]}:{fr.get('line','?')}"
                        by_site[site] += sz
                        by_site_count[site] += 1
                top = sorted(by_site.items(), key=lambda x: -x[1])[:top_n]
                print(f"[mem-top {tag}] total_active={total_active/1024**3:.2f}G top {len(top)} sites:", flush=True)
                for site, sz in top:
                    n = by_site_count[site]
                    print(f"  {sz/1024**2:8.1f}MB  ({n:5d} blocks)  {site}", flush=True)

            # ── Per-task GENERATION exact-match evaluator (PHASE2_SELECT_BY_GEN) ──
            # Runs model.generate on a fixed held-out slice per CoT task each epoch and
            # scores exact-match. Catches generation collapse that teacher-forced CE
            # misses. Guarded: any failure returns None so the caller falls back to CE.
            import re as _re_gen
            def _gen_norm(s):
                _n = _re_gen.sub(r"\s+", " ", _re_gen.sub(r"[^\w\s]", " ", (s or "").lower())).strip()
                # drop leaked special tokens (e.g. gold 'shifting seasonal<eos>' -> 'shifting seasonal')
                return " ".join(w for w in _n.split() if w not in ("eos", "bos", "pad", "unk", "ts")).strip()
            def _gen_suffix(t):
                m = _re_gen.search(r"answer\s*[:\-]\s*([^\n.<]+)", t or "", _re_gen.I)
                if m:
                    return m.group(1).strip().rstrip(".").strip()
                lines = [l.strip() for l in (t or "").splitlines() if l.strip()]
                return lines[-1].rstrip(".").strip() if lines and len(lines[-1]) <= 40 else ""
            def _gen_extract(t, cands):
                suf = _gen_suffix(t); ns = _gen_norm(suf)
                if suf and cands:
                    for c in cands:
                        if _gen_norm(c) == ns:
                            return c
                    for c in cands:
                        if _gen_norm(c) in ns or ns in _gen_norm(c):
                            return c
                    return suf
                if suf:
                    return suf
                nt = _gen_norm(t)
                for c in cands:
                    if _gen_norm(c) in nt:
                        return c
                return ""
            _GEN_TYPES = {"har": "har_cot", "sleep": "sleep_cot", "ecg": "ecg_cot", "pamap2": "pamap2_cot", "tsqa": "tsqa_cot"}
            def _gen_em(max_new_tokens=200, n_per_task=int(os.environ.get("PHASE2_GEN_EM_N", "48"))):
                try:
                    base = val_loader.dataset
                    if hasattr(base, "datasets") and base.datasets:
                        base = base.datasets[0]
                    if not hasattr(base, "samples"):
                        return None
                    # Build the fixed cache once (deterministic order → identical on all ranks).
                    if not _gen_em_cache:
                        for task, ttype in _GEN_TYPES.items():
                            idxs = [i for i, s in enumerate(base.samples)
                                    if isinstance(s, dict) and s.get("type") == ttype][:n_per_task]
                            _gen_em_cache[task] = idxs
                    # Label candidate lists (lazy import).
                    from opentslm.time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset
                    from opentslm.time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset
                    from opentslm.time_series_datasets.ecg_qa.ECGQACoTQADataset import ECGQACoTQADataset
                    _CANDS = {"har": HARCoTQADataset.get_labels(),
                              "sleep": SleepEDFCoTQADataset.get_labels(),
                              "ecg": ECGQACoTQADataset.get_labels()}
                    try:
                        from opentslm.time_series_datasets.pamap2.PAMAP2CoTQADataset import PAMAP2CoTQADataset
                        _CANDS["pamap2"] = PAMAP2CoTQADataset.get_labels()
                    except Exception:
                        _CANDS["pamap2"] = []
                    try:
                        from opentslm.time_series_datasets.tsqa.TSQAClsQADataset import TSQAClsQADataset
                        _CANDS["tsqa"] = TSQAClsQADataset.get_labels()
                    except Exception:
                        _CANDS["tsqa"] = []
                    model_ref = self._get_model(); model_ref.eval()
                    out = {}
                    for task, idxs in _gen_em_cache.items():
                        if not idxs:
                            out[task] = float("nan"); continue
                        shard = idxs[self.rank::max(1, self.world_size)]
                        correct = total = 0
                        with torch.no_grad():
                            for i in shard:
                                row = base[i]
                                gold = _gen_suffix(row.get("answer", "")) or row.get("label", "")
                                try:
                                    # Collate exactly as the DataLoader would before
                                    # generating. Passing a RAW dataset row here is a
                                    # bug: OpenTSLMSPFixPos2D tolerates it (it calls
                                    # torch.as_tensor internally) but the official SP
                                    # and Flamingo classes index `.dim()` / `.shape`
                                    # on the series and raise. The bare `except` below
                                    # then scored those tasks 0.0 silently, forever --
                                    # which drove per-task patience to the limit and
                                    # early-stopped the official baselines at epoch 5
                                    # while their val loss was still falling.
                                    # ECG could never self-repair because its rows come
                                    # from a lazy adapter that rebuilds the dict on
                                    # every access. dict(row) because the collate
                                    # mutates the mapping it is handed.
                                    _b = extend_time_series_to_match_patch_size_and_aggregate(
                                        [dict(row)], patch_size=PATCH_SIZE
                                    )
                                    _out = model_ref.generate(_b, max_new_tokens=max_new_tokens)
                                    texts = _out[0] if isinstance(_out, tuple) else _out
                                    pred = texts[0] if texts else ""
                                except Exception as _ge:
                                    # Never fail silently again: report the first
                                    # failure per task so a structurally-zero column
                                    # is visible in the log instead of looking like a
                                    # model result.
                                    if self.rank == 0 and task not in _gen_em_warned:
                                        _gen_em_warned.add(task)
                                        tqdm.write(
                                            f"⚠️  [gen-EM] generate failed on task '{task}' "
                                            f"({type(_ge).__name__}: {_ge}) — this task will "
                                            f"score 0.0 and drive its early-stop patience."
                                        )
                                    pred = ""
                                pl = _gen_extract(pred, _CANDS[task])
                                total += 1
                                correct += int(_gen_norm(pl) == _gen_norm(gold) and bool(gold))
                        if dist.is_initialized():
                            ct = torch.tensor([correct, total], dtype=torch.float64, device=self.device)
                            dist.all_reduce(ct, op=dist.ReduceOp.SUM)
                            correct, total = ct[0].item(), ct[1].item()
                        out[task] = (correct / total) if total > 0 else float("nan")
                    return out
                except Exception as _e:
                    if self.rank == 0:
                        tqdm.write(f"[gen-EM] failed ({type(_e).__name__}: {_e}); falling back to CE")
                    return None

            # Count of micro-batches skipped because their loss was non-finite
            # (see the guard before backward — protects the model from nan wipeout).
            _nonfinite_skips = 0

            for epoch in range(start_epoch, num_epochs + 1):
                torch.cuda.empty_cache()
                gc.collect()
                _log_gpu_mem(f"epoch{epoch} start (post empty_cache+gc)")
                if epoch >= 2:
                    _log_top_allocations(f"epoch{epoch} start")
                # Set epoch for distributed sampler
                epoch_sampler = None
                if hasattr(train_loader, "batch_sampler") and hasattr(
                    train_loader.batch_sampler, "set_epoch"
                ):
                    epoch_sampler = train_loader.batch_sampler
                elif hasattr(train_loader, "sampler") and hasattr(
                    train_loader.sampler, "set_epoch"
                ):
                    epoch_sampler = train_loader.sampler
                if epoch_sampler is not None:
                    epoch_sampler.set_epoch(epoch)

                # Training
                self.model.train()

                # Reset per-epoch improvement flags (early stopping is epoch-level).
                improved_this_epoch = False
                mse_improved_this_epoch = False
                ce_improved_this_epoch = False
                running_loss = 0.0
                num_train_steps = 0
                prog = tqdm(
                    train_loader,
                    desc=f"Epoch {epoch}/{num_epochs}",
                    disable=self.rank != 0,
                )
                accum_step = 0
                optimizer_step = 0
                # Per-task running accumulators for wandb logging.
                # Each accumulator is reset every train_log_every optimizer steps
                # so logged values reflect the recent window, not the full epoch.
                # `mse_chronos_uv` / `mse_synthetic` capture per-source breakdowns
                # for the chronos task: filled by an extra no-grad forward on
                # source-filtered sub-batches (see the per-source block below).
                _acc = {"total": 0.0, "ce": 0.0, "tsqa": 0.0, "m4": 0.0, "mse": 0.0,
                        "ntp": 0.0, "recon": 0.0, "next_patch": 0.0, "final_patch": 0.0,
                        "mse_chronos_uv": 0.0, "mse_synthetic": 0.0}
                _acc_n = {"total": 0, "ce": 0, "tsqa": 0, "m4": 0, "mse": 0,
                          "ntp": 0, "recon": 0, "next_patch": 0, "final_patch": 0,
                          "mse_chronos_uv": 0, "mse_synthetic": 0}

                # B4: single-worker thread that owns _acc / _acc_n so the main
                # training loop doesn't wait on `reduced_tensor.tolist()` and
                # the periodic wandb.log() call. Single worker → serialised
                # writes to the dicts (no lock needed). The all_reduce stays
                # on the main thread to keep the cross-rank collective in
                # the same call site for every rank.
                _log_executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix=f"log-{stage_name}-r{self.rank}",
                )
                _log_futures: List[Any] = []
                _log_failed: List[BaseException] = []
                # Cascade guard: if too many consecutive forward OOMs hit, the GPU
                # is wedged (allocator can't recover) and continuing produces zero
                # training while burning the wall clock. Crash hard so SLURM can
                # tear the job down and we resubmit with smaller batches.
                _consecutive_fwd_oom = 0
                _MAX_CONSECUTIVE_FWD_OOM = 50

                def _drain_log_futures():
                    # Reap completed futures so a thread-side exception bubbles up.
                    nonlocal _log_futures
                    still_pending = []
                    for f in _log_futures:
                        if not f.done():
                            still_pending.append(f)
                            continue
                        exc = f.exception()
                        if exc is not None:
                            _log_failed.append(exc)
                    _log_futures = still_pending

                _smoke_max_steps = int(os.environ.get("PHASE2_MAX_STEPS_PER_EPOCH", "0"))
                for i, batch in enumerate(prog):
                    if _smoke_max_steps > 0 and i >= _smoke_max_steps:
                        break   # smoke-test cap: end the epoch early to reach validation fast
                    if _should_skip_empty_batch(batch, "train", i):
                        continue
                    if i > 0 and i % 50 == 0:
                        _log_gpu_mem(f"epoch{epoch} train step={i}")
                    if epoch >= 2 and i > 0 and i % 200 == 0:
                        _log_top_allocations(f"epoch{epoch} step={i}")
                    # print('batch: ', batch[0].keys())
                    # print('batch["time_series"]: ', batch[0]["time_series"].shape)
                    accum_step += 1
                    is_accum_first = (accum_step - 1) % grad_accum == 0
                    is_accum_last  = (accum_step % grad_accum == 0) or (i == len(train_loader) - 1)

                    if epoch == start_epoch and i == 0 and self.rank == 0 and self.args.verbose:
                        print(f"[DEBUG] Batch {i} - batch size: {len(batch)}")
                        if isinstance(batch, list) and isinstance(batch[0], dict):
                            for k, v in batch[0].items():
                                if hasattr(v, "shape"):
                                    print(f"[DEBUG] Sample key '{k}' shape: {v.shape}")
                                elif isinstance(v, list):
                                    print(
                                        f"[DEBUG] Sample key '{k}' list length: {len(v)}"
                                    )

                        print(
                            torch.cuda.memory_summary()
                            if torch.cuda.is_available()
                            else "No CUDA"
                        )
                    if is_accum_first:
                        optimizer.zero_grad()

                    next_patch_mse = None
                    final_patch_mse = None
                    with torch.amp.autocast('cuda', dtype=amp_dtype):
                        if self._uses_pretrain_loss(stage_name):
                            if stage_name == "phase1_chronos_seq2seq":
                                loss, ntp_loss, recon_loss = self._get_model().compute_chronos_seq2seq_loss(batch)
                            elif stage_name == "phase1_chronos_seq2seq_separate_mse":
                                loss, ntp_loss, recon_loss = self._get_model().compute_chronos_seq2seq_enc_mse_loss(batch)
                            else:
                                loss, ntp_loss, recon_loss = self._get_model().compute_pretrain_loss(
                                    batch, mse_only=stage_name in PHASE1_MSE_ONLY_STAGES
                                )
                            if stage_name in PHASE1_MSE_ONLY_STAGES:
                                model_ref = self._get_model()
                                next_patch_mse = getattr(model_ref, "_last_pretrain_next_patch_mse", None)
                                final_patch_mse = getattr(model_ref, "_last_pretrain_final_patch_mse", None)
                        else:
                            if stage_name in ("stage2_all"):
                                loss, loss_ce, loss_mse = self._get_model().compute_phase2_loss(batch, pred_len=PRED_LEN, patch_size=PATCH_SIZE)
                            elif stage_name in JOINT_PHASE1_STAGES:
                                _loss_fn = (
                                    self._get_model().compute_joint_phase1_enc_mse_loss
                                    if stage_name == "joint_phase1_enc_mse"
                                    else self._get_model().compute_joint_phase1_loss
                                )
                                loss, loss_ce, loss_tsqa, loss_m4, loss_mse = _loss_fn(
                                    batch,
                                    alpha=getattr(self.args, "mse_alpha", 1.0),
                                    dynamic_weighting=getattr(self.args, "dynamic_loss_weighting", False),
                                    detach_llm_for_mse=(stage_name in JOINT_PHASE1_DETACH_LLM_MSE_STAGES or getattr(self.args, "detach_llm_for_mse", False)),
                                )
                            elif stage_name in JOINT_PHASE2_STAGES:
                                _fwd_oom = False
                                try:
                                    loss, loss_ce, loss_mse = self._get_model().compute_joint_phase2_loss(
                                        batch,
                                        alpha=getattr(self.args, "mse_alpha", 1.0),
                                        dynamic_weighting=getattr(self.args, "dynamic_loss_weighting", False),
                                    )
                                except torch.cuda.OutOfMemoryError:
                                    torch.cuda.empty_cache()
                                    optimizer.zero_grad()
                                    _fwd_oom = True
                                # Broadcast skip decision so all ranks agree —
                                # asymmetric OOM (one rank OOMs, another doesn't)
                                # at an is_accum_last boundary causes the non-OOM
                                # ranks to block inside DDP's all_reduce while the
                                # OOM rank has already `continue`d, deadlocking.
                                if dist.is_initialized():
                                    _oom_t = torch.tensor([int(_fwd_oom)], device="cuda")
                                    dist.all_reduce(_oom_t, op=dist.ReduceOp.MAX)
                                    _fwd_oom = bool(_oom_t.item())
                                if _fwd_oom:
                                    _consecutive_fwd_oom += 1
                                    if _consecutive_fwd_oom >= _MAX_CONSECUTIVE_FWD_OOM:
                                        raise RuntimeError(
                                            f"Forward-OOM cascade: {_consecutive_fwd_oom} consecutive "
                                            f"forward OOMs at step {i+1}; aborting to prevent silent "
                                            f"stall. Reduce --max_tokens_per_batch or batch_size."
                                        )
                                    if self.rank == 0:
                                        tqdm.write(f"⚠️  Forward OOM at step {i+1} (joint_phase2), skipping batch")
                                    continue
                                _consecutive_fwd_oom = 0
                            else:
                                loss = self._get_model().compute_loss(batch)
                                loss_ce, loss_mse = None, None

                    # Per-source chronos MSE (train) — mirror the val-loop split so
                    # wandb shows `train/mse_chronos_uv` vs `train/mse_synthetic`
                    # separately. Re-runs the loss on source-filtered sub-batches
                    # under no_grad so the main backward path is untouched. Cost:
                    # ~2× forward per chronos-bearing batch on rank 0 only.
                    if self.rank == 0 and (
                        stage_name in PHASE1_MSE_ONLY_STAGES
                        or stage_name in JOINT_PHASE1_STAGES
                    ):
                        if stage_name in PHASE1_MSE_ONLY_STAGES:
                            _chr_uv_sub  = [s for s in batch if s.get("source") == "chronos_uv"]
                            _chr_syn_sub = [s for s in batch if s.get("source") == "synthetic"]
                            if _chr_uv_sub or _chr_syn_sub:
                                with torch.no_grad(), torch.amp.autocast('cuda', dtype=amp_dtype):
                                    if _chr_uv_sub:
                                        try:
                                            _, _, _rec_uv = self._get_model().compute_pretrain_loss(
                                                _chr_uv_sub, mse_only=True,
                                            )
                                            _v = float(_rec_uv.item())
                                            if math.isfinite(_v) and _v > 0:
                                                _acc["mse_chronos_uv"] += _v
                                                _acc_n["mse_chronos_uv"] += 1
                                        except Exception as _exc:
                                            if self.args.verbose:
                                                print(f"[per-source train MSE] uv probe failed: {_exc}")
                                    if _chr_syn_sub:
                                        try:
                                            _, _, _rec_syn = self._get_model().compute_pretrain_loss(
                                                _chr_syn_sub, mse_only=True,
                                            )
                                            _v = float(_rec_syn.item())
                                            if math.isfinite(_v) and _v > 0:
                                                _acc["mse_synthetic"] += _v
                                                _acc_n["mse_synthetic"] += 1
                                        except Exception as _exc:
                                            if self.args.verbose:
                                                print(f"[per-source train MSE] syn probe failed: {_exc}")
                        elif stage_name in JOINT_PHASE1_STAGES:
                            _chr_uv_sub  = [s for s in batch if s.get("type")=="chronos" and s.get("source")=="chronos_uv"]
                            _chr_syn_sub = [s for s in batch if s.get("type")=="chronos" and s.get("source")=="synthetic"]
                            if _chr_uv_sub or _chr_syn_sub:
                                with torch.no_grad(), torch.amp.autocast('cuda', dtype=amp_dtype):
                                    if _chr_uv_sub:
                                        try:
                                            _, _, _, _, _mse_uv = _loss_fn(
                                                _chr_uv_sub,
                                                alpha=getattr(self.args, "mse_alpha", 1.0),
                                                dynamic_weighting=getattr(self.args, "dynamic_loss_weighting", False),
                                                detach_llm_for_mse=(stage_name in JOINT_PHASE1_DETACH_LLM_MSE_STAGES or getattr(self.args, "detach_llm_for_mse", False)),
                                            )
                                            _v = float(_mse_uv.item())
                                            if math.isfinite(_v) and _v > 0:
                                                _acc["mse_chronos_uv"] += _v
                                                _acc_n["mse_chronos_uv"] += 1
                                        except Exception as _exc:
                                            if self.args.verbose:
                                                print(f"[per-source train MSE] uv probe failed: {_exc}")
                                    if _chr_syn_sub:
                                        try:
                                            _, _, _, _, _mse_syn = _loss_fn(
                                                _chr_syn_sub,
                                                alpha=getattr(self.args, "mse_alpha", 1.0),
                                                dynamic_weighting=getattr(self.args, "dynamic_loss_weighting", False),
                                                detach_llm_for_mse=(stage_name in JOINT_PHASE1_DETACH_LLM_MSE_STAGES or getattr(self.args, "detach_llm_for_mse", False)),
                                            )
                                            _v = float(_mse_syn.item())
                                            if math.isfinite(_v) and _v > 0:
                                                _acc["mse_synthetic"] += _v
                                                _acc_n["mse_synthetic"] += 1
                                        except Exception as _exc:
                                            if self.args.verbose:
                                                print(f"[per-source train MSE] syn probe failed: {_exc}")

                    # Guard: one non-finite sample loss backpropagates nan into every
                    # parameter and permanently destroys the model — and now that
                    # gradients are correctly averaged it destroys ALL ranks, not just
                    # one replica (observed: gen-EM collapsing to 0.000 on every task).
                    # Skip the backward for such micro-batches instead.
                    # Rank-safe: gradients are synchronised explicitly in
                    # _sync_gradients(), whose all-reduced has_grad handshake tolerates
                    # ranks disagreeing about which params received gradients, so a rank
                    # skipping a backward cannot desynchronise the collectives.
                    # Cross-rank agreement BEFORE skipping. A per-rank `continue`
                    # here is a deadlock: the rest of this loop body contains
                    # collectives (the metrics all_reduce under `is_accum_last`,
                    # and _sync_gradients), so a single rank bailing out leaves
                    # the others blocked in an ALLREDUCE forever. That is exactly
                    # how job 3455465 wedged for ~3 days and how 3458313 died --
                    # watchdog trace: "WorkNCCL(SeqNum=38782, OpType=ALLREDUCE,
                    # NumelIn=3) ran for 600029 ms", i.e. the 3-element metrics
                    # tensor below. The OOM path a few lines up already does this
                    # handshake; this one did not. Either every rank skips the
                    # batch, or none does.
                    _bad = torch.tensor(
                        [0 if torch.isfinite(loss) else 1],
                        device=self.device, dtype=torch.int32,
                    )
                    if dist.is_initialized():
                        dist.all_reduce(_bad, op=dist.ReduceOp.MAX)
                    if bool(_bad.item()):
                        _nonfinite_skips += 1
                        if self.rank == 0 and _nonfinite_skips <= 20:
                            tqdm.write(
                                f"⚠️  non-finite loss on >=1 rank, epoch {epoch} "
                                f"batch {i} — ALL ranks skipping backward; types="
                                f"{sorted({s.get('type', '?') for s in batch})} "
                                f"(skipped so far: {_nonfinite_skips})"
                            )
                        del loss
                        continue

                    _sync_ctx = (
                        self.model.no_sync()
                        if hasattr(self.model, "no_sync") and not is_accum_last
                        else contextlib.nullcontext()
                    )
                    try:
                        with _sync_ctx:
                            scaler.scale(loss / grad_accum).backward()
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        optimizer.zero_grad()
                        # Backward OOM under DDP is unrecoverable: gradient all-reduces
                        # were partially queued via autograd hooks before the OOM, so
                        # other ranks will hang forever on unmatched collectives if we
                        # `continue`. Fail fast on every rank so SLURM tears the job
                        # down cleanly instead of deadlocking for hours.
                        print(
                            f"[rank{self.rank}] Backward OOM at step {i+1}; "
                            f"DDP collectives are now in an unrecoverable state. "
                            f"Aborting job to avoid NCCL deadlock.",
                            flush=True,
                        )
                        raise

                    running_loss += loss.item()
                    num_train_steps += 1
                    if self.rank == 0:
                        prog.set_postfix(
                            loss=f"{loss.item():.4f}",
                            lr=f"{scheduler.get_last_lr()[0]:.2e}",
                        )
                        # Accumulate per-task losses every sub-batch so that
                        # tsqa/m4/mse are logged even when not the last
                        # gradient-accumulation sub-batch.
                        if stage_name in JOINT_PHASE1_STAGES:
                            _v = loss_tsqa.item()
                            if _v > 0:
                                _acc["tsqa"] += _v; _acc_n["tsqa"] += 1
                            if stage_name in JOINT_PHASE1_WITH_M4_STAGES:
                                _v = loss_m4.item()
                                if _v > 0:
                                    _acc["m4"] += _v; _acc_n["m4"] += 1
                            _v = loss_mse.item()
                            if _v > 0:
                                _acc["mse"] += _v; _acc_n["mse"] += 1

                    if is_accum_last:
                        optimizer_step += 1
                        global_optimizer_step += 1

                        reduced_metrics = [loss.detach().float()]
                        if self._uses_pretrain_loss(stage_name):
                            reduced_metrics.extend(
                                [ntp_loss.detach().float(), recon_loss.detach().float()]
                            )
                            if stage_name in PHASE1_MSE_ONLY_STAGES:
                                zero = loss.detach().new_zeros(())
                                reduced_metrics.extend(
                                    [
                                        (final_patch_mse.detach().float() if final_patch_mse is not None else zero),
                                    ]
                                )
                        elif stage_name in ("stage2_all",):
                            reduced_metrics.extend(
                                [loss_ce.detach().float(), loss_mse.detach().float()]
                            )
                        elif stage_name in JOINT_PHASE1_STAGES:
                            reduced_metrics.extend(
                                [
                                    loss_ce.detach().float(),
                                    loss_tsqa.detach().float(),
                                    loss_m4.detach().float(),
                                    loss_mse.detach().float(),
                                ]
                            )
                        elif stage_name in JOINT_PHASE2_STAGES:
                            reduced_metrics.extend(
                                [loss_ce.detach().float(), loss_mse.detach().float()]
                            )

                        reduced_tensor = torch.stack(reduced_metrics)
                        # phase1_chronos_mse already synchronizes gradients through
                        # DDP during backward; the extra metric all-reduce here is
                        # only for logging. On multi-node runs this stage has been
                        # stalling immediately after the first batch, so keep the
                        # hot path free of additional collectives and log rank-0
                        # local metrics instead. Other stages retain exact
                        # cross-rank metric averaging.
                        should_reduce_metrics = (
                            dist.is_initialized()
                            and stage_name not in PHASE1_MSE_ONLY_STAGES
                        )
                        if should_reduce_metrics:
                            # all_reduce stays on the main thread: it's a
                            # cross-rank collective and every rank must reach
                            # it in the same call site.
                            dist.all_reduce(reduced_tensor, op=dist.ReduceOp.SUM)
                            reduced_tensor /= self.world_size

                        # B4: hand the reduced tensor to the logger thread.
                        # Detach + clone so the main thread can free / overwrite
                        # `reduced_tensor` immediately. The thread does the
                        # `.tolist()` sync, the per-task accumulation into _acc,
                        # and the periodic wandb.log() — none of which the
                        # optimiser step depends on.
                        _snapshot_metrics = reduced_tensor.detach().clone()
                        _is_rank0 = (self.rank == 0)
                        _opt_step_for_log = optimizer_step
                        _global_step_for_log = global_optimizer_step
                        _stage_name_for_log = stage_name

                        def _do_log(snap=_snapshot_metrics,
                                    opt_step=_opt_step_for_log,
                                    global_step=_global_step_for_log,
                                    stage=_stage_name_for_log,
                                    is_rank0=_is_rank0,
                                    _acc=_acc,
                                    _acc_n=_acc_n):
                            reduced_values = snap.tolist()

                            avg_loss = reduced_values[0]
                            avg_ntp = avg_recon = avg_next_patch = avg_final_patch = None
                            avg_ce = avg_tsqa = avg_m4 = avg_mse = None
                            if self._uses_pretrain_loss(stage):
                                if stage in PHASE1_MSE_ONLY_STAGES:
                                    avg_ntp, avg_recon, avg_final_patch = reduced_values[1:]
                                    avg_next_patch = avg_recon
                                else:
                                    avg_ntp, avg_recon = reduced_values[1:]
                            elif stage in ("stage2_all",):
                                avg_ce, avg_mse = reduced_values[1:]
                            elif stage in JOINT_PHASE1_STAGES:
                                avg_ce, avg_tsqa, avg_m4, avg_mse = reduced_values[1:]
                            elif stage in JOINT_PHASE2_STAGES:
                                avg_ce, avg_mse = reduced_values[1:]

                            if is_rank0:
                                _acc["total"] += avg_loss; _acc_n["total"] += 1
                                if stage in PHASE1_MSE_ONLY_STAGES:
                                    _acc["ntp"]   += avg_ntp;   _acc_n["ntp"]   += 1
                                    _acc["recon"] += avg_recon; _acc_n["recon"] += 1
                                    _acc["next_patch"] += avg_next_patch; _acc_n["next_patch"] += 1
                                    _acc["final_patch"] += avg_final_patch; _acc_n["final_patch"] += 1
                                elif self._uses_pretrain_loss(stage):
                                    _acc["ntp"]   += avg_ntp;   _acc_n["ntp"]   += 1
                                    _acc["recon"] += avg_recon; _acc_n["recon"] += 1
                                elif stage in ("stage2_all",) + tuple(JOINT_PHASE1_STAGES) + tuple(JOINT_PHASE2_STAGES):
                                    _acc["ce"] += avg_ce; _acc_n["ce"] += 1
                                    if stage not in JOINT_PHASE1_STAGES:
                                        if avg_mse > 0:
                                            _acc["mse"] += avg_mse; _acc_n["mse"] += 1

                            if is_rank0 and opt_step % train_log_every == 0:
                                def _wavg(key):
                                    n = _acc_n[key]
                                    return _acc[key] / n if n > 0 else None

                                if self._uses_pretrain_loss(stage):
                                    if stage in PHASE1_MSE_ONLY_STAGES:
                                        log_dict = {
                                            "pretrain/total_loss": _wavg("total"),
                                            "pretrain/next_patch_mse": _wavg("next_patch"),
                                            "pretrain/final_patch_mse": _wavg("final_patch"),
                                            "pretrain/mse_chronos_uv": _wavg("mse_chronos_uv"),
                                            "pretrain/mse_synthetic":  _wavg("mse_synthetic"),
                                        }
                                    else:
                                        log_dict = {
                                            "pretrain/total_loss": _wavg("total"),
                                            "pretrain/ntp_loss":   _wavg("ntp"),
                                            "pretrain/recon_loss": _wavg("recon"),
                                            "pretrain/mse_chronos_uv": _wavg("mse_chronos_uv"),
                                            "pretrain/mse_synthetic":  _wavg("mse_synthetic"),
                                        }
                                elif stage == "stage2_all":
                                    log_dict = {
                                        "train/loss":     _wavg("total"),
                                        "train/ce_loss":  _wavg("ce"),
                                        "train/mse_loss": _wavg("mse"),
                                    }
                                elif stage in JOINT_PHASE1_STAGES:
                                    log_dict = {
                                        "train/loss":      _wavg("total"),
                                        "train/ce_loss":   _wavg("ce"),
                                        "train/tsqa_loss": _wavg("tsqa"),
                                        "train/mse_loss":  _wavg("mse"),
                                        "train/mse_chronos_uv": _wavg("mse_chronos_uv"),
                                        "train/mse_synthetic":  _wavg("mse_synthetic"),
                                    }
                                    if stage in JOINT_PHASE1_WITH_M4_STAGES:
                                        log_dict["train/m4_loss"] = _wavg("m4")
                                elif stage in JOINT_PHASE2_STAGES:
                                    log_dict = {
                                        "train/loss":     _wavg("total"),
                                        "train/ce_loss":  _wavg("ce"),
                                        "train/mse_loss": _wavg("mse"),
                                    }
                                else:
                                    log_dict = {"train/loss": _wavg("total")}

                                wandb.log(
                                    {k: v for k, v in log_dict.items() if v is not None},
                                    step=global_step,
                                )
                                for k in _acc:
                                    _acc[k] = 0.0; _acc_n[k] = 0

                        _log_futures.append(_log_executor.submit(_do_log))
                        # Periodic reap so a thread-side exception fails fast.
                        if optimizer_step % 64 == 0:
                            _drain_log_futures()
                            if _log_failed:
                                raise _log_failed[0]

                        # Handle gradient clipping for distributed training
                        scaler.unscale_(optimizer)
                        # DDP.forward is bypassed by this trainer, so its reducer
                        # never fires — average gradients explicitly before clipping.
                        self._sync_gradients()
                        clip_grad_norm_(self._get_model().parameters(), GRAD_CLIP_NORM)

                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()

                        # Periodic mid-epoch checkpoint (skip last batch — end-of-epoch save handles it)
                        save_every = getattr(self.args, "save_every_n_steps", 0)
                        is_last_batch = (i == len(train_loader) - 1)
                        if save_every > 0 and (i + 1) % save_every == 0 and not is_last_batch:
                            self._save_periodic_checkpoint(stage_dir, epoch, i + 1, optimizer, scheduler)

                        # Memory-leak workaround: ~25 MB/step accumulates in
                        # Gemma3 layer outputs (modeling_gemma3.py:400) starting
                        # in epoch 2; reaches 32 GB by step ~1180 and OOMs.
                        # When alloc approaches the wall, save ckpt and exit 42
                        # cooperatively so the SLURM wrapper restarts python
                        # (fresh process = leak resets).
                        if torch.cuda.is_available():
                            _alloc_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                            # Threshold is a FRACTION of the card, not the literal
                            # 22 GB this used to hardcode. 0.70 * 31.73 GB = 22.2 GB,
                            # so V100 behaviour is unchanged to within 1%, but a
                            # 96 GB card now restarts near ~66 GB instead of 22 GB.
                            # That matters because a restart is not free: resume
                            # granularity is one EPOCH (periodic_checkpoint.pt is
                            # written at :1026 and deleted at :871 but never read),
                            # so every mid-epoch restart discards the partial epoch.
                            # The leak scales with hidden size x depth, so a 1B
                            # backbone on a big card would otherwise thrash.
                            _total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                            _mem_frac = float(os.environ.get("MEM_RESTART_FRAC", "0.70"))
                            _restart_t = torch.tensor(
                                [1.0 if _alloc_gb > _mem_frac * _total_gb else 0.0], device="cuda")
                            if dist.is_initialized():
                                dist.all_reduce(_restart_t, op=dist.ReduceOp.MAX)
                            if _restart_t.item() > 0:
                                if self.rank == 0:
                                    tqdm.write(
                                        f"[mem-restart] alloc={_alloc_gb:.2f}G > "
                                        f"{_mem_frac:.0%} of {_total_gb:.1f}G; "
                                        f"saving ckpt + exit code 42 for SLURM restart"
                                    )
                                # Rank 0 saves; others no-op inside the func.
                                self._save_periodic_checkpoint(stage_dir, epoch, i + 1, optimizer, scheduler)
                                if dist.is_initialized():
                                    dist.barrier()  # all ranks exit together
                                sys.exit(42)

                        # Periodic mid-epoch validation every --eval_every_n_iters
                        # steps — capped to --eval_max_samples_per_task for speed.
                        if global_optimizer_step % eval_every_n_iters == 0:
                            _run_validation(
                                epoch, global_optimizer_step,
                                running_loss / max(1, num_train_steps),
                                max_samples_per_task=eval_max_samples_per_task,
                            )
                            last_eval_step = global_optimizer_step
                            self.model.train()

                # B4: drain pending log tasks at the end of the train epoch so
                # all wandb.log calls land before the val block opens, and so a
                # thread-side exception fails the run instead of silently dropping.
                _log_executor.shutdown(wait=True)
                _drain_log_futures()
                if _log_failed:
                    raise _log_failed[0]

                avg_train_loss = running_loss / max(1, num_train_steps)
                if self.rank == 0:
                    tqdm.write(f"Epoch {epoch} — train loss: {avg_train_loss:.4f}")

                # Tail validation: evaluate the epoch's final state if any
                # optimizer steps elapsed since the last periodic eval. Runs the
                # FULL val set (no per-task cap) for an accurate epoch metric.
                if global_optimizer_step > last_eval_step:
                    _run_validation(epoch, global_optimizer_step, avg_train_loss)
                    last_eval_step = global_optimizer_step
                    self.model.train()

                # ── Early stopping (epoch-level; uses periodic-eval flags) ──
                is_joint = stage_name in JOINT_CE_MSE_STAGES
                if improved_this_epoch:
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if self.rank == 0:
                        tqdm.write(f"No improvement for {epochs_no_improve}/{EARLY_STOP_PAT} epochs.\n")

                if _pertask_es:
                    # Per-task patience counters. Stop if ANY ACTIVE task stagnates
                    # for EARLY_STOP_PAT epochs. "Active" = a task that has produced a
                    # real metric (best_val_cot finite) — this skips tasks absent from
                    # the mix (e.g. har/sleep/ecg in a pamap2-only run stay inf/nan),
                    # so single-task finetunes don't false-trigger on absent tasks.
                    _active = [_t for _t in epochs_no_improve_cot if best_val_cot.get(_t, float("inf")) != float("inf")]
                    for _t in _active:
                        if cot_improved[_t]:
                            epochs_no_improve_cot[_t] = 0
                        else:
                            epochs_no_improve_cot[_t] += 1
                    if self.rank == 0:
                        tqdm.write("  CoT patience: " + "  ".join(
                            f"{_t}={epochs_no_improve_cot[_t]}/{EARLY_STOP_PAT}"
                            for _t in _active))
                    _cot_stag = any(epochs_no_improve_cot[_t] >= EARLY_STOP_PAT
                                    for _t in _active)
                    if dist.is_initialized():
                        _st = torch.tensor(1 if _cot_stag else 0, device=self.device)
                        dist.all_reduce(_st, op=dist.ReduceOp.SUM)
                        _cot_stag = _st.item() > 0
                    if _cot_stag:
                        if self.rank == 0:
                            _which = [f"{_t} ({epochs_no_improve_cot[_t]} ep)"
                                      for _t in _active
                                      if epochs_no_improve_cot[_t] >= EARLY_STOP_PAT]
                            tqdm.write(f"\nEarly stopping (per-task): {', '.join(_which)} "
                                       f"stagnated for {EARLY_STOP_PAT} epochs.")
                        break

                if is_joint:
                    if mse_improved_this_epoch:
                        epochs_no_improve_mse = 0
                    else:
                        epochs_no_improve_mse += 1
                    if ce_improved_this_epoch:
                        epochs_no_improve_ce = 0
                    else:
                        epochs_no_improve_ce += 1
                    if self.rank == 0:
                        tqdm.write(
                            f"  MSE patience: {epochs_no_improve_mse}/{EARLY_STOP_PAT}  "
                            f"CE patience: {epochs_no_improve_ce}/{EARLY_STOP_PAT}"
                        )
                    task_stagnated = (
                        epochs_no_improve_mse >= EARLY_STOP_PAT or
                        epochs_no_improve_ce  >= EARLY_STOP_PAT
                    )
                    if dist.is_initialized():
                        stag_tensor = torch.tensor(1 if task_stagnated else 0, device=self.device)
                        dist.all_reduce(stag_tensor, op=dist.ReduceOp.SUM)
                        task_stagnated = stag_tensor.item() > 0
                    if task_stagnated:
                        if self.rank == 0:
                            which = []
                            if epochs_no_improve_mse >= EARLY_STOP_PAT:
                                which.append(f"MSE ({epochs_no_improve_mse} epochs)")
                            if epochs_no_improve_ce >= EARLY_STOP_PAT:
                                which.append(f"CE ({epochs_no_improve_ce} epochs)")
                            tqdm.write(f"\nEarly stopping triggered: {', '.join(which)} stagnated.")
                        break

                if not is_joint and not _pertask_es and epochs_no_improve >= EARLY_STOP_PAT:
                    if self.rank == 0:
                        tqdm.write(f"\nEarly stopping triggered after {epoch} epochs.")
                        tqdm.write(f"Final stats: best_val_loss={best_val_loss:.4f}")
                    break

                if dist.is_initialized():
                    best_loss_tensor = torch.tensor(best_val_loss, device=self.device)
                    epochs_tensor = torch.tensor(epochs_no_improve, device=self.device)
                    dist.broadcast(best_loss_tensor, src=0)
                    dist.broadcast(epochs_tensor, src=0)
                    best_val_loss = best_loss_tensor.item()
                    epochs_no_improve = int(epochs_tensor.item())

        # Load best model and evaluate
        gc.collect()
        torch.cuda.empty_cache()
        best_epoch, _ = self._load_checkpoint(stage_dir, optimizer, scheduler)
        if best_epoch is not None:
            if self.rank == 0:
                print(
                    f"📂 Loaded best checkpoint from epoch {best_epoch} for evaluation."
                )

        if self.rank == 0:
            if epoch is None:
                epoch = best_epoch
                print(f"🏁 Training completed for {stage_name}")
                print(f"   Total epochs run: {epoch}")
            else:
                print(f"🏁 Training completed for {stage_name}")
                print(f"   Total epochs run: {epoch}")
                print(f"   Best validation loss: {best_val_loss:.4f}")
                print(f"   Epochs without improvement: {epochs_no_improve}")

        metrics = self._evaluate_stage(
            stage_dir, test_loader, stage_dir, metric_func, best_epoch
        )
        # Expose stage_dir so run_curriculum can pass it to _mark_stage_completed.
        metrics["_stage_dir"] = stage_dir
        # B4: drain pending log tasks and shut down the logger thread so any
        # exceptions raised inside it propagate before we move on.
        try:
            _log_executor.shutdown(wait=True)
            _drain_log_futures()
            if _log_failed:
                raise _log_failed[0]
        except NameError:
            pass  # _log_executor may not have been created on early-exit paths
        if self.rank == 0:
            wandb.finish()
        return metrics

    def stage1_mcq(
        self, batch_size: int = None, val_batch_size: int = None, eval_only: bool = False,
        warm_start_from: str = None,
    ) -> Dict[str, Any]:
        """Stage 1: Multiple Choice Question Answering (TSQA).

        Configuration:
        - Epochs: 20
        - OpenTSLMSP: encoder_lr=2e-4, projector_lr=1e-4
        - OpenTSLMFlamingo: base_lr=2e-4
        - Metric: Accuracy
        """
        return self._train_stage(
            stage_name="stage1_mcq",
            dataset_class=TSQADataset,
            num_epochs=self.args.num_epochs,
            lr_encoder=2e-4,
            lr_projector=1e-4,
            lr_base=2e-4,
            metric_func=lambda preds, golds: {
                "accuracy": self._calculate_accuracy(preds, golds)
            },
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            eval_only=eval_only,
            warm_start_from=warm_start_from,
        )

    def stage2_captioning(
        self, batch_size: int = None, val_batch_size: int = None, eval_only: bool = False,
        warm_start_from: str = None,
    ) -> Dict[str, Any]:
        """Stage 2: Caption Generation (M4).

        Configuration:
        - Epochs: 15
        - OpenTSLMSP: encoder_lr=1e-4, projector_lr=5e-5 (lower for fine-tuning)
        - OpenTSLMFlamingo: base_lr=1e-4 (lower for fine-tuning)
        - Metric: Test loss only
        """
        return self._train_stage(
            stage_name="stage2_captioning",
            dataset_class=M4QADataset,
            num_epochs=self.args.num_epochs,
            lr_encoder=2e-4,
            lr_projector=1e-4,
            lr_base=2e-4,
            metric_func=None,  # Only test loss for captioning
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            eval_only=eval_only,
            warm_start_from=warm_start_from,
        )

    def stage3_cot(
        self, batch_size: int = None, val_batch_size: int = None, eval_only: bool = False,
        warm_start_from: str = None,
    ) -> Dict[str, Any]:
        """Stage CoT: Chain-of-Thought Reasoning (HAR).

        Configuration:
        - Epochs: 100
        - OpenTSLMSP: encoder_lr=2e-4, projector_lr=1e-4
        - OpenTSLMFlamingo: base_lr=2e-4
        - Metric: Test loss only (chain-of-thought reasoning)
        """
        sampler = None

        return self._train_stage(
            stage_name="stage3_cot",
            dataset_class=HARCoTQADataset,
            num_epochs=self.args.num_epochs,
            lr_encoder=2e-4,
            lr_projector=1e-4,
            lr_base=2e-4,
            metric_func=None,  # Only test loss for chain-of-thought reasoning
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            eval_only=eval_only,
            sampler=sampler,
            warm_start_from=warm_start_from,
        )

    def stage4_sleep_cot(
        self, batch_size: int = None, val_batch_size: int = None, eval_only: bool = False,
        warm_start_from: str = None,
    ) -> Dict[str, Any]:
        """Stage 4: Chain-of-Thought Reasoning (SleepEDF).

        Configuration:
        - Epochs: 60
        - OpenTSLMSP: encoder_lr=2e-4, projector_lr=1e-4
        - OpenTSLMFlamingo: base_lr=2e-4
        - Metric: Test loss only (chain-of-thought reasoning)
        """
        sampler = None

        return self._train_stage(
            stage_name="stage4_sleep_cot",
            dataset_class=SleepEDFCoTQADataset,
            num_epochs=self.args.num_epochs,
            lr_encoder=2e-4,
            lr_projector=1e-4,
            lr_base=2e-4,
            metric_func=None,  # Only test loss for chain-of-thought reasoning
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            eval_only=eval_only,
            sampler=sampler,
            warm_start_from=warm_start_from,
        )

    def stage5_ecg_cot(
        self, batch_size: int = None, val_batch_size: int = None, eval_only: bool = False,
        warm_start_from: str = None,
    ) -> Dict[str, Any]:
        """Stage 5: Chain-of-Thought Reasoning (ECG QA CoT).

        Configuration:
        - Epochs: 60
        - OpenTSLMSP: encoder_lr=2e-4, projector_lr=1e-4
        - OpenTSLMFlamingo: base_lr=2e-4
        - Metric: Test loss only (chain-of-thought reasoning)
        """
        sampler = None

        return self._train_stage(
            stage_name="stage5_ecg_cot",
            dataset_class=ECGQACoTQADataset,
            num_epochs=self.args.num_epochs,
            lr_encoder=2e-4,
            lr_projector=1e-4,
            lr_base=2e-4,
            metric_func=None,  # Only test loss for chain-of-thought reasoning
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            eval_only=eval_only,
            sampler=sampler,
            warm_start_from=warm_start_from,
        )


    def joint_phase1(
        self, batch_size: int = None, val_batch_size: int = None, eval_only: bool = False
    ) -> Dict[str, Any]:
        """Joint Phase-1: Simultaneous training on three tasks:
        - Chronos forecasting (pretraining)
        - TSQA multiple-choice QA (stage1_mcq)

        Configuration:
        - OpenTSLMSP: encoder_lr=2e-4, projector_lr=1e-4
        - OpenTSLMFlamingo: base_lr=2e-4
        - Metric: Accuracy (from TSQA MCQ task)
        """
        max_samp = getattr(self.args, "max_samples_per_source", None)
        dataset_cls = (
            partial(Phase1JointDataset, max_samples_per_source=max_samp)
            if max_samp else Phase1JointDataset
        )
        return self._train_stage(
            stage_name="joint_phase1",
            dataset_class=dataset_cls,
            num_epochs=self.args.num_epochs,
            lr_encoder=LR_ENCODER_JOINT,
            lr_projector=1e-4,
            lr_base=2e-4,
            metric_func=lambda preds, golds: {
                "accuracy": self._calculate_accuracy(preds, golds)
            },
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            eval_only=eval_only,
        )

    def joint_phase2(
        self, batch_size: int = None, val_batch_size: int = None, eval_only: bool = False,
        warm_start_from: str = None,
    ) -> Dict[str, Any]:
        """Joint Phase-2: Simultaneous training on four tasks:
        - Chronos forecasting with DUAL loss:
            * CE on answer tokens (<ts> v1 v2 ... </ts>) — teaches LLM to generate forecasts as text
            * MSE via decoder head (next-patch prediction) — preserves numerical forecasting accuracy
        - HAR Chain-of-Thought QA (CE loss)
        - SleepEDF Chain-of-Thought QA (CE loss)
        - ECG Chain-of-Thought QA (CE loss)
        """
        max_samp = getattr(self.args, "max_samples_per_source", None)
        dataset_cls = (
            partial(Phase2JointDataset, max_samples_per_source=max_samp)
            if max_samp else Phase2JointDataset
        )
        return self._train_stage(
            stage_name="joint_phase2",
            dataset_class=dataset_cls,
            num_epochs=self.args.num_epochs,
            lr_encoder=LR_ENCODER_JOINT,   # unused (encoder frozen), kept for consistency
            lr_projector=1e-4,             # unused (projector frozen), kept for consistency
            lr_base=2e-4,
            metric_func=None,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            eval_only=eval_only,
            warm_start_from=warm_start_from,
        )

    def joint_phase2_open(
        self, batch_size: int = None, val_batch_size: int = None, eval_only: bool = False,
        warm_start_from: str = None,
    ) -> Dict[str, Any]:
        """Joint Phase-2 Open: same tasks as joint_phase2 but with encoder and projector
        unfrozen at very small learning rates, allowing the soft-token representations to
        adapt end-to-end to the instruction-tuning objectives.

        Learning rates:
          encoder:          LR_ENCODER_JOINT / 5  (~2e-6) — conservative, avoids forgetting phase-1
          projector:        LR_PROJECTOR / 5       (~2e-5) — conservative
          decoder:          LR_DECODER             (2e-4)
          decoder_projector: LR_PROJECTOR          (1e-4)
          LoRA (LLM):       2e-4
        """
        max_samp = getattr(self.args, "max_samples_per_source", None)
        dataset_cls = (
            partial(Phase2JointDataset, max_samples_per_source=max_samp)
            if max_samp else Phase2JointDataset
        )
        return self._train_stage(
            stage_name="joint_phase2_open",
            dataset_class=dataset_cls,
            num_epochs=self.args.num_epochs,
            lr_encoder=LR_ENCODER_JOINT / 5,   # 2e-6
            lr_projector=LR_PROJECTOR / 5,      # 2e-5
            lr_base=2e-4,
            metric_func=None,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            eval_only=eval_only,
            warm_start_from=warm_start_from,
        )

    def joint_phase2_downstream(
        self, batch_size: int = None, val_batch_size: int = None, eval_only: bool = False,
        warm_start_from: str = None,
    ) -> Dict[str, Any]:
        """Joint Phase-2 Downstream: fine-tuning on two tasks:
        - Downstream forecasting benchmarks (ETTh1/2, ETTm1/2, Weather, Traffic, Electricity)
          with DUAL loss: CE on boundary tokens + MSE via decoder head.
        - ChatTS sft + ift instruction-following (CE loss only).

        Freeze policy: same as joint_phase2 — encoder + projector frozen,
        LoRA on LLM (r=16, alpha=32), decoder + decoder_projector trainable.
        """
        return self._train_stage(
            stage_name="joint_phase2_downstream",
            dataset_class=Phase2DownstreamDataset,
            num_epochs=self.args.num_epochs,
            lr_encoder=LR_ENCODER_JOINT,   # frozen — kept for consistency
            lr_projector=1e-4,             # frozen — kept for consistency
            lr_base=2e-4,
            metric_func=None,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            eval_only=eval_only,
            warm_start_from=warm_start_from,
        )

    def joint_phase1_extend(
        self, batch_size: int = None, val_batch_size: int = None, eval_only: bool = False,
        warm_start_from: str = None,
    ) -> Dict[str, Any]:
        """Joint Phase-1 Extended: Simultaneous training on three tasks:
        - Chronos forecasting (MSE loss)
        - TSQA multiple-choice QA (CE loss)
        - M4 caption generation (CE loss)
        """
        max_samp = getattr(self.args, "max_samples_per_source", None)
        dataset_cls = (
            partial(Phase1JointExtendDataset, max_samples_per_source=max_samp)
            if max_samp else Phase1JointExtendDataset
        )
        return self._train_stage(
            stage_name="joint_phase1_extend",
            dataset_class=dataset_cls,
            num_epochs=self.args.num_epochs,
            lr_encoder=LR_ENCODER_JOINT,
            lr_projector=1e-4,
            lr_base=2e-4,
            metric_func=lambda preds, golds: {
                "accuracy": self._calculate_accuracy(preds, golds)
            },
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            eval_only=eval_only,
            warm_start_from=warm_start_from,
        )

    def joint_phase1_tsqa_m4(
        self, batch_size: int = None, val_batch_size: int = None, eval_only: bool = False,
        warm_start_from: str = None,
    ) -> Dict[str, Any]:
        """Joint Phase-1 TSQA+M4: simultaneous CE-only training on two tasks:
        - TSQA multiple-choice QA
        - M4 caption generation
        """
        max_samp = getattr(self.args, "max_samples_per_source", None)
        dataset_cls = (
            partial(Phase1JointTsqaM4Dataset, max_samples_per_source=max_samp)
            if max_samp else Phase1JointTsqaM4Dataset
        )
        return self._train_stage(
            stage_name="joint_phase1_tsqa_m4",
            dataset_class=dataset_cls,
            num_epochs=self.args.num_epochs,
            lr_encoder=LR_ENCODER_JOINT,
            lr_projector=1e-4,
            lr_base=2e-4,
            metric_func=lambda preds, golds: {
                "accuracy": self._calculate_accuracy(preds, golds)
            },
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            eval_only=eval_only,
            warm_start_from=warm_start_from,
        )

    def joint_phase1_tsqa(
        self, batch_size: int = None, val_batch_size: int = None, eval_only: bool = False,
        warm_start_from: str = None,
    ) -> Dict[str, Any]:
        """Joint Phase-1 TSQA-only: CE training on TSQA multiple-choice QA only
        (no M4 captioning, no Chronos pretraining).
        """
        max_samp = getattr(self.args, "max_samples_per_source", None)
        dataset_cls = (
            partial(Phase1JointTsqaOnlyDataset, max_samples_per_source=max_samp)
            if max_samp else Phase1JointTsqaOnlyDataset
        )
        return self._train_stage(
            stage_name="joint_phase1_tsqa",
            dataset_class=dataset_cls,
            num_epochs=self.args.num_epochs,
            lr_encoder=LR_ENCODER_JOINT,
            lr_projector=1e-4,
            lr_base=2e-4,
            metric_func=lambda preds, golds: {
                "accuracy": self._calculate_accuracy(preds, golds)
            },
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            eval_only=eval_only,
            warm_start_from=warm_start_from,
        )

    def phase1_chronos_seq2seq(
        self, batch_size: int = None, val_batch_size: int = None, eval_only: bool = False,
        warm_start_from: str = None,
    ) -> Dict[str, Any]:
        """Phase-1 seq2seq (Transfusion-inspired): MSE training on Chronos in an
        autoregressive sequence-to-sequence format.

        The LLM receives ``[text | input_patches | text | target_patches]`` causally.
        Forecast supervision is pure next-patch MSE over the target horizon; the
        text scaffold CE is tracked only as an auxiliary metric.
        Gradients flow back through the LLM (unfrozen) to the encoder+projector.
        ``pred_len`` may be a single int or a list — see ``Phase1ChronosSeq2SeqDataset``.
        """
        max_samp = getattr(self.args, "max_samples_per_source", None)
        dataset_cls = (
            partial(Phase1ChronosSeq2SeqDataset, max_samples_per_source=max_samp)
            if max_samp else Phase1ChronosSeq2SeqDataset
        )
        return self._train_stage(
            stage_name="phase1_chronos_seq2seq",
            dataset_class=dataset_cls,
            num_epochs=self.args.num_epochs,
            lr_encoder=LR_ENCODER_JOINT,
            lr_projector=1e-4,
            lr_base=2e-4,
            metric_func=None,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            eval_only=eval_only,
            warm_start_from=warm_start_from,
        )

    def phase1_chronos_seq2seq_separate_mse(
        self, batch_size: int = None, val_batch_size: int = None, eval_only: bool = False,
        warm_start_from: str = None,
    ) -> Dict[str, Any]:
        """Chronos seq2seq forecasting with the dedicated encoder/MSE decoder path.

        Uses ``Phase1ChronosSeq2SeqDataset`` for explicit ``history -> future``
        supervision, but optimizes only the numeric forecasting stack:

            encoder -> mse_projector -> decoder

        This stage is deprecated in the current workflow and kept only for
        backward compatibility with older experiment names.
        """
        max_samp = getattr(self.args, "max_samples_per_source", None)
        dataset_cls = (
            partial(Phase1ChronosSeq2SeqDataset, max_samples_per_source=max_samp)
            if max_samp else Phase1ChronosSeq2SeqDataset
        )
        return self._train_stage(
            stage_name="phase1_chronos_seq2seq_separate_mse",
            dataset_class=dataset_cls,
            num_epochs=self.args.num_epochs,
            lr_encoder=LR_ENCODER_JOINT,
            lr_projector=1e-4,
            lr_base=2e-4,
            metric_func=None,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            eval_only=eval_only,
            warm_start_from=warm_start_from,
        )

    def phase1_chronos_mse(
        self, batch_size: int = None, val_batch_size: int = None, eval_only: bool = False,
        warm_start_from: str = None,
    ) -> Dict[str, Any]:
        """Standalone Chronos-only Phase-1 next-patch MSE training.

        This stage now follows the same shifted-sequence pattern as standard
        autoregressive next-token training, but on continuous patch targets:

            [history patches, true next patch]
              -> encoder
              -> projector
              -> decoder_projector
              -> causal decoder
              -> one-step shifted next-patch MSE
        """
        max_samp = getattr(self.args, "max_samples_per_source", None)
        min_hist_patches = max(1, int(getattr(self.args, "phase1_min_history_patches", 8)))
        dataset_kwargs = {"min_history_patches": min_hist_patches}
        if max_samp:
            dataset_kwargs["max_samples_per_source"] = max_samp
        stride = getattr(self.args, "chronos_window_stride", None)
        if stride is not None:
            dataset_kwargs["chronos_window_stride"] = int(stride)
        sampling = getattr(self.args, "chronos_window_sampling", "stride")
        if sampling and sampling != "stride":
            dataset_kwargs["chronos_window_sampling"] = sampling
        rand_n = getattr(self.args, "chronos_random_samples_per_row", None)
        if rand_n is not None:
            dataset_kwargs["chronos_random_samples_per_row"] = int(rand_n)
        if getattr(self.args, "chronos_univariate", False):
            dataset_kwargs["chronos_univariate"] = True
        synth_dir = getattr(self.args, "synthetic_data_dir", None)
        if synth_dir:
            dataset_kwargs["synthetic_data_dir"] = synth_dir
        synth_cap = getattr(self.args, "synthetic_max_rows_per_file", None)
        if synth_cap is not None:
            dataset_kwargs["synthetic_max_rows_per_file"] = int(synth_cap)
        dataset_cls = partial(Phase1ChronosFuturePatchDataset, **dataset_kwargs)

        # Source-aware sampler. When the corpus mixes registry-univariate Chronos
        # and synthetic-MV, the raw pool is ~125 uv stubs vs ~4.5M synthetic
        # stubs (1:36000). Without weighting, random sampling makes every batch
        # essentially all-synthetic and uv contributes ~0% of training. Group
        # chronos rows into uv/synthetic buckets and let the user balance them
        # via --task_weight_chronos_uv / --task_weight_synthetic (default 1:2).
        # Only kicks in when both sources are present; otherwise falls through
        # to the default loader and behaviour is unchanged.
        sampler = None
        _has_uv     = bool(getattr(self.args, "chronos_univariate", False))
        _has_synth  = synth_dir is not None and bool(synth_dir)
        if _has_uv and _has_synth:
            train_dataset = dataset_cls(
                "train",
                EOS_TOKEN=self._get_model().get_eos_token(),
                data_seed=self.args.data_seed,
                patch_size=PATCH_SIZE,
            )
            _w_uv    = float(getattr(self.args, "task_weight_chronos_uv", 1.0))
            _w_synth = float(getattr(self.args, "task_weight_synthetic",  2.0))

            def _task_key_source_aware(row):
                if not isinstance(row, dict):
                    return "chronos"
                return row.get("source") or "chronos"

            _task_weights = {
                "chronos_uv": _w_uv,
                "synthetic":  _w_synth,
                # Legacy single-bucket fallback for rows that don't carry source.
                "chronos":    0.5 * (_w_uv + _w_synth),
            }
            sampler = TaskWeightedDistributedBatchSampler(
                train_dataset,
                batch_size=batch_size if batch_size is not None else BATCH_SIZE,
                task_weights=_task_weights,
                num_replicas=self.world_size if dist.is_initialized() else 1,
                rank=self.rank if dist.is_initialized() else 0,
                shuffle=True,
                drop_last=True,
                seed=self.args.data_seed,
                task_key_fn=_task_key_source_aware,
            )
            if self.rank == 0:
                from collections import Counter
                task_counts = Counter(_task_key_source_aware(r) for r in train_dataset.rows)
                print(f"[phase1_chronos_mse] source-aware sampler enabled.")
                print(f"[phase1_chronos_mse] task counts (source-aware): {dict(task_counts)}")
                print(
                    f"[phase1_chronos_mse] task weights: "
                    f"chronos_uv={_w_uv} synthetic={_w_synth}"
                )
                print(
                    f"[phase1_chronos_mse] balanced batches per source ≈ "
                    f"{len(sampler) // max(1, len(task_counts))}"
                )

        return self._train_stage(
            stage_name="phase1_chronos_mse",
            dataset_class=dataset_cls,
            num_epochs=self.args.num_epochs,
            lr_encoder=LR_ENCODER,
            lr_projector=LR_PROJECTOR,
            lr_base=2e-4,
            metric_func=None,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            eval_only=eval_only,
            warm_start_from=warm_start_from,
            sampler=sampler,
        )

    def phase1_chronos_mse_lora(
        self, batch_size: int = None, val_batch_size: int = None, eval_only: bool = False,
        warm_start_from: str = None,
    ) -> Dict[str, Any]:
        """Identical to phase1_chronos_mse but with LLM LoRA (r=4, alpha=8) enabled.

        LoRA allows the LLM to adapt its representations to time-series patches
        while keeping the number of new parameters small.  All other training
        details (dataset, sampler, LRs) are unchanged from the frozen-LLM baseline
        so results are directly comparable.
        """
        max_samp = getattr(self.args, "max_samples_per_source", None)
        min_hist_patches = max(1, int(getattr(self.args, "phase1_min_history_patches", 8)))
        dataset_kwargs = {"min_history_patches": min_hist_patches}
        if max_samp:
            dataset_kwargs["max_samples_per_source"] = max_samp
        stride = getattr(self.args, "chronos_window_stride", None)
        if stride is not None:
            dataset_kwargs["chronos_window_stride"] = int(stride)
        sampling = getattr(self.args, "chronos_window_sampling", "stride")
        if sampling and sampling != "stride":
            dataset_kwargs["chronos_window_sampling"] = sampling
        rand_n = getattr(self.args, "chronos_random_samples_per_row", None)
        if rand_n is not None:
            dataset_kwargs["chronos_random_samples_per_row"] = int(rand_n)
        if getattr(self.args, "chronos_univariate", False):
            dataset_kwargs["chronos_univariate"] = True
        synth_dir = getattr(self.args, "synthetic_data_dir", None)
        if synth_dir:
            dataset_kwargs["synthetic_data_dir"] = synth_dir
        synth_cap = getattr(self.args, "synthetic_max_rows_per_file", None)
        if synth_cap is not None:
            dataset_kwargs["synthetic_max_rows_per_file"] = int(synth_cap)
        dataset_cls = partial(Phase1ChronosFuturePatchDataset, **dataset_kwargs)

        sampler = None
        _has_uv    = bool(getattr(self.args, "chronos_univariate", False))
        _has_synth = synth_dir is not None and bool(synth_dir)
        if _has_uv and _has_synth:
            train_dataset = dataset_cls(
                "train",
                EOS_TOKEN=self._get_model().get_eos_token(),
                data_seed=self.args.data_seed,
                patch_size=PATCH_SIZE,
            )
            _w_uv    = float(getattr(self.args, "task_weight_chronos_uv", 1.0))
            _w_synth = float(getattr(self.args, "task_weight_synthetic",  2.0))

            def _task_key_source_aware(row):
                if not isinstance(row, dict):
                    return "chronos"
                return row.get("source") or "chronos"

            _task_weights = {
                "chronos_uv": _w_uv,
                "synthetic":  _w_synth,
                "chronos":    0.5 * (_w_uv + _w_synth),
            }
            sampler = TaskWeightedDistributedBatchSampler(
                train_dataset,
                batch_size=batch_size if batch_size is not None else BATCH_SIZE,
                task_weights=_task_weights,
                num_replicas=self.world_size if dist.is_initialized() else 1,
                rank=self.rank if dist.is_initialized() else 0,
                shuffle=True,
                drop_last=True,
                seed=self.args.data_seed,
                task_key_fn=_task_key_source_aware,
            )
            if self.rank == 0:
                from collections import Counter
                task_counts = Counter(_task_key_source_aware(r) for r in train_dataset.rows)
                print(f"[phase1_chronos_mse_lora] source-aware sampler enabled.")
                print(f"[phase1_chronos_mse_lora] task counts: {dict(task_counts)}")
                print(
                    f"[phase1_chronos_mse_lora] task weights: "
                    f"chronos_uv={_w_uv} synthetic={_w_synth}"
                )

        return self._train_stage(
            stage_name="phase1_chronos_mse_lora",
            dataset_class=dataset_cls,
            num_epochs=self.args.num_epochs,
            lr_encoder=LR_ENCODER,
            lr_projector=LR_PROJECTOR,
            lr_base=2e-4,
            metric_func=None,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            eval_only=eval_only,
            warm_start_from=warm_start_from,
            sampler=sampler,
        )

    def phase1_chronos_mse_full(
        self, batch_size: int = None, val_batch_size: int = None, eval_only: bool = False,
        warm_start_from: str = None,
    ) -> Dict[str, Any]:
        """Identical to phase1_chronos_mse but unfreezes the FULL LLM (no LoRA).

        Encoder / projector / decoder / decoder_projector keep their defaults
        (1e-4). The whole Gemma-3-270M backbone is unfrozen and added to the
        optimizer with ``lr_base=5e-5`` — a small LR to keep the pretrained LLM
        close to its starting point while still letting it adapt to time-series
        tokens (the standard convention for full fine-tuning alongside trained
        adapters/heads).
        """
        max_samp = getattr(self.args, "max_samples_per_source", None)
        min_hist_patches = max(1, int(getattr(self.args, "phase1_min_history_patches", 8)))
        dataset_kwargs = {"min_history_patches": min_hist_patches}
        if max_samp:
            dataset_kwargs["max_samples_per_source"] = max_samp
        stride = getattr(self.args, "chronos_window_stride", None)
        if stride is not None:
            dataset_kwargs["chronos_window_stride"] = int(stride)
        sampling = getattr(self.args, "chronos_window_sampling", "stride")
        if sampling and sampling != "stride":
            dataset_kwargs["chronos_window_sampling"] = sampling
        rand_n = getattr(self.args, "chronos_random_samples_per_row", None)
        if rand_n is not None:
            dataset_kwargs["chronos_random_samples_per_row"] = int(rand_n)
        if getattr(self.args, "chronos_univariate", False):
            dataset_kwargs["chronos_univariate"] = True
        synth_dir = getattr(self.args, "synthetic_data_dir", None)
        if synth_dir:
            dataset_kwargs["synthetic_data_dir"] = synth_dir
        synth_cap = getattr(self.args, "synthetic_max_rows_per_file", None)
        if synth_cap is not None:
            dataset_kwargs["synthetic_max_rows_per_file"] = int(synth_cap)
        dataset_cls = partial(Phase1ChronosFuturePatchDataset, **dataset_kwargs)

        sampler = None
        _has_uv    = bool(getattr(self.args, "chronos_univariate", False))
        _has_synth = synth_dir is not None and bool(synth_dir)
        if _has_uv and _has_synth:
            train_dataset = dataset_cls(
                "train",
                EOS_TOKEN=self._get_model().get_eos_token(),
                data_seed=self.args.data_seed,
                patch_size=PATCH_SIZE,
            )
            _w_uv    = float(getattr(self.args, "task_weight_chronos_uv", 1.0))
            _w_synth = float(getattr(self.args, "task_weight_synthetic",  2.0))

            def _task_key_source_aware(row):
                if not isinstance(row, dict):
                    return "chronos"
                return row.get("source") or "chronos"

            _task_weights = {
                "chronos_uv": _w_uv,
                "synthetic":  _w_synth,
                "chronos":    0.5 * (_w_uv + _w_synth),
            }
            sampler = TaskWeightedDistributedBatchSampler(
                train_dataset,
                batch_size=batch_size if batch_size is not None else BATCH_SIZE,
                task_weights=_task_weights,
                num_replicas=self.world_size if dist.is_initialized() else 1,
                rank=self.rank if dist.is_initialized() else 0,
                shuffle=True,
                drop_last=True,
                seed=self.args.data_seed,
                task_key_fn=_task_key_source_aware,
            )
            if self.rank == 0:
                from collections import Counter
                task_counts = Counter(_task_key_source_aware(r) for r in train_dataset.rows)
                print(f"[phase1_chronos_mse_full] source-aware sampler enabled.")
                print(f"[phase1_chronos_mse_full] task counts: {dict(task_counts)}")
                print(
                    f"[phase1_chronos_mse_full] task weights: "
                    f"chronos_uv={_w_uv} synthetic={_w_synth}"
                )

        return self._train_stage(
            stage_name="phase1_chronos_mse_full",
            dataset_class=dataset_cls,
            num_epochs=self.args.num_epochs,
            lr_encoder=LR_ENCODER,
            lr_projector=LR_PROJECTOR,
            lr_base=5e-5,
            metric_func=None,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            eval_only=eval_only,
            warm_start_from=warm_start_from,
            sampler=sampler,
        )

    def phase1_chronos_joint_mse(
        self, batch_size: int = None, val_batch_size: int = None, eval_only: bool = False,
        warm_start_from: str = None,
    ) -> Dict[str, Any]:
        """Chronos-only Phase-1 next-patch training with the joint-Phase-1 MSE loss.

        This stage uses ``Phase1ChronosOnlyDataset`` but routes through
        ``compute_joint_phase1_loss`` instead of ``compute_pretrain_loss`` so the
        forecasting loss matches the Chronos branch used inside the joint Phase-1
        stages:

            encoder -> projector -> decoder_projector -> decoder (via joint loss)

        With a Chronos-only dataset, CE remains zero and the optimization target
        is exactly the Chronos MSE term from joint Phase-1 training.
        """
        max_samp = getattr(self.args, "max_samples_per_source", None)
        dataset_cls = (
            partial(Phase1ChronosOnlyDataset, max_samples_per_source=max_samp)
            if max_samp else Phase1ChronosOnlyDataset
        )
        return self._train_stage(
            stage_name="phase1_chronos_joint_mse",
            dataset_class=dataset_cls,
            num_epochs=self.args.num_epochs,
            lr_encoder=LR_ENCODER_JOINT,
            lr_projector=1e-4,
            lr_base=2e-4,
            metric_func=None,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            eval_only=eval_only,
            warm_start_from=warm_start_from,
        )

    def phase1_MSE_then_CE(
        self, batch_size: int = None, val_batch_size: int = None, eval_only: bool = False,
        warm_start_from: str = None,
    ) -> Dict[str, Any]:
        """Phase-1 sequential: MSE pretraining on Chronos, then CE fine-tuning on TSQA+M4.

        Sub-stage 1 — ``phase1_chronos_mse``:
            Dataset : Phase1ChronosOnlyDataset (Chronos only)
            Loss    : compute_pretrain_loss(..., mse_only=True)
            Gradient: encoder → projector → LLM → decoder_projector → decoder (MSE path)

        Sub-stage 2 — ``phase1_tsqa_m4_ce``:
            Dataset : Phase1JointTsqaM4Dataset (TSQA + M4, no Chronos)
            Loss    : compute_loss (CE only)
            Warm-start from sub-stage 1 checkpoint.
        """
        max_samp = getattr(self.args, "max_samples_per_source", None)

        # ── Sub-stage 1: Chronos MSE ──────────────────────────────────────
        chronos_cls = (
            partial(Phase1ChronosOnlyDataset, max_samples_per_source=max_samp)
            if max_samp else Phase1ChronosOnlyDataset
        )
        self._train_stage(
            stage_name="phase1_chronos_mse",
            dataset_class=chronos_cls,
            num_epochs=self.args.num_epochs,
            lr_encoder=LR_ENCODER,       # aggressive encoder LR for pure MSE pretraining (not joint preservation)
            lr_projector=LR_PROJECTOR,
            lr_base=2e-4,
            metric_func=None,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            eval_only=eval_only,
            warm_start_from=warm_start_from,
        )

        # ── Sub-stage 2: TSQA + M4 CE ─────────────────────────────────────
        ce_cls = (
            partial(Phase1JointTsqaM4Dataset, max_samples_per_source=max_samp)
            if max_samp else Phase1JointTsqaM4Dataset
        )
        return self._train_stage(
            stage_name="phase1_tsqa_m4_ce",
            dataset_class=ce_cls,
            num_epochs=self.args.num_epochs,
            lr_encoder=LR_ENCODER_JOINT,
            lr_projector=1e-4,
            lr_base=2e-4,
            metric_func=lambda preds, golds: {
                "accuracy": self._calculate_accuracy(preds, golds)
            },
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            eval_only=eval_only,
            warm_start_from="phase1_chronos_mse",
        )

    def joint_phase1_extend_lorallm(
        self, batch_size: int = None, val_batch_size: int = None, eval_only: bool = False,
        warm_start_from: str = None,
    ) -> Dict[str, Any]:
        """Joint Phase-1 Extended with LLM LoRA (r=4): identical to joint_phase1_extend
        but the LLM is lightly adapted via low-rank adapters to improve TS-embedding
        processing without drifting far from its pretrained weights.
        """
        max_samp = getattr(self.args, "max_samples_per_source", None)
        dataset_cls = (
            partial(Phase1JointExtendDataset, max_samples_per_source=max_samp)
            if max_samp else Phase1JointExtendDataset
        )
        return self._train_stage(
            stage_name="joint_phase1_extend_lorallm",
            dataset_class=dataset_cls,
            num_epochs=self.args.num_epochs,
            lr_encoder=LR_ENCODER_JOINT,
            lr_projector=1e-4,
            lr_base=2e-4,
            metric_func=lambda preds, golds: {
                "accuracy": self._calculate_accuracy(preds, golds)
            },
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            eval_only=eval_only,
            warm_start_from=warm_start_from,
        )

    def joint_phase1_extended_separate_mse(
        self, batch_size: int = None, val_batch_size: int = None, eval_only: bool = False,
        warm_start_from: str = None,
    ) -> Dict[str, Any]:
        """Joint Phase-1 Extended with a separate forecasting MSE path.

        CE tasks still use the normal SoftPrompt LLM path. Chronos forecasting
        bypasses the LLM hidden states and reads the projected encoder outputs:
        encoder -> projector -> decoder_projector -> decoder.
        """
        max_samp = getattr(self.args, "max_samples_per_source", None)
        dataset_cls = (
            partial(Phase1JointExtendDataset, max_samples_per_source=max_samp)
            if max_samp else Phase1JointExtendDataset
        )
        return self._train_stage(
            stage_name="joint_phase1_extended_separate_mse",
            dataset_class=dataset_cls,
            num_epochs=self.args.num_epochs,
            lr_encoder=LR_ENCODER_JOINT,
            lr_projector=1e-4,
            lr_base=2e-4,
            metric_func=lambda preds, golds: {
                "accuracy": self._calculate_accuracy(preds, golds)
            },
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            eval_only=eval_only,
            warm_start_from=warm_start_from,
        )

    def joint_phase1_flamingo(
        self, batch_size: int = None, val_batch_size: int = None, eval_only: bool = False,
        warm_start_from: str = None,
    ) -> Dict[str, Any]:
        """Joint Phase-1 for OpenTSLMFlamingo: Chronos MSE + TSQA/M4 CE.

        Uses TransformerCNNEncoder2D + TransformerCNNDecoder2D for the MSE path
        (bypassing the LLM entirely) and the Flamingo perceiver + gated cross-attn
        layers for the CE path.  No LoRA — the LLM backbone stays frozen.
        """
        max_samp = getattr(self.args, "max_samples_per_source", None)
        dataset_cls = (
            partial(Phase1JointExtendDataset, max_samples_per_source=max_samp)
            if max_samp else Phase1JointExtendDataset
        )
        return self._train_stage(
            stage_name="joint_phase1_flamingo",
            dataset_class=dataset_cls,
            num_epochs=self.args.num_epochs,
            lr_encoder=LR_ENCODER_JOINT,
            lr_projector=1e-4,   # perceiver LR
            lr_base=2e-4,        # gated cross-attn LR
            metric_func=lambda preds, golds: {
                "accuracy": self._calculate_accuracy(preds, golds)
            },
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            eval_only=eval_only,
            warm_start_from=warm_start_from,
        )

    def joint_phase1_extend_balanced(
        self, batch_size: int = None, val_batch_size: int = None, eval_only: bool = False,
        warm_start_from: str = None,
    ) -> Dict[str, Any]:
        """Joint Phase-1 Extended with task-balanced sampling (Transfusion-style).

        Identical data to joint_phase1_extend (Chronos + TSQA + M4, full datasets),
        but uses TaskWeightedDistributedBatchSampler with equal weights (1:1:1)
        so each task contributes the same number of batches per epoch regardless
        of dataset size.  Chronos (525k) is undersampled; TSQA (38k) and M4 (80k)
        are oversampled to match.

        Motivated by the observed accuracy drop from 96% → 93% when adding
        Chronos at its natural 80% proportion — CE signal was diluted 5x.
        """
        if batch_size is None:
            batch_size = BATCH_SIZE
        max_samp = getattr(self.args, "max_samples_per_source", None)
        stride = getattr(self.args, "chronos_window_stride", None)
        extra_kwargs = {}
        if max_samp:
            extra_kwargs["max_samples_per_source"] = max_samp
        if stride is not None:
            extra_kwargs["chronos_window_stride"] = int(stride)
        sampling = getattr(self.args, "chronos_window_sampling", "stride")
        if sampling and sampling != "stride":
            extra_kwargs["chronos_window_sampling"] = sampling
        rand_n = getattr(self.args, "chronos_random_samples_per_row", None)
        if rand_n is not None:
            extra_kwargs["chronos_random_samples_per_row"] = int(rand_n)
        # Same Chronos-corpus controls as phase1_chronos_mse: keep the registry
        # data univariate and/or fold in the synthetic MV datasets. The TSQA and
        # M4 sub-tasks are unaffected — they are loaded independently.
        if getattr(self.args, "chronos_univariate", False):
            extra_kwargs["chronos_univariate"] = True
        synth_dir = getattr(self.args, "synthetic_data_dir", None)
        if synth_dir:
            extra_kwargs["synthetic_data_dir"] = synth_dir
        synth_cap = getattr(self.args, "synthetic_max_rows_per_file", None)
        if synth_cap is not None:
            extra_kwargs["synthetic_max_rows_per_file"] = int(synth_cap)
        dataset_cls = (
            partial(Phase1JointExtendDataset, **extra_kwargs)
            if extra_kwargs else Phase1JointExtendDataset
        )
        train_dataset = dataset_cls(
            "train",
            EOS_TOKEN=self._get_model().get_eos_token(),
            data_seed=self.args.data_seed,
            patch_size=PATCH_SIZE,
            val_fraction=JOINT_VAL_FRACTION,
        )
        # When the chronos corpus mixes registry-univariate and synthetic-MV
        # data, group those as two SEPARATE sampler tasks (chronos_uv /
        # synthetic) so their per-batch ratio is user-controlled instead of
        # being driven by raw pool sizes (synthetic out-counts uv ~30k:1).
        # Falls back to the legacy single "chronos" bucket when no source tag
        # is present (back-compat).
        _w_uv     = float(getattr(self.args, "task_weight_chronos_uv", 1.0))
        _w_synth  = float(getattr(self.args, "task_weight_synthetic",  2.0))
        _w_tsqa   = float(getattr(self.args, "task_weight_tsqa",       1.0))
        _w_m4     = float(getattr(self.args, "task_weight_m4",         1.0))

        def _task_key_source_aware(row):
            if not isinstance(row, dict):
                return "unknown"
            t = row.get("type", "unknown")
            if t == "chronos":
                # source ∈ {"chronos_uv", "synthetic"}; absent on legacy splits.
                return row.get("source") or "chronos"
            return t

        _task_weights = {
            "chronos_uv": _w_uv,
            "synthetic":  _w_synth,
            "tsqa":       _w_tsqa,
            "m4":         _w_m4,
            # Legacy single-bucket weight when source isn't tagged. Use the
            # average of uv+synthetic so behaviour falls back gracefully.
            "chronos":    0.5 * (_w_uv + _w_synth),
        }

        sampler = TaskWeightedDistributedBatchSampler(
            train_dataset,
            batch_size=batch_size,
            task_weights=_task_weights,
            num_replicas=self.world_size if dist.is_initialized() else 1,
            rank=self.rank if dist.is_initialized() else 0,
            shuffle=True,
            drop_last=True,
            seed=self.args.data_seed,
            task_key_fn=_task_key_source_aware,
        )
        if self.rank == 0:
            from collections import Counter
            task_counts = Counter(_task_key_source_aware(r) for r in train_dataset.rows)
            n_tasks = max(1, len(_task_weights))
            print(f"[joint_phase1_extend_balanced] task counts (source-aware): {dict(task_counts)}")
            print(
                f"[joint_phase1_extend_balanced] task weights: "
                f"chronos_uv={_w_uv} synthetic={_w_synth} tsqa={_w_tsqa} m4={_w_m4}"
            )
            print(
                f"[joint_phase1_extend_balanced] balanced batches per task ≈ "
                f"{len(sampler) // max(1, len(task_counts))}"
            )
        return self._train_stage(
            stage_name="joint_phase1_extend_balanced",
            dataset_class=dataset_cls,
            num_epochs=self.args.num_epochs,
            lr_encoder=LR_ENCODER_JOINT,
            lr_projector=1e-4,
            lr_base=2e-4,
            metric_func=lambda preds, golds: {
                "accuracy": self._calculate_accuracy(preds, golds)
            },
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            eval_only=eval_only,
            warm_start_from=warm_start_from,
            sampler=sampler,
        )

    def joint_phase1_extend_balanced_lora(
        self, batch_size: int = None, val_batch_size: int = None, eval_only: bool = False,
        warm_start_from: str = None,
    ) -> Dict[str, Any]:
        """Identical to joint_phase1_extend_balanced but with LLM LoRA (r=4, alpha=8).

        LoRA is applied to all attention and feed-forward projection matrices in
        Gemma-3-270M while encoder, projector, decoder, and decoder_projector remain
        fully trainable.  All dataset, sampler, and LR settings are unchanged from
        the frozen-LLM baseline so results are directly comparable.
        """
        if batch_size is None:
            batch_size = BATCH_SIZE
        max_samp = getattr(self.args, "max_samples_per_source", None)
        stride = getattr(self.args, "chronos_window_stride", None)
        extra_kwargs = {}
        if max_samp:
            extra_kwargs["max_samples_per_source"] = max_samp
        if stride is not None:
            extra_kwargs["chronos_window_stride"] = int(stride)
        sampling = getattr(self.args, "chronos_window_sampling", "stride")
        if sampling and sampling != "stride":
            extra_kwargs["chronos_window_sampling"] = sampling
        rand_n = getattr(self.args, "chronos_random_samples_per_row", None)
        if rand_n is not None:
            extra_kwargs["chronos_random_samples_per_row"] = int(rand_n)
        if getattr(self.args, "chronos_univariate", False):
            extra_kwargs["chronos_univariate"] = True
        synth_dir = getattr(self.args, "synthetic_data_dir", None)
        if synth_dir:
            extra_kwargs["synthetic_data_dir"] = synth_dir
        synth_cap = getattr(self.args, "synthetic_max_rows_per_file", None)
        if synth_cap is not None:
            extra_kwargs["synthetic_max_rows_per_file"] = int(synth_cap)
        dataset_cls = (
            partial(Phase1JointExtendDataset, **extra_kwargs)
            if extra_kwargs else Phase1JointExtendDataset
        )
        train_dataset = dataset_cls(
            "train",
            EOS_TOKEN=self._get_model().get_eos_token(),
            data_seed=self.args.data_seed,
            patch_size=PATCH_SIZE,
            val_fraction=JOINT_VAL_FRACTION,
        )
        _w_uv    = float(getattr(self.args, "task_weight_chronos_uv", 1.0))
        _w_synth = float(getattr(self.args, "task_weight_synthetic",  2.0))
        _w_tsqa  = float(getattr(self.args, "task_weight_tsqa",       1.0))
        _w_m4    = float(getattr(self.args, "task_weight_m4",         1.0))

        def _task_key_source_aware(row):
            if not isinstance(row, dict):
                return "unknown"
            t = row.get("type", "unknown")
            if t == "chronos":
                return row.get("source") or "chronos"
            return t

        _task_weights = {
            "chronos_uv": _w_uv,
            "synthetic":  _w_synth,
            "tsqa":       _w_tsqa,
            "m4":         _w_m4,
            "chronos":    0.5 * (_w_uv + _w_synth),
        }
        sampler = TaskWeightedDistributedBatchSampler(
            train_dataset,
            batch_size=batch_size,
            task_weights=_task_weights,
            num_replicas=self.world_size if dist.is_initialized() else 1,
            rank=self.rank if dist.is_initialized() else 0,
            shuffle=True,
            drop_last=True,
            seed=self.args.data_seed,
            task_key_fn=_task_key_source_aware,
        )
        if self.rank == 0:
            from collections import Counter
            task_counts = Counter(_task_key_source_aware(r) for r in train_dataset.rows)
            print(f"[joint_phase1_extend_balanced_lora] task counts: {dict(task_counts)}")
            print(
                f"[joint_phase1_extend_balanced_lora] task weights: "
                f"chronos_uv={_w_uv} synthetic={_w_synth} tsqa={_w_tsqa} m4={_w_m4}"
            )
        return self._train_stage(
            stage_name="joint_phase1_extend_balanced_lora",
            dataset_class=dataset_cls,
            num_epochs=self.args.num_epochs,
            lr_encoder=LR_ENCODER_JOINT,
            lr_projector=1e-4,
            lr_base=2e-4,
            metric_func=lambda preds, golds: {
                "accuracy": self._calculate_accuracy(preds, golds)
            },
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            eval_only=eval_only,
            warm_start_from=warm_start_from,
            sampler=sampler,
        )

    def joint_phase1_enc_mse(
        self, batch_size: int = None, val_batch_size: int = None, eval_only: bool = False,
        warm_start_from: str = None,
    ) -> Dict[str, Any]:
        """Joint Phase-1 with independent encoder MSE path (Chronos + TSQA + M4).

        Identical data and sampling to joint_phase1_extend_balanced (1:1:1 task
        weights), but uses compute_joint_phase1_enc_mse_loss instead of the
        standard compute_joint_phase1_loss.

        MSE path: encoder → mse_projector → decoder (bypasses projector and LLM)
        CE  path: encoder → projector → LLM (frozen) → CE loss  (unchanged)

        The two paths share only the encoder.  mse_projector is trained purely
        by MSE loss and never sees CE pressure, so it cannot collapse all patch
        representations into a single direction.
        """
        if batch_size is None:
            batch_size = BATCH_SIZE
        max_samp = getattr(self.args, "max_samples_per_source", None)
        dataset_cls = (
            partial(Phase1JointExtendDataset, max_samples_per_source=max_samp)
            if max_samp else Phase1JointExtendDataset
        )
        train_dataset = dataset_cls(
            "train",
            EOS_TOKEN=self._get_model().get_eos_token(),
            data_seed=self.args.data_seed,
            patch_size=PATCH_SIZE,
            val_fraction=JOINT_VAL_FRACTION,
            **({"max_samples_per_source": max_samp} if max_samp else {}),
        )
        sampler = TaskWeightedDistributedBatchSampler(
            train_dataset,
            batch_size=batch_size,
            task_weights={"chronos": 1.0, "tsqa": 1.0, "m4": 1.0},
            num_replicas=self.world_size if dist.is_initialized() else 1,
            rank=self.rank if dist.is_initialized() else 0,
            shuffle=True,
            drop_last=True,
            seed=self.args.data_seed,
        )
        if self.rank == 0:
            from collections import Counter
            task_counts = Counter(r.get("type", "unknown") for r in train_dataset.rows)
            print(f"[joint_phase1_enc_mse] task counts: {dict(task_counts)}")
            print(f"[joint_phase1_enc_mse] balanced batches per task ≈ {len(sampler) // 3}")
        return self._train_stage(
            stage_name="joint_phase1_enc_mse",
            dataset_class=dataset_cls,
            num_epochs=self.args.num_epochs,
            lr_encoder=LR_ENCODER_JOINT,
            lr_projector=1e-4,
            lr_base=2e-4,
            metric_func=lambda preds, golds: {
                "accuracy": self._calculate_accuracy(preds, golds)
            },
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            eval_only=eval_only,
            warm_start_from=warm_start_from,
            sampler=sampler,
        )

    def joint_phase1_extend_balanced_m4_by_size(
        self, batch_size: int = None, val_batch_size: int = None, eval_only: bool = False,
        warm_start_from: str = None,
    ) -> Dict[str, Any]:
        """Joint Phase-1 Extended with equal Chronos/TSQA weights and M4 by data size.

        Uses the same full Chronos + TSQA + M4 data as ``joint_phase1_extend``,
        but sets task weights so:
        - chronos : tsqa = 1 : 1
        - m4 : tsqa = |m4_train| : |tsqa_train|

        The M4 weight is computed from the actual train split used for this run,
        so any ``max_samples_per_source`` cap is reflected automatically.
        """
        if batch_size is None:
            batch_size = BATCH_SIZE
        max_samp = getattr(self.args, "max_samples_per_source", None)
        dataset_cls = (
            partial(Phase1JointExtendDataset, max_samples_per_source=max_samp)
            if max_samp else Phase1JointExtendDataset
        )
        train_dataset = dataset_cls(
            "train",
            EOS_TOKEN=self._get_model().get_eos_token(),
            data_seed=self.args.data_seed,
            patch_size=PATCH_SIZE,
            val_fraction=JOINT_VAL_FRACTION,
            **({"max_samples_per_source": max_samp} if max_samp else {}),
        )
        from collections import Counter

        task_counts = Counter(r.get("type", "unknown") for r in train_dataset.rows)
        tsqa_count = task_counts.get("tsqa", 0)
        m4_count = task_counts.get("m4", 0)
        if tsqa_count <= 0 or m4_count <= 0:
            raise ValueError(
                "joint_phase1_extend_balanced_m4_by_size requires both TSQA and M4 "
                f"samples, got counts: {dict(task_counts)}"
            )
        m4_weight = m4_count / tsqa_count
        task_weights = {"chronos": 1.0, "tsqa": 1.0, "m4": m4_weight}
        sampler = TaskWeightedDistributedBatchSampler(
            train_dataset,
            batch_size=batch_size,
            task_weights=task_weights,
            num_replicas=self.world_size if dist.is_initialized() else 1,
            rank=self.rank if dist.is_initialized() else 0,
            shuffle=True,
            drop_last=True,
            seed=self.args.data_seed,
        )
        if self.rank == 0:
            print(
                "[joint_phase1_extend_balanced_m4_by_size] "
                f"task counts: {dict(task_counts)}"
            )
            print(
                "[joint_phase1_extend_balanced_m4_by_size] "
                f"task weights: {task_weights}"
            )
            print(
                "[joint_phase1_extend_balanced_m4_by_size] "
                f"target ratio chronos:tsqa:m4 = 1:1:{m4_weight:.6f}"
            )
        return self._train_stage(
            stage_name="joint_phase1_extend_balanced_m4_by_size",
            dataset_class=dataset_cls,
            num_epochs=self.args.num_epochs,
            lr_encoder=LR_ENCODER_JOINT,
            lr_projector=1e-4,
            lr_base=2e-4,
            metric_func=lambda preds, golds: {
                "accuracy": self._calculate_accuracy(preds, golds)
            },
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            eval_only=eval_only,
            warm_start_from=warm_start_from,
            sampler=sampler,
        )

    def _joint_phase1_extend_m4_fixed_weight(
        self,
        stage_name: str,
        m4_weight: float,
        batch_size: int = None,
        val_batch_size: int = None,
        eval_only: bool = False,
        warm_start_from: str = None,
    ) -> Dict[str, Any]:
        """Shared implementation for fixed-M4-weight joint Phase-1 stages.

        chronos : tsqa : m4 = 1 : 1 : m4_weight
        """
        if batch_size is None:
            batch_size = BATCH_SIZE
        max_samp = getattr(self.args, "max_samples_per_source", None)
        dataset_cls = (
            partial(Phase1JointExtendDataset, max_samples_per_source=max_samp)
            if max_samp else Phase1JointExtendDataset
        )
        task_weights = {"chronos": 1.0, "tsqa": 1.0, "m4": m4_weight}
        train_dataset = dataset_cls(
            "train",
            EOS_TOKEN=self._get_model().get_eos_token(),
            data_seed=self.args.data_seed,
            patch_size=PATCH_SIZE,
            val_fraction=JOINT_VAL_FRACTION,
            **({"max_samples_per_source": max_samp} if max_samp else {}),
        )
        sampler = TaskWeightedDistributedBatchSampler(
            train_dataset,
            batch_size=batch_size,
            task_weights=task_weights,
            num_replicas=self.world_size if dist.is_initialized() else 1,
            rank=self.rank if dist.is_initialized() else 0,
            shuffle=True,
            drop_last=True,
            seed=self.args.data_seed,
        )
        if self.rank == 0:
            from collections import Counter
            task_counts = Counter(r.get("type", "unknown") for r in train_dataset.rows)
            print(f"[{stage_name}] task counts: {dict(task_counts)}")
            print(f"[{stage_name}] task weights: {task_weights}")
            print(f"[{stage_name}] target ratio chronos:tsqa:m4 = 1:1:{m4_weight:.2f}")
        return self._train_stage(
            stage_name=stage_name,
            dataset_class=dataset_cls,
            num_epochs=self.args.num_epochs,
            lr_encoder=LR_ENCODER_JOINT,
            lr_projector=1e-4,
            lr_base=2e-4,
            metric_func=lambda preds, golds: {
                "accuracy": self._calculate_accuracy(preds, golds)
            },
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            eval_only=eval_only,
            warm_start_from=warm_start_from,
            sampler=sampler,
        )

    def joint_phase1_extend_m4_w05(
        self, batch_size: int = None, val_batch_size: int = None, eval_only: bool = False,
        warm_start_from: str = None,
    ) -> Dict[str, Any]:
        """Joint Phase-1 Extended: chronos:tsqa:m4 = 1:1:0.5.

        M4 is halved relative to the equal-weight baseline to reduce
        gradient interference with TSQA.
        """
        return self._joint_phase1_extend_m4_fixed_weight(
            stage_name="joint_phase1_extend_m4_w05",
            m4_weight=0.5,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            eval_only=eval_only,
            warm_start_from=warm_start_from,
        )

    def joint_phase1_extend_m4_w15(
        self, batch_size: int = None, val_batch_size: int = None, eval_only: bool = False,
        warm_start_from: str = None,
    ) -> Dict[str, Any]:
        """Joint Phase-1 Extended: chronos:tsqa:m4 = 1:1:1.5.

        M4 sits between the equal-weight (1:1:1) and by-size (~1:2:1) baselines.
        """
        return self._joint_phase1_extend_m4_fixed_weight(
            stage_name="joint_phase1_extend_m4_w15",
            m4_weight=1.5,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            eval_only=eval_only,
            warm_start_from=warm_start_from,
        )

    def stage2_all(
        self, batch_size: int = None, val_batch_size: int = None, eval_only: bool = False
    ) -> Dict[str, Any]:
        """Stage 2 Foundational: Instruction tuning on all datasets, including:
        - Chronos (forecasting dataset)
        - ChatTS (multivariate dataset with chat format)

        Configuration:
        - Epochs: 60
        - OpenTSLMSP: encoder_lr=2e-4, projector_lr=1e-4
        - OpenTSLMFlamingo: base_lr=2e-4
        - Metric: Test loss only
        """
        sampler = None

        return self._train_stage(
            stage_name="stage2_all",
            dataset_class=Stage2AllDataset,
            num_epochs=self.args.num_epochs,
            lr_encoder=1e-4, # lower learning rate for fine-tuning
            lr_projector=5e-5, # lower learning rate for fine-tuning
            lr_base=1e-4, # lower learning rate for fine-tuning
            metric_func=None,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            eval_only=eval_only,
            sampler=sampler,
            warm_start_from="pretrain_chronos",
        )


    def run_curriculum(
        self,
        stages: List[str] = None,
        batch_size: int = None,
        val_batch_size: int = None,
        eval_only: bool = False,
    ):
        """Run the complete curriculum learning pipeline."""
        if stages is None:
            stages = CURRICULUM_STAGES

        # Filter out completed stages
        _warm = getattr(self.args, "warm_start_from", None)
        incomplete_stages = []
        for stage in stages:
            _stage_dir = f"{stage}_from_{_warm}" if _warm else stage
            if self._is_stage_completed(_stage_dir):
                if self.rank == 0:
                    print(f"⏭️  Skipping completed stage: {_stage_dir}")
            else:
                incomplete_stages.append(stage)

        if self.rank == 0:
            print(f"🎓 Starting Curriculum Learning with {self.model_type}")
            if eval_only:
                print("🔍 EVAL-ONLY MODE: Will skip training and only run evaluation")
            print(f"📊 All stages: {', '.join(stages)}")
            print(f"🔄 Incomplete stages: {', '.join(incomplete_stages)}")
            print(f"💻 Device: {self.device}")
            if batch_size:
                print(f"📦 Batch size: {batch_size}")
            if val_batch_size:
                print(f"🧪 Val batch size: {val_batch_size}")
            if self.world_size > 1:
                print(f"🌐 Distributed training with {self.world_size} GPUs")
            print("=" * 80)

        results = {}

        # Run only incomplete stages
        for stage in incomplete_stages:
            # Synchronize all ranks before starting each stage
            if dist.is_initialized():
                dist.barrier()

            if stage == "stage1_mcq":
                stage_results = self.stage1_mcq(
                    batch_size=batch_size, val_batch_size=val_batch_size, eval_only=eval_only,
                    warm_start_from=getattr(self.args, "warm_start_from", None),
                )
                results[stage] = stage_results
                self._mark_stage_completed(stage_results.get("_stage_dir", stage), stage_results)
            elif stage == "stage2_captioning":
                stage_results = self.stage2_captioning(
                    batch_size=batch_size, val_batch_size=val_batch_size, eval_only=eval_only,
                    warm_start_from=getattr(self.args, "warm_start_from", None),
                )
                results[stage] = stage_results
                self._mark_stage_completed(stage_results.get("_stage_dir", stage), stage_results)
            elif stage == "stage3_cot":
                stage_results = self.stage3_cot(
                    batch_size=batch_size, val_batch_size=val_batch_size, eval_only=eval_only,
                    warm_start_from=getattr(self.args, "warm_start_from", None),
                )
                results[stage] = stage_results
                self._mark_stage_completed(stage_results.get("_stage_dir", stage), stage_results)
            elif stage == "stage4_sleep_cot":
                stage_results = self.stage4_sleep_cot(
                    batch_size=batch_size, val_batch_size=val_batch_size, eval_only=eval_only,
                    warm_start_from=getattr(self.args, "warm_start_from", None),
                )
                results[stage] = stage_results
                self._mark_stage_completed(stage_results.get("_stage_dir", stage), stage_results)
            elif stage == "stage5_ecg_cot":
                stage_results = self.stage5_ecg_cot(
                    batch_size=batch_size, val_batch_size=val_batch_size, eval_only=eval_only,
                    warm_start_from=getattr(self.args, "warm_start_from", None),
                )
                results[stage] = stage_results
                self._mark_stage_completed(stage_results.get("_stage_dir", stage), stage_results)
            elif stage == "joint_phase1":
                stage_results = self.joint_phase1(
                    batch_size=batch_size, val_batch_size=val_batch_size, eval_only=eval_only
                )
                results[stage] = stage_results
                self._mark_stage_completed(stage_results.get("_stage_dir", stage), stage_results)

            elif stage == "joint_phase1_extend":
                stage_results = self.joint_phase1_extend(
                    batch_size=batch_size, val_batch_size=val_batch_size, eval_only=eval_only,
                    warm_start_from=getattr(self.args, "warm_start_from", None),
                )
                results[stage] = stage_results

            elif stage == "joint_phase1_tsqa_m4":
                stage_results = self.joint_phase1_tsqa_m4(
                    batch_size=batch_size, val_batch_size=val_batch_size, eval_only=eval_only,
                    warm_start_from=getattr(self.args, "warm_start_from", None),
                )
                results[stage] = stage_results

            elif stage == "joint_phase1_tsqa":
                stage_results = self.joint_phase1_tsqa(
                    batch_size=batch_size, val_batch_size=val_batch_size, eval_only=eval_only,
                    warm_start_from=getattr(self.args, "warm_start_from", None),
                )
                results[stage] = stage_results

            elif stage == "joint_phase1_extend_lorallm":
                stage_results = self.joint_phase1_extend_lorallm(
                    batch_size=batch_size, val_batch_size=val_batch_size, eval_only=eval_only,
                    warm_start_from=getattr(self.args, "warm_start_from", None),
                )
                results[stage] = stage_results

            elif stage == "joint_phase1_extended_separate_mse":
                stage_results = self.joint_phase1_extended_separate_mse(
                    batch_size=batch_size, val_batch_size=val_batch_size, eval_only=eval_only,
                    warm_start_from=getattr(self.args, "warm_start_from", None),
                )
                results[stage] = stage_results

            elif stage == "joint_phase1_extend_balanced":
                stage_results = self.joint_phase1_extend_balanced(
                    batch_size=batch_size, val_batch_size=val_batch_size, eval_only=eval_only,
                    warm_start_from=getattr(self.args, "warm_start_from", None),
                )
                results[stage] = stage_results

            elif stage == "joint_phase1_extend_balanced_lora":
                stage_results = self.joint_phase1_extend_balanced_lora(
                    batch_size=batch_size, val_batch_size=val_batch_size, eval_only=eval_only,
                    warm_start_from=getattr(self.args, "warm_start_from", None),
                )
                results[stage] = stage_results

            elif stage == "joint_phase1_extend_balanced_m4_by_size":
                stage_results = self.joint_phase1_extend_balanced_m4_by_size(
                    batch_size=batch_size, val_batch_size=val_batch_size, eval_only=eval_only,
                    warm_start_from=getattr(self.args, "warm_start_from", None),
                )
                results[stage] = stage_results

            elif stage == "joint_phase1_enc_mse":
                stage_results = self.joint_phase1_enc_mse(
                    batch_size=batch_size, val_batch_size=val_batch_size, eval_only=eval_only,
                    warm_start_from=getattr(self.args, "warm_start_from", None),
                )
                results[stage] = stage_results

            elif stage == "joint_phase1_extend_m4_w05":
                stage_results = self.joint_phase1_extend_m4_w05(
                    batch_size=batch_size, val_batch_size=val_batch_size, eval_only=eval_only,
                    warm_start_from=getattr(self.args, "warm_start_from", None),
                )
                results[stage] = stage_results

            elif stage == "joint_phase1_extend_m4_w15":
                stage_results = self.joint_phase1_extend_m4_w15(
                    batch_size=batch_size, val_batch_size=val_batch_size, eval_only=eval_only,
                    warm_start_from=getattr(self.args, "warm_start_from", None),
                )
                results[stage] = stage_results

            elif stage == "phase1_chronos_seq2seq":
                stage_results = self.phase1_chronos_seq2seq(
                    batch_size=batch_size, val_batch_size=val_batch_size, eval_only=eval_only,
                    warm_start_from=getattr(self.args, "warm_start_from", None),
                )
                results[stage] = stage_results

            elif stage == "phase1_chronos_seq2seq_separate_mse":
                stage_results = self.phase1_chronos_seq2seq_separate_mse(
                    batch_size=batch_size, val_batch_size=val_batch_size, eval_only=eval_only,
                    warm_start_from=getattr(self.args, "warm_start_from", None),
                )
                results[stage] = stage_results

            elif stage == "phase1_chronos_mse":
                stage_results = self.phase1_chronos_mse(
                    batch_size=batch_size, val_batch_size=val_batch_size, eval_only=eval_only,
                    warm_start_from=getattr(self.args, "warm_start_from", None),
                )
                results[stage] = stage_results

            elif stage == "phase1_chronos_mse_lora":
                stage_results = self.phase1_chronos_mse_lora(
                    batch_size=batch_size, val_batch_size=val_batch_size, eval_only=eval_only,
                    warm_start_from=getattr(self.args, "warm_start_from", None),
                )
                results[stage] = stage_results

            elif stage == "phase1_chronos_mse_full":
                stage_results = self.phase1_chronos_mse_full(
                    batch_size=batch_size, val_batch_size=val_batch_size, eval_only=eval_only,
                    warm_start_from=getattr(self.args, "warm_start_from", None),
                )
                results[stage] = stage_results

            elif stage == "phase1_chronos_joint_mse":
                stage_results = self.phase1_chronos_joint_mse(
                    batch_size=batch_size, val_batch_size=val_batch_size, eval_only=eval_only,
                    warm_start_from=getattr(self.args, "warm_start_from", None),
                )
                results[stage] = stage_results

            elif stage == "phase1_mse_then_ce":
                stage_results = self.phase1_MSE_then_CE(
                    batch_size=batch_size, val_batch_size=val_batch_size, eval_only=eval_only,
                    warm_start_from=getattr(self.args, "warm_start_from", None),
                )
                results[stage] = stage_results
                self._mark_stage_completed(stage_results.get("_stage_dir", stage), stage_results)

            elif stage == "joint_phase2":
                stage_results = self.joint_phase2(
                    batch_size=batch_size, val_batch_size=val_batch_size, eval_only=eval_only,
                    warm_start_from=getattr(self.args, "warm_start_from", None),
                )
                results[stage] = stage_results
                self._mark_stage_completed(stage_results.get("_stage_dir", stage), stage_results)

            elif stage == "joint_phase2_open":
                stage_results = self.joint_phase2_open(
                    batch_size=batch_size, val_batch_size=val_batch_size, eval_only=eval_only,
                    warm_start_from=getattr(self.args, "warm_start_from", None),
                )
                results[stage] = stage_results
                self._mark_stage_completed(stage_results.get("_stage_dir", stage), stage_results)

            elif stage == "joint_phase2_downstream":
                stage_results = self.joint_phase2_downstream(
                    batch_size=batch_size, val_batch_size=val_batch_size, eval_only=eval_only,
                    warm_start_from=getattr(self.args, "warm_start_from", None),
                )
                results[stage] = stage_results
                self._mark_stage_completed(stage_results.get("_stage_dir", stage), stage_results)

            elif stage == "stage2_all":
                stage_results = self.stage2_all(
                    batch_size=batch_size, val_batch_size=val_batch_size, eval_only=eval_only
                )
                results[stage] = stage_results
                self._mark_stage_completed(stage_results.get("_stage_dir", stage), stage_results)
            else:
                if self.rank == 0:
                    print(f"⚠️  Unknown stage: {stage}, skipping...")

            # Synchronize all ranks after completing each stage
            if dist.is_initialized():
                dist.barrier()

        # Save overall results only on rank 0
        if self.rank == 0:
            overall_results_file = os.path.join(
                self.results_dir, "curriculum_results.json"
            )
            with open(overall_results_file, "w") as f:
                json.dump(results, f, indent=2)

            print(f"\n🎉 Curriculum Learning Complete!")
            print(f"📁 All results saved to: {self.results_dir}/")
            print(f"📊 Overall results: {overall_results_file}")

        return results

    def _should_use_distributed(self) -> bool:
        """Check if distributed training should be used."""
        return ("WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1) or (
            "LOCAL_RANK" in os.environ and int(os.environ["LOCAL_RANK"]) >= 0
        )

    def _init_distributed(self):
        """Initialize distributed training."""
        if "WORLD_SIZE" in os.environ:
            self.world_size = int(os.environ["WORLD_SIZE"])
        if "RANK" in os.environ:
            self.rank = int(os.environ["RANK"])
        elif "LOCAL_RANK" in os.environ:
            self.rank = int(os.environ["LOCAL_RANK"])
        # Initialize process group
        dist.init_process_group(
            backend=self.dist_backend,
            init_method=self.dist_url,
            world_size=self.world_size,
            rank=self.rank,
            timeout=datetime.timedelta(minutes=10),
        )

        # Set device for this process
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)

        if self.rank == 0:
            print(f"Initialized distributed training with {self.world_size} GPUs")

    def _is_stage_completed(self, stage: str) -> bool:
        """Check if a stage is completed by verifying both training and evaluation were successful."""
        if stage == "phase1_mse_then_ce":
            return self._is_stage_completed("phase1_tsqa_m4_ce")

        metrics_file = os.path.join(self._stage_dir(stage), "results", "metrics.json")

        if not os.path.exists(metrics_file):
            return False

        try:
            with open(metrics_file, "r") as f:
                metrics = json.load(f)

            # Check if the completion flag exists
            if not metrics.get("completed", False):
                return False

            # Check if evaluation was actually completed by looking for test_loss
            if "test_loss" not in metrics:
                return False

            # Check if test predictions file exists
            test_predictions_file = os.path.join(self._stage_dir(stage), "results", "test_predictions.jsonl"
            )

            if not os.path.exists(test_predictions_file):
                return False

            return True

        except:
            return False

    def _mark_stage_completed(self, stage: str, metrics: Dict[str, Any]):
        """Mark a stage as completed by adding completion flag to metrics."""
        metrics["completed"] = True
        metrics["completion_epoch"] = metrics.get("epoch", "?")

        metrics_file = os.path.join(self._stage_dir(stage), "results", "metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        if self.rank == 0:
            print(f"✅ Stage {stage} marked as completed")

    def _get_model(self):
        """Get the underlying model (handles DDP wrapping)."""
        if hasattr(self.model, "module"):
            return self.model.module
        return self.model

    def _sync_gradients(self):
        """Average gradients across DDP ranks.

        REQUIRED: the training loop computes every loss through
        ``self._get_model()`` — i.e. the raw module, not the DDP wrapper.
        PyTorch only arms DDP's reducer inside ``DDP.forward``
        (``prepare_for_backward``), so when forward is bypassed its autograd
        hooks no-op and gradients are NEVER all-reduced. Without this call each
        rank trains an independent replica on its own DistributedSampler shard
        and only rank 0 is ever checkpointed — i.e. an N-GPU run silently
        degrades to a 1-GPU run on 1/N of the data per epoch. The ``no_sync()``
        context around accumulation steps is inert for the same reason.

        Called after ``scaler.unscale_()`` (so we average true, unscaled
        gradients regardless of per-rank scale factors) and before
        ``clip_grad_norm_`` (so clipping sees synchronized gradient norms).
        """
        if not dist.is_initialized() or self.world_size == 1:
            return
        params = [p for p in self._get_model().parameters() if p.requires_grad]
        if not params:
            return
        # Ranks can disagree on which params received gradients (task-homogeneous
        # batches, find_unused_parameters). Agree on the participating set first,
        # otherwise ranks issue mismatched collectives and hang. Params that no
        # rank touched keep grad=None so the optimizer skips them identically
        # everywhere (materialising zeros would let weight decay shrink them).
        has_grad = torch.tensor(
            [p.grad is not None for p in params], dtype=torch.uint8, device=self.device
        )
        dist.all_reduce(has_grad, op=dist.ReduceOp.MAX)
        active = [p for p, f in zip(params, has_grad.tolist()) if f]
        if not active:
            return
        for p in active:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(self.world_size)

    def _checkpoint_exists(self, stage: str) -> bool:
        """Check if a checkpoint exists for a specific stage."""
        checkpoint_path = os.path.join(self._stage_dir(stage), "checkpoints", "best_model.pt"
        )
        return os.path.exists(checkpoint_path)


    def _unfreeze_llm_if_needed(self, stage_name: str):
        """Unfreeze the entire LLM for full-fine-tune stages.

        For stages in :data:`FULL_LLM_FINETUNE_STAGES`, flip ``requires_grad``
        back to True on every LLM parameter. The optimizer then picks them up
        via the LLM param group constructed in :meth:`_get_optimizer`.

        Mutually exclusive with LoRA — full FT stages should not be in
        :data:`stages_with_lora`.
        """
        if self.model_type not in ["OpenTSLMSP", "OpenTSLMSPFixPos", "OpenTSLMSPFixPos2D", "OpenTSLMSPFixPos2DTimerXL"]:
            return
        if not _is_full_ft(stage_name):
            return
        model = self._get_model()
        n_unfrozen = 0
        for p in model.llm.parameters():
            if not p.requires_grad:
                p.requires_grad = True
            n_unfrozen += p.numel()
        if self.rank == 0:
            print(f"🔓 {stage_name}: unfroze full LLM ({n_unfrozen:,} parameters) for full fine-tuning.")
        # DDP was constructed with the LLM frozen → no AllReduce hooks for
        # those params. Re-wrap so gradients sync across ranks.
        if self.world_size > 1:
            if self.rank == 0:
                print("🔄 Re-wrapping model with DDP to register newly-trainable LLM parameters...")
            self.model = DDP(
                model,
                device_ids=[self.local_rank] if torch.cuda.is_available() else None,
                find_unused_parameters=True,
            )

    def _enable_lora_if_needed(self, stage_name: str, ckpt_stage: str = None):
        """Enable LoRA for OpenTSLMSP models in stages after stage2.

        No-ops when the stage is a full fine-tune: the two are mutually
        exclusive, and joint_phase2_downstream is in stages_with_lora, so
        without this guard the full-FT arm would silently also grow an adapter.

        ckpt_stage is the directory this run actually writes checkpoints to. It
        differs from stage_name whenever --warm_start_from is used, because
        run_stage() builds stage_dir = f"{stage_name}_from_{warm_start_from}"
        (see :1863-1867) while keeping stage_name for logic/set-membership.
        The resume-rank recovery below MUST look in the real checkpoint dir:
        reading _stage_dir(stage_name) finds nothing, silently falls back to the
        stage default rank, and then _load_checkpoint tries to copy the saved
        adapter into a differently-shaped one and raises. That is invisible for
        any run whose adapter rank happens to equal the stage default (which is
        why the r=4 270m run never hit it) and fatal on the first restart for
        the gemma-3-1b run, whose phase-1 adapter is r=8/alpha=16.
        """
        if self.model_type not in ["OpenTSLMSP", "OpenTSLMSPFixPos", "OpenTSLMSPFixPos2D", "OpenTSLMSPFixPos2DTimerXL"]:
            return  # LoRA only for OpenTSLMSP
        if _is_full_ft(stage_name):
            if self.rank == 0:
                print(f"🔓 {stage_name}: full LLM fine-tune requested — skipping LoRA "
                      f"(the two are mutually exclusive)")
            return

        # Get the underlying model (handles DDP wrapping)
        model = self._get_model()

        # Enable LoRA for stages after stage2_captioning.
        # Each entry: stage_name -> (lora_r, lora_alpha)
        # joint_phase1_extend_lorallm uses a small rank to keep the LLM close
        # to its pretrained weights while still allowing slight adaptation.
        stages_with_lora = {
            "stage3_cot":               (16, 32),
            "stage4_sleep_cot":         (16, 32),
            "stage5_ecg_cot":           (16, 32),
            "stage2_all":               (16, 32),
            "joint_phase1_extend_lorallm":       (4, 8),
            "phase1_chronos_mse_lora":           (4, 8),
            "joint_phase1_extend_balanced_lora": (4, 8),
            "joint_phase2":             (16, 32),
            "joint_phase2_open":        (16, 32),
            "joint_phase2_downstream":  (4, 8),
        }

        # Env override: enable LoRA on a stage not in the map above (e.g. the
        # text-only phase-1 joint_phase1_tsqa_m4). Set FORCE_STAGE_LORA="r,alpha".
        # No env var set -> behaviour is identical to before (fully backward-compatible).
        _force = os.environ.get("FORCE_STAGE_LORA")
        if _force and stage_name not in stages_with_lora:
            try:
                _r, _a = (int(x) for x in _force.split(","))
                stages_with_lora[stage_name] = (_r, _a)
            except Exception:
                pass

        if stage_name in stages_with_lora:
            if not getattr(model, "lora_enabled", False):
                lora_r, lora_alpha = stages_with_lora[stage_name]
                # On RESUME the stage's own checkpoint dictates the adapter shape:
                # _load_checkpoint() copies its lora_state into whatever adapter we
                # create here, so creating one at the stage default when the saved
                # adapter has a different rank raises and kills every restart
                # (e.g. warm-starting from the official OpenTSLM ckpts, which carry
                # r=16, into joint_phase2_downstream, whose default is r=4).
                # Read the rank back from the checkpoint when one exists. Runs whose
                # saved rank already equals the stage default are unaffected.
                _own_ckpt = os.path.join(
                    self._stage_dir(ckpt_stage or stage_name), "checkpoints", "best_model.pt"
                )
                if os.path.exists(_own_ckpt):
                    try:
                        _meta = torch.load(_own_ckpt, map_location="cpu", weights_only=False)
                        if _meta.get("lora_enabled", False):
                            _lcfg = _meta.get("lora_config")
                            _pcfg = next(iter(_lcfg.values())) if isinstance(_lcfg, dict) and _lcfg else _lcfg
                            _ckpt_r = int(getattr(_pcfg, "r", lora_r))
                            _ckpt_a = int(getattr(_pcfg, "lora_alpha", lora_alpha))
                            if (_ckpt_r, _ckpt_a) != (lora_r, lora_alpha):
                                if self.rank == 0:
                                    print(f"🔧 LoRA rank from existing checkpoint: "
                                          f"r={_ckpt_r}, alpha={_ckpt_a} "
                                          f"(stage default r={lora_r}, alpha={lora_alpha}) "
                                          f"— using the checkpoint's so resume works")
                                lora_r, lora_alpha = _ckpt_r, _ckpt_a
                        del _meta
                    except Exception as _e:
                        if self.rank == 0:
                            print(f"⚠️  could not read LoRA config from {_own_ckpt}: {_e}")
                if self.rank == 0:
                    print(f"🔧 Enabling LoRA for {stage_name} (r={lora_r}, alpha={lora_alpha})")
                try:
                    model.enable_lora(lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0)
                    if self.rank == 0:
                        print(f"✅ LoRA enabled for {stage_name}")
                    # LoRA params were added AFTER DDP was constructed, so DDP has no
                    # AllReduce hooks for them. Re-wrap with DDP so that:
                    # (1) lora_A/lora_B are broadcast from rank 0 to all ranks
                    #     (each rank otherwise has a different random lora_A), and
                    # (2) LoRA gradient hooks are registered so grads are AllReduced.
                    if self.world_size > 1:
                        if self.rank == 0:
                            print("🔄 Re-wrapping model with DDP to register LoRA parameters...")
                        self.model = DDP(
                            model,
                            device_ids=[self.local_rank] if torch.cuda.is_available() else None,
                            find_unused_parameters=True,
                        )
                        if self.rank == 0:
                            print("✅ DDP re-wrap complete — LoRA params will be AllReduced.")
                except Exception as e:
                    if self.rank == 0:
                        print(f"❌ Failed to enable LoRA for {stage_name}: {e}")
                        print("   Continuing without LoRA...")
            else:
                if self.rank == 0:
                    print(f"✅ LoRA already enabled for {stage_name}")
        else:
            if self.rank == 0:
                if stage_name in ["stage1_mcq", "stage2_captioning"]:
                    print(
                        f"ℹ️  LoRA disabled for {stage_name} (only enabled for stages 3+)"
                    )
                else:
                    print(f"ℹ️  LoRA not configured for {stage_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Curriculum Learning for OpenTSLM Models"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["OpenTSLMSP", "OpenTSLMSP_MultiVariate", "OpenTSLMFlamingo", "OpenTSLMFlamingoFixPos2D", "OpenTSLMSPFixPos", "OpenTSLMSPFixPos2D", "OpenTSLMSPFixPos2DTimerXL", "OpenTSLMSPMomentSoftPrompt"],
        required=True,
        help="Model type to train",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=AVAILABLE_STAGES,
        default=CURRICULUM_STAGES,
        help="Stages to run (default: all stages)",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (cuda, mps, cpu)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for training (default: use value from model_config.py)",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=None,
        help="Batch size for validation (default: same as train batch size)",
    )

    # Evaluation arguments
    parser.add_argument(
        "--eval_only",
        default=False,
        action="store_true",
        help="Skip training and only run evaluation (requires existing checkpoint)",
    )

    # Model-specific arguments
    parser.add_argument(
        "--llm_id",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="LLM model ID for OpenTSLMFlamingo (e.g., 'google/medgemma-2b', 'meta-llama/Llama-3.2-1B')",
    )

    # Distributed training arguments
    parser.add_argument(
        "--gradient_checkpointing",
        default=False,
        action="store_true",
        help="Enable gradient checkpointing",
    )
    parser.add_argument(
        "--dist_url",
        default="env://",
        type=str,
        help="URL used to set up distributed training",
    )
    parser.add_argument(
        "--dist_backend", default="nccl", type=str, help="Distributed backend"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=int(os.environ.get("LOCAL_RANK", 0)),
        help="Local GPU rank",
    )

    # Logging arguments
    parser.add_argument(
        "--verbose", default=False, action="store_true", help="Enable verbose logging"
    )

    # Number of training epochs
    parser.add_argument(
        "--num_epochs", required=True, type=int, help="number of training epoch"
    )

    parser.add_argument(
        "--pe_type", required=False
    )

    parser.add_argument(
        "--use_only_time_pe", default=False, action="store_true", help="Use only time positional encoding (no channel encoding)"
    )

    parser.add_argument(
        "--use_only_channel_pe", default=False, action="store_true", help="Use only channel positional encoding (no time encoding)"
    )

    parser.add_argument(
        "--pe_base_t", type=float, default=10000.0,
        help="Sin-cos PE base for the TIME axis. Standard Transformer default "
             "is 10000 (good for positions up to ~10k). Lower values give the "
             "low-frequency PE dims a faster cycle, which helps when t_max is "
             "small relative to 10000 (e.g. t_max=64 patches → base ~10-100)."
    )
    parser.add_argument(
        "--pe_base_c", type=float, default=10000.0,
        help="Sin-cos PE base for the CHANNEL axis. Default 10000 (matches "
             "the time axis). Channel indices typically range over only "
             "~5-15 values, so a much smaller base (e.g. 10) gives visibly "
             "distinct sin-cos signatures per channel. Training scripts may "
             "override this explicitly via --pe_base_c <value>."
    )

    parser.add_argument(
        "--use_freq_embed", default=False, action="store_true",
        help="Augment patch embedding with a frequency-domain branch: each patch is "
             "transformed via rfft, embedded independently, then concatenated with the "
             "time-domain patch embedding and projected back to transformer_input_dim.",
    )

    parser.add_argument(
        "--freq_mode", default="linear", choices=["linear", "spectral"],
        help="How --use_freq_embed encodes each patch spectrum. 'linear' (default, "
             "legacy) uses concat(real, imag) through a bias-free Linear -- because "
             "rfft is linear and no nonlinearity intervenes before the fusion, this is "
             "an EXACT linear reparameterisation of the time-domain patch embedding and "
             "adds parameters but no expressivity. 'spectral' uses "
             "concat(magnitude, phase) followed by GELU, which is genuinely nonlinear "
             "in the input. Parameter shapes are identical in both modes, so a "
             "checkpoint trained under one mode will load silently under the other and "
             "be misinterpreted -- keep this flag consistent across phases.",
    )

    parser.add_argument(
        '--is_pretraining', default=False, required=False, action='store_true', help='Whether to run in pretraining mode (only for OpenTSLMSP)'
    )

    parser.add_argument(
        '--kl_beta',
        type=float,
        default=0.1,  # or whatever default you want
        help='Beta parameter for KL divergence weighting in VAE-style training'
    )

    parser.add_argument(
        '--wandb_run_name',
        type=str,
        default=None,
        help='Override the wandb run name. If omitted, defaults to '
             '"pretrain-<stages>_kl<kl_beta>". Set this per-script so '
             'concurrent training runs land in distinct wandb runs.',
    )
    parser.add_argument(
        '--wandb_project',
        type=str,
        default=None,
        help='Override the wandb project name. If omitted, defaults to '
             '"opentslm-pretrain-<stages>". Useful when several scripts '
             'share the same --stages but should be grouped separately.',
    )

    parser.add_argument(
        '--save_every_n_steps',
        type=int,
        default=10000,
        help='Save a periodic checkpoint every N training steps (0 = disabled). '
             'Saves to periodic_checkpoint.pt so progress is never lost if job hangs.'
    )

    parser.add_argument(
        '--eval_every_n_iters',
        type=int,
        default=500,
        help='Run validation (metrics + wandb logging + best-checkpoint save) '
             'every N optimizer steps, instead of only once per epoch. A tail '
             'validation also runs at the end of each epoch. Early stopping '
             'stays epoch-level. Default 500.'
    )

    parser.add_argument(
        '--eval_max_samples_per_task',
        type=int,
        default=1000,
        help='Cap the periodic mid-epoch validation to at most N samples per '
             'task (chronos / tsqa / m4) — a global count, split across ranks. '
             'Keeps each periodic eval fast and bounded regardless of val-set '
             'size. The epoch-tail validation always runs the FULL val set. '
             '0 or negative = no cap. Default 1000.'
    )

    parser.add_argument(
        '--max_samples_per_source',
        type=int,
        default=None,
        help='Cap each data source to this many samples per split (useful for quick smoke-tests, e.g. 100).'
    )
    parser.add_argument(
        "--data_seed",
        type=int,
        default=42,
        help="Seed for dataset-side sampling/shuffling. Must match across DDP ranks.",
    )
    parser.add_argument(
        "--mse_alpha",
        type=float,
        default=1.0,
        help="Fixed weight for MSE loss relative to CE loss in joint training. Ignored when --dynamic_loss_weighting is set.",
    )
    parser.add_argument(
        "--detach_llm_for_mse",
        action="store_true",
        default=False,
        help="Force the joint Phase-1 chronos MSE path to read from the projector "
             "output instead of the LLM hidden states, so MSE gradients update "
             "only encoder+projector+decoder (not the LLM). The LLM is still "
             "trained by CE (TSQA+M4). Adds to JOINT_PHASE1_DETACH_LLM_MSE_STAGES "
             "at runtime — use this for ablations or to enable on stages not in "
             "the hard-coded set.",
    )
    parser.add_argument(
        "--dynamic_loss_weighting",
        action="store_true",
        default=False,
        help="Dynamically rescale MSE loss to match CE magnitude each step, so all tasks contribute equally to encoder gradients.",
    )

    parser.add_argument(
        "--patch_size",
        type=int,
        default=None,
        help="Patch size for tokenization (default: use value from model_config.py)",
    )
    parser.add_argument(
        "--enc_dec_type",
        type=str,
        default="transformer",
        choices=["transformer", "mlp"],
        help="Backbone used for the time-series encoder and decoder. "
             "'transformer' keeps the original TransformerCNNEncoder2D + "
             "TransformerCNNDecoder2D; 'mlp' replaces both with position-wise "
             "2-layer MLPs (same 2D sin-cos PE on the encoder side).",
    )
    parser.add_argument(
        "--chronos_window_stride",
        type=int,
        default=None,
        help="If set, enumerate Chronos training windows at this stride "
             "(in timesteps) so that each window becomes its own training "
             "sample. Useful for tiny corpora where a single HF row otherwise "
             "yields only one sample per epoch. Val/test keep the deterministic "
             "last-window behaviour.",
    )
    parser.add_argument(
        "--chronos_window_sampling",
        type=str,
        default="stride",
        choices=["stride", "random"],
        help="How chronos training windows are drawn from each row. "
             "'stride' (default) enumerates a fixed grid at --chronos_window_stride "
             "(deterministic). 'random' emits N copies of an anchor-less stub per "
             "row and picks a fresh uniform random anchor on each __getitem__ "
             "(Chronos-style; matches `ExpectedNumInstanceSampler` semantics). "
             "Val/test stay on the deterministic last-window logic regardless.",
    )
    parser.add_argument(
        "--chronos_random_samples_per_row",
        type=int,
        default=None,
        help="Number of stub copies (= random window draws) per training row "
             "when --chronos_window_sampling=random. Larger N -> more steps per "
             "epoch and more variety per row. Default 1 (Chronos-style).",
    )
    # ─── Per-source row cap (debug + small-corpus runs) ─────────────────────
    # NOTE: --max_samples_per_source is already declared further up; --debug
    # below fills it in when the user hasn't set it explicitly.
    parser.add_argument(
        "--synthetic_max_rows_per_file",
        type=int,
        default=None,
        help="Optional cap applied at LOAD time inside _load_synthetic_arrow. "
             "Limits each unified_*.arrow file to its first N rows so the arrow "
             "read + transpose finishes quickly. Useful with --debug. Default "
             "None = read the whole file (full corpus = ~844k rows).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Tiny-corpus debug mode for fast iteration. Sets sensible defaults "
             "for: --max_samples_per_source (200), "
             "--synthetic_max_rows_per_file (500), "
             "--chronos_random_samples_per_row (2). User-set values take "
             "precedence; this only fills in the ones that are still None.",
    )
    parser.add_argument(
        "--build_cache_only",
        action="store_true",
        default=False,
        help="Construct the train/val/test datasets to populate the on-disk "
             "Chronos splits + stub caches, then exit before any training. "
             "Intended for cheap CPU-only sbatch jobs that pre-build the data "
             "caches so subsequent GPU jobs start training in seconds.",
    )
    parser.add_argument(
        "--chronos_univariate",
        action="store_true",
        help="Keep registry Chronos datasets strictly univariate: skip "
             "row-stacking into a multivariate series, and explode multi-key "
             "datasets into one univariate [T, 1] series per key. Use a fresh "
             "CHRONOS_PREPROCESSED_SPLITS_DIR when toggling this (the built "
             "split cache is keyed by directory, not by this flag).",
    )
    parser.add_argument(
        "--synthetic_data_dir",
        type=str,
        default=None,
        help="Directory of unified_*.arrow multivariate synthetic datasets "
             "(see synthetic_data/unified/scripts/collect_synthetic.py). When "
             "set, every unified_*.arrow file is loaded as a multivariate "
             "[T, C] source and concatenated into the Phase-1 Chronos corpus. "
             "Use a fresh CHRONOS_PREPROCESSED_SPLITS_DIR when toggling this.",
    )
    # ─── Per-task sampler weights (joint_phase1_extend_balanced) ────────────
    # Source-aware task weights for the TaskWeightedDistributedBatchSampler.
    # In the joint Phase-1 extend stage, batches are drawn from four buckets:
    # chronos_uv, synthetic, tsqa, m4. Only ratios matter; the sampler
    # normalises internally. Default 1:2:1:1 oversamples synthetic relative to
    # the (tiny) chronos_uv pool while keeping TSQA/M4 at the same weight.
    parser.add_argument(
        "--task_weight_chronos_uv",
        type=float,
        default=1.0,
        help="Sampler weight for chronos_uv (univariate registry Chronos) "
             "batches in joint_phase1_extend_balanced. Default 1.0.",
    )
    parser.add_argument(
        "--task_weight_synthetic",
        type=float,
        default=2.0,
        help="Sampler weight for synthetic MV batches in "
             "joint_phase1_extend_balanced. Default 2.0.",
    )
    parser.add_argument(
        "--task_weight_tsqa",
        type=float,
        default=1.0,
        help="Sampler weight for TSQA batches in joint_phase1_extend_balanced. "
             "Default 1.0.",
    )
    parser.add_argument(
        "--task_weight_m4",
        type=float,
        default=1.0,
        help="Sampler weight for M4 batches in joint_phase1_extend_balanced. "
             "Default 1.0.",
    )
    parser.add_argument(
        "--mlp_enc_hidden_dim",
        type=int,
        default=None,
        help="MLP encoder internal width (transformer_input_dim). "
             "Default 768. Reduce to shrink encoder capacity.",
    )
    parser.add_argument(
        "--mlp_enc_ff_dim",
        type=int,
        default=None,
        help="MLP encoder intermediate width (hidden dim of the 2-layer MLP). "
             "Default 3072.",
    )
    parser.add_argument(
        "--mlp_dec_hidden_dim",
        type=int,
        default=None,
        help="MLP decoder internal width (decoder_dim). "
             "Default 512. Reduce to shrink decoder capacity.",
    )
    parser.add_argument(
        "--mlp_dec_ff_dim",
        type=int,
        default=None,
        help="MLP decoder intermediate width (hidden dim of the 2-layer MLP). "
             "Default 2048.",
    )
    parser.add_argument(
        "--trans_enc_dim",
        type=int,
        default=None,
        help="Transformer encoder d_model (transformer_input_dim). "
             "Default 768. Reduce to shrink encoder capacity.",
    )
    parser.add_argument(
        "--trans_enc_num_layers",
        type=int,
        default=None,
        help="Transformer encoder depth (num_layers). Default 24.",
    )
    parser.add_argument(
        "--trans_enc_num_heads",
        type=int,
        default=None,
        help="Transformer encoder attention heads (nhead). Default 12.",
    )
    parser.add_argument(
        "--trans_enc_ff_dim",
        type=int,
        default=None,
        help="Transformer encoder feed-forward dim (dim_feedforward). Default 3072.",
    )
    parser.add_argument(
        "--trans_dec_dim",
        type=int,
        default=None,
        help="Transformer decoder d_model (decoder_dim). Default 512.",
    )
    parser.add_argument(
        "--trans_dec_num_layers",
        type=int,
        default=None,
        help="Transformer decoder depth (num_layers). Default 4.",
    )
    parser.add_argument(
        "--trans_dec_num_heads",
        type=int,
        default=None,
        help="Transformer decoder attention heads (nhead). Default 8.",
    )
    parser.add_argument(
        "--trans_dec_ff_dim",
        type=int,
        default=None,
        help="Transformer decoder feed-forward dim (dim_feedforward). Default 2048.",
    )
    parser.add_argument(
        "--phase1_min_history_patches",
        type=int,
        default=8,
        help="Minimum number of history patches required for phase1_chronos_mse windows. "
             "Shorter Chronos series are excluded from that stage.",
    )

    parser.add_argument(
        "--warm_start_from",
        type=str,
        default=None,
        help="Stage name to warm-start from before training (e.g. joint_phase1). "
             "Loads weights from that stage's checkpoint if no checkpoint exists for the current stage.",
    )

    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps before an optimizer update. "
             "Effective batch size = batch_size * grad_accum_steps * world_size.",
    )

    parser.add_argument(
        "--dataloader_workers",
        type=int,
        default=-1,
        help="Number of DataLoader worker processes for async data loading. "
             "-1 (default) auto-derives from SLURM_CPUS_PER_TASK / LOCAL_WORLD_SIZE, "
             "reserving one core per rank for the main process and capping at 16. "
             "Set to 0 to disable (single-process, useful for debugging).",
    )
    parser.add_argument(
        "--llm_dtype",
        type=str,
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32"],
        help="LLM weight dtype. 'auto' (default) picks bf16 on bf16-capable "
             "GPUs (A100+) and fp16 elsewhere (V100), matching the AMP dtype "
             "so the LLM forward doesn't pay an implicit upcast on every op. "
             "Override only if you need a specific precision.",
    )
    parser.add_argument(
        "--use_processor",
        action="store_true",
        help="Use OpenTSLMProcessor (HF-style) as the DataLoader collate. "
             "Tokenization and per-sample TS prep run in dataloader workers, "
             "and pad_and_apply_batch skips its CPU-bound steps on the GPU rank.",
    )
    parser.add_argument(
        "--train_log_every_n_steps",
        type=int,
        default=1,
        help="Log training metrics to Weights & Biases every N optimizer steps.",
    )
    parser.add_argument(
        "--joint_monitor_every_n_epochs",
        type=int,
        default=1,
        help="For joint stages, compute per-task validation monitoring every N epochs.",
    )
    parser.add_argument(
        "--joint_generate_every_n_epochs",
        type=int,
        default=5,
        help="For joint stages, generate qualitative validation samples every N epochs. Set to 0 to disable.",
    )
    parser.add_argument(
        "--windows_per_col_train",
        type=int,
        default=None,
        help="Number of random training windows per (column, horizon) pair in Phase2DownstreamDataset. "
             "Defaults to the dataset's built-in default (100). Higher values increase forecast coverage.",
    )
    parser.add_argument(
        "--max_tokens_per_batch",
        type=int,
        default=None,
        help="Token budget for dynamic batch sizing in joint_phase2_downstream. "
             "Each batch contains at most this many total (padded) tokens: "
             "B × max_len_in_batch <= budget. Short samples are packed together "
             "(B up to --batch_size); long samples run alone (B=1). "
             "Recommended: 3200 for V100-32GB. Default None = fixed batch_size.",
    )

    args = parser.parse_args()

    # ─── --debug umbrella ────────────────────────────────────────────────────
    # Fast-iteration mode: small data subset, cheap arrow loads, few anchors.
    # Only fills in args that the user didn't explicitly set (None today), so
    # this composes safely with explicit overrides.
    if args.debug:
        _debug_defaults = {
            "max_samples_per_source": 200,
            "synthetic_max_rows_per_file": 500,
            "chronos_random_samples_per_row": 2,
        }
        _filled = []
        for k, v in _debug_defaults.items():
            if getattr(args, k, None) is None:
                setattr(args, k, v)
                _filled.append(f"{k}={v}")
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                "🐛 DEBUG MODE — small-corpus defaults applied: "
                + (", ".join(_filled) if _filled else "all overridden by user")
            )

    # Auto-resolve --dataloader_workers if left at the -1 sentinel. We pick
    # SLURM_CPUS_PER_TASK / LOCAL_WORLD_SIZE - 1 (reserve one core per rank
    # for the main process), capped at 16. Falls back to os.cpu_count() and
    # the visible GPU count when SLURM/torchrun env vars are missing.
    if args.dataloader_workers < 0:
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        cpus = int(slurm_cpus) if slurm_cpus else (os.cpu_count() or 8)
        local_world_env = os.environ.get("LOCAL_WORLD_SIZE")
        if local_world_env is not None:
            local_world = max(1, int(local_world_env))
        else:
            local_world = max(1, torch.cuda.device_count() if torch.cuda.is_available() else 1)
        per_rank = max(1, cpus // local_world)
        args.dataloader_workers = max(2, min(16, per_rank - 1))
        # Print only on rank 0 to avoid duplicate noise across torchrun children.
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                f"[dataloader] auto-selected workers={args.dataloader_workers} "
                f"(SLURM_CPUS_PER_TASK={slurm_cpus}, cpus={cpus}, "
                f"local_world={local_world})"
            )

    # Override PATCH_SIZE if provided via CLI
    if args.patch_size is not None:
        global PATCH_SIZE
        PATCH_SIZE = args.patch_size

    # Set up global logging
    set_global_verbose(args.verbose)
    logger = get_logger(verbose=args.verbose)

    # Initialize trainer
    trainer = CurriculumTrainer(
        args.model,
        args.device,
        gradient_checkpointing=args.gradient_checkpointing,
        dist_url=args.dist_url,
        dist_backend=args.dist_backend,
        local_rank=args.local_rank,
        llm_id=args.llm_id,
        args=args
    )

    # Run curriculum
    results = trainer.run_curriculum(
        args.stages,
        args.batch_size,
        args.val_batch_size,
        args.eval_only,
    )

    # Print summary
    logger.info("Final Results Summary:")
    logger.info("=" * 40)
    for stage, metrics in results.items():
        logger.info(f"{stage.upper()}:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric}: {value:.4f}")
            else:
                logger.info(f"  {metric}: {value}")


if __name__ == "__main__":
    main()
