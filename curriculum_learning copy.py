# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT
import sys
sys.path.append("./src")  # Ensure src is in the path for imports


import json
import os
import argparse
import contextlib
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
import torch
import torch.distributed as dist
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
    "phase1_chronos_joint_mse",
    "phase1_chronos_seq2seq",
    "phase1_chronos_seq2seq_separate_mse",
    "joint_phase1_extended_separate_mse",
    "joint_phase1_extend_balanced",
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
    "phase1_chronos_seq2seq",
    "phase1_chronos_seq2seq_separate_mse",
}
PHASE1_CE_ONLY_STAGES = {"phase1_tsqa_m4_ce"}
JOINT_PHASE1_WITH_M4_STAGES = {
    "joint_phase1_extend",
    "joint_phase1_extend_lorallm",
    "joint_phase1_extended_separate_mse",
    "joint_phase1_extend_balanced",
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
        patch_suffix = f"_patch{PATCH_SIZE}"
        weight_suffix = "_dynw" if getattr(self.args, "dynamic_loss_weighting", False) else ""
        # Append an enc/dec backbone suffix so MLP runs live in their own tree
        # and don't clobber transformer checkpoints (and vice versa).
        _enc_dec_type = str(getattr(self.args, "enc_dec_type", "transformer")).lower()
        if _enc_dec_type != "transformer":
            patch_suffix = f"{patch_suffix}_{_enc_dec_type}"
        if self.args.pe_type is None:
            self.results_dir = os.path.join(f"results/base_opentslm", self.llm_id_safe, self.model_type)
        else:
            if not self.args.use_freq_embed:
                if self.args.use_only_time_pe:
                    self.results_dir = os.path.join(f"results/{self.args.pe_type}_time_only{patch_suffix}{weight_suffix}", self.llm_id_safe, self.model_type)
                elif self.args.use_only_channel_pe:
                    self.results_dir = os.path.join(f"results/{self.args.pe_type}_channel_only{patch_suffix}{weight_suffix}", self.llm_id_safe, self.model_type)
                else:
                    self.results_dir = os.path.join(f"results/{self.args.pe_type}{patch_suffix}{weight_suffix}", self.llm_id_safe, self.model_type)
            else:
                if self.args.use_only_time_pe:
                    self.results_dir = os.path.join(f"results/{self.args.pe_type}_time_only{patch_suffix}{weight_suffix}_seed_{self.args.data_seed}_freq", self.llm_id_safe, self.model_type)
                elif self.args.use_only_channel_pe:
                    self.results_dir = os.path.join(f"results/{self.args.pe_type}_channel_only{patch_suffix}{weight_suffix}_seed_{self.args.data_seed}_freq", self.llm_id_safe, self.model_type)
                else:
                    self.results_dir = os.path.join(f"results/{self.args.pe_type}{patch_suffix}{weight_suffix}_seed_{self.args.data_seed}_freq", self.llm_id_safe, self.model_type)
        self._create_results_dir()

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
        """Create the results directory structure."""
        os.makedirs(self.results_dir, exist_ok=True)
        model_dir = self.results_dir
        os.makedirs(model_dir, exist_ok=True)

        # Create stage directories based on global configuration
        for stage in CURRICULUM_STAGES:
            stage_dir = os.path.join(model_dir, stage)
                
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
            # Both joint_phase2 variants freeze encoder + projector so CE gradients
            # never reach them; only decoder head and LoRA train from CE/MSE.
            freeze_enc_and_proj = (stage_name in JOINT_PHASE2_STAGES)

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
                    # Always use lr_base for LoRA in joint-phase2 stages; fall back to projector_lr otherwise
                    lora_lr = (lr_base if lr_base is not None else projector_lr) if stage_name in JOINT_PHASE2_STAGES \
                              else (lr_base if (self.model_type == "PlainTextLM" and lr_base is not None) else projector_lr)
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
        _collate = partial(
            extend_time_series_to_match_patch_size_and_aggregate, patch_size=patch_size
        )

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
        checkpoint_dir = os.path.join(self.results_dir, stage, "checkpoints")

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

        checkpoint_dir = os.path.join(self.results_dir, stage, "checkpoints")
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

        checkpoint_dir = os.path.join(self.results_dir, stage, "checkpoints")
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

        checkpoint_dir = os.path.join(self.results_dir, stage, "checkpoints")
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
        checkpoint_path = os.path.join(
            self.results_dir, stage, "checkpoints", "best_model.pt"
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
            metrics_file = os.path.join(
                self.results_dir, previous_stage, "results", "metrics.json"
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
            checkpoint_path = os.path.join(
                self.results_dir, previous_stage, "checkpoints", "best_model.pt"
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

    def _load_weights_from_stage(self, stage_name: str) -> bool:
        """Load model weights from any stage checkpoint, without requiring metrics.

        Returns True if the checkpoint was found and loaded, False otherwise.
        """
        checkpoint_path = os.path.join(
            self.results_dir, stage_name, "checkpoints", "best_model.pt"
        )
        if not os.path.exists(checkpoint_path):
            if self.rank == 0:
                print(f"⚠️  No checkpoint found for stage '{stage_name}' at {checkpoint_path}")
            return False

        if self.rank == 0:
            print(f"📥 Loading weights from stage '{stage_name}': {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = self._get_model()

        if self.model_type in ["OpenTSLMSP", "OpenTSLMSPFixPos", "OpenTSLMSPFixPos2D", "OpenTSLMSPFixPos2DTimerXL"]:
            model.encoder.load_state_dict(checkpoint["encoder_state"])
            model.projector.load_state_dict(checkpoint["projector_state"])
            model.decoder.load_state_dict(checkpoint["decoder_state"])
            model.decoder_projector.load_state_dict(checkpoint["decoder_projector_state"])
            self._restore_llm_embeddings_from_checkpoint(model, checkpoint)
            model.load_lora_state_from_checkpoint(checkpoint, allow_missing=True)
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
        results_file_rank = os.path.join(
            self.results_dir,
            stage_name,
            "results",
            f"test_predictions_rank_{self.rank if dist.is_initialized() else 0}.jsonl",
        )
        final_results_file = os.path.join(
            self.results_dir, stage_name, "results", "test_predictions.jsonl"
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
                        part_file = os.path.join(
                            self.results_dir,
                            stage_name,
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
            metrics_file = os.path.join(
                self.results_dir, stage_name, "results", "metrics.json"
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
        test_predictions_file = os.path.join(
            self.results_dir, stage, "results", "test_predictions.jsonl"
        )
        metrics_file = os.path.join(self.results_dir, stage, "results", "metrics.json")

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
                f"Eval-only mode requires a checkpoint for {stage_dir}, but none found at {os.path.join(self.results_dir, stage_dir, 'checkpoints', 'best_model.pt')}"
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
            metrics_file = os.path.join(
                self.results_dir, stage_dir, "results", "metrics.json"
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

        # Enable LoRA if needed for this stage
        self._enable_lora_if_needed(stage_name)

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
            task_batch_sampler = TaskHomogeneousDistributedBatchSampler(
                train_dataset,
                batch_size=batch_size,
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
                collate_fn=partial(
                    extend_time_series_to_match_patch_size_and_aggregate, patch_size=PATCH_SIZE
                ),
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
                    collate_fn=partial(
                        extend_time_series_to_match_patch_size_and_aggregate, patch_size=PATCH_SIZE
                    ),
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
                    collate_fn=lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
                        batch, patch_size=PATCH_SIZE
                    ),
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
                wandb.init(
                    project=f"opentslm-pretrain-{self.args.stages}",
                    config=vars(self.args),
                    name=f"pretrain-{self.args.stages}_kl{self.args.kl_beta}",
                )

            # Global optimizer step counter — persists across epochs and accounts
            # for resumed training so wandb x-axis is monotonically correct.
            steps_per_epoch = max(1, len(train_loader) // grad_accum)
            global_optimizer_step = (start_epoch - 1) * steps_per_epoch

            for epoch in range(start_epoch, num_epochs + 1):
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
                _acc = {"total": 0.0, "ce": 0.0, "tsqa": 0.0, "m4": 0.0, "mse": 0.0,
                        "ntp": 0.0, "recon": 0.0, "next_patch": 0.0, "final_patch": 0.0}
                _acc_n = {"total": 0, "ce": 0, "tsqa": 0, "m4": 0, "mse": 0,
                          "ntp": 0, "recon": 0, "next_patch": 0, "final_patch": 0}

                for i, batch in enumerate(prog):
                    if _should_skip_empty_batch(batch, "train", i):
                        continue
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
                                    detach_llm_for_mse=(stage_name in JOINT_PHASE1_DETACH_LLM_MSE_STAGES),
                                )
                            elif stage_name in JOINT_PHASE2_STAGES:
                                loss, loss_ce, loss_mse = self._get_model().compute_joint_phase2_loss(
                                    batch,
                                    dynamic_weighting=getattr(self.args, "dynamic_loss_weighting", False),
                                )
                            else:
                                loss = self._get_model().compute_loss(batch)
                                loss_ce, loss_mse = None, None

                    _sync_ctx = (
                        self.model.no_sync()
                        if hasattr(self.model, "no_sync") and not is_accum_last
                        else contextlib.nullcontext()
                    )
                    with _sync_ctx:
                        scaler.scale(loss / grad_accum).backward()

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
                            dist.all_reduce(reduced_tensor, op=dist.ReduceOp.SUM)
                            reduced_tensor /= self.world_size
                        reduced_values = reduced_tensor.tolist()

                        avg_loss = reduced_values[0]
                        if self._uses_pretrain_loss(stage_name):
                            if stage_name in PHASE1_MSE_ONLY_STAGES:
                                avg_ntp, avg_recon, avg_final_patch = reduced_values[1:]
                                avg_next_patch = avg_recon
                            else:
                                avg_ntp, avg_recon = reduced_values[1:]
                        elif stage_name in ("stage2_all",):
                            avg_ce, avg_mse = reduced_values[1:]
                        elif stage_name in JOINT_PHASE1_STAGES:
                            avg_ce, avg_tsqa, avg_m4, avg_mse = reduced_values[1:]
                        elif stage_name in JOINT_PHASE2_STAGES:
                            avg_ce, avg_mse = reduced_values[1:]

                        # ── Accumulate per-task running sums ──────────────────
                        if self.rank == 0:
                            _acc["total"] += avg_loss; _acc_n["total"] += 1
                            if stage_name in PHASE1_MSE_ONLY_STAGES:
                                _acc["ntp"]   += avg_ntp;   _acc_n["ntp"]   += 1
                                _acc["recon"] += avg_recon; _acc_n["recon"] += 1
                                _acc["next_patch"] += avg_next_patch; _acc_n["next_patch"] += 1
                                _acc["final_patch"] += avg_final_patch; _acc_n["final_patch"] += 1
                            elif self._uses_pretrain_loss(stage_name):
                                _acc["ntp"]   += avg_ntp;   _acc_n["ntp"]   += 1
                                _acc["recon"] += avg_recon; _acc_n["recon"] += 1
                            elif stage_name in ("stage2_all",) + tuple(JOINT_PHASE1_STAGES) + tuple(JOINT_PHASE2_STAGES):
                                _acc["ce"] += avg_ce; _acc_n["ce"] += 1
                                # tsqa / m4 / mse are accumulated per sub-batch
                                # above (every batch, not just is_accum_last) to
                                # avoid missing them when the last sub-batch is a
                                # different task type. Only non-joint stages
                                # (stage2_all, JOINT_PHASE2) still use avg_mse here.
                                if stage_name not in JOINT_PHASE1_STAGES:
                                    if avg_mse > 0:
                                        _acc["mse"] += avg_mse; _acc_n["mse"] += 1

                        # ── Log window averages every train_log_every steps ───
                        if self.rank == 0 and optimizer_step % train_log_every == 0:
                            def _wavg(key):
                                n = _acc_n[key]
                                return _acc[key] / n if n > 0 else None

                            if self._uses_pretrain_loss(stage_name):
                                if stage_name in PHASE1_MSE_ONLY_STAGES:
                                    log_dict = {
                                        "pretrain/total_loss": _wavg("total"),
                                        "pretrain/next_patch_mse": _wavg("next_patch"),
                                        "pretrain/final_patch_mse": _wavg("final_patch"),
                                    }
                                else:
                                    log_dict = {
                                        "pretrain/total_loss": _wavg("total"),
                                        "pretrain/ntp_loss":   _wavg("ntp"),
                                        "pretrain/recon_loss": _wavg("recon"),
                                    }
                            elif stage_name == "stage2_all":
                                log_dict = {
                                    "train/loss":     _wavg("total"),
                                    "train/ce_loss":  _wavg("ce"),
                                    "train/mse_loss": _wavg("mse"),
                                }
                            elif stage_name in JOINT_PHASE1_STAGES:
                                log_dict = {
                                    "train/loss":      _wavg("total"),
                                    "train/ce_loss":   _wavg("ce"),
                                    "train/tsqa_loss": _wavg("tsqa"),
                                    "train/mse_loss":  _wavg("mse"),
                                }
                                if stage_name in JOINT_PHASE1_WITH_M4_STAGES:
                                    log_dict["train/m4_loss"] = _wavg("m4")
                            elif stage_name in JOINT_PHASE2_STAGES:
                                log_dict = {
                                    "train/loss":     _wavg("total"),
                                    "train/ce_loss":  _wavg("ce"),
                                    "train/mse_loss": _wavg("mse"),
                                }
                            else:
                                log_dict = {"train/loss": _wavg("total")}

                            # Drop None values (task never appeared in the window)
                            wandb.log(
                                {k: v for k, v in log_dict.items() if v is not None},
                                step=global_optimizer_step,
                            )

                            # Reset accumulators for the next window
                            for k in _acc:
                                _acc[k] = 0.0; _acc_n[k] = 0

                        # Handle gradient clipping for distributed training
                        scaler.unscale_(optimizer)
                        clip_grad_norm_(self._get_model().parameters(), GRAD_CLIP_NORM)

                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()

                        # Periodic mid-epoch checkpoint (skip last batch — end-of-epoch save handles it)
                        save_every = getattr(self.args, "save_every_n_steps", 0)
                        is_last_batch = (i == len(train_loader) - 1)
                        if save_every > 0 and (i + 1) % save_every == 0 and not is_last_batch:
                            self._save_periodic_checkpoint(stage_dir, epoch, i + 1, optimizer, scheduler)

                avg_train_loss = running_loss / max(1, num_train_steps)
                if self.rank == 0:
                    tqdm.write(f"Epoch {epoch} — train loss: {avg_train_loss:.4f}")

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
                monitor_batch = []
                self.model.eval()
                with torch.no_grad(), torch.amp.autocast('cuda', dtype=amp_dtype):
                    for j, batch in enumerate(tqdm(
                        val_loader,
                        desc=f"Validating {stage_name}",
                        disable=self.rank != 0,
                    )):
                        if _should_skip_empty_batch(batch, "validation", j):
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
                                    detach_llm_for_mse=(stage_name in JOINT_PHASE1_DETACH_LLM_MSE_STAGES),
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
                            elif stage_name in JOINT_PHASE2_STAGES:
                                loss, loss_ce, loss_mse = self._get_model().compute_joint_phase2_loss(batch)
                                val_loss     += loss.item()
                                val_mse_loss += loss_mse.item()
                                val_ce_loss  += loss_ce.item()
                            else:
                                val_loss += self._get_model().compute_loss(batch).item()
                        num_val_steps += 1

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
                ) = [float(v) for v in val_metrics.tolist()]
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
                    tqdm.write(f"Epoch {epoch} — best  loss: {best_val_loss:.4f}")
                    if self._uses_pretrain_loss(stage_name):
                        if stage_name in PHASE1_MSE_ONLY_STAGES:
                            wandb.log(
                                {
                                    "val/next_patch_mse": avg_val_next_patch_mse,
                                    "val/final_patch_mse": avg_val_final_patch_mse,
                                    "val/loss": avg_val_loss,
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
                            },
                            step=global_optimizer_step,
                        )
                    else:
                        wandb.log({"val/loss": avg_val_loss}, step=global_optimizer_step)

                # ── Phase-1/2 deep monitoring ────────────────────────────────
                if (
                    stage_name in JOINT_STAGES
                    and self.rank == 0
                    and monitor_batch
                    and epoch % joint_monitor_every == 0
                ):
                    per_task = self._get_model().compute_loss_per_task(monitor_batch)
                    wandb.log({f"val/{k}": v for k, v in per_task.items()}, step=global_optimizer_step)
                    tqdm.write(
                        "  Per-task val — " +
                        "  ".join(f"{k}={v:.4f}" for k, v in per_task.items())
                    )
                    if joint_generate_every > 0 and epoch % joint_generate_every == 0:
                        self._get_model().generate_phase1_samples(monitor_batch)

                # Save loss history for this epoch
                self._save_loss_history(stage_dir, epoch, avg_train_loss, avg_val_loss)

                # Early stopping - all ranks need to make the same decision
                is_joint = stage_name in JOINT_CE_MSE_STAGES

                if is_joint:
                    mse_improved = avg_val_mse_loss + 1e-4 < best_val_mse
                    ce_improved  = avg_val_ce_loss  + 1e-4 < best_val_ce
                    should_save  = avg_val_loss + 1e-4 < best_val_loss
                else:
                    should_save = avg_val_loss + 1e-4 < best_val_loss

                if dist.is_initialized():
                    save_tensor = torch.tensor(1 if should_save else 0, device=self.device)
                    dist.all_reduce(save_tensor, op=dist.ReduceOp.SUM)
                    should_save = save_tensor.item() > 0

                if should_save:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    checkpoint_extra = {"scaler_state": scaler.state_dict()}
                    if is_joint:
                        checkpoint_extra["joint_init_mse"] = _joint_init_mse
                        checkpoint_extra["joint_init_ce"] = _joint_init_ce
                    self._save_checkpoint(
                        stage_dir,
                        epoch,
                        avg_val_loss,
                        optimizer,
                        scheduler,
                        extra_state=checkpoint_extra,
                    )
                    if self.rank == 0:
                        tqdm.write("✔️  New best model saved.\n")
                else:
                    epochs_no_improve += 1
                    if self.rank == 0:
                        tqdm.write(f"No improvement for {epochs_no_improve}/{EARLY_STOP_PAT} epochs.\n")

                # For joint stages: also track per-task patience independently.
                # Early stop if EITHER MSE or CE has stagnated for EARLY_STOP_PAT epochs.
                if is_joint:
                    if mse_improved:
                        best_val_mse = avg_val_mse_loss
                        epochs_no_improve_mse = 0
                    else:
                        epochs_no_improve_mse += 1

                    if ce_improved:
                        best_val_ce = avg_val_ce_loss
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

                # Regular (non-joint) early stopping check
                if not is_joint and epochs_no_improve >= EARLY_STOP_PAT:
                    if self.rank == 0:
                        tqdm.write(f"\nEarly stopping triggered after {epoch} epochs.")
                        tqdm.write(f"Final stats: best_val_loss={best_val_loss:.4f}")
                    break

                # Synchronize best_val_loss and epochs_no_improve across all ranks
                if dist.is_initialized():
                    best_loss_tensor = torch.tensor(best_val_loss, device=self.device)
                    epochs_tensor = torch.tensor(epochs_no_improve, device=self.device)
                    dist.broadcast(best_loss_tensor, src=0)
                    dist.broadcast(epochs_tensor, src=0)
                    best_val_loss = best_loss_tensor.item()
                    epochs_no_improve = int(epochs_tensor.item())

        # Load best model and evaluate
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
        if self.rank == 0:
            wandb.finish()
        return metrics

    def stage1_mcq(
        self, batch_size: int = None, val_batch_size: int = None, eval_only: bool = False
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
        dataset_cls = partial(Phase1ChronosFuturePatchDataset, **dataset_kwargs)
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
            print(f"[joint_phase1_extend_balanced] task counts: {dict(task_counts)}")
            print(f"[joint_phase1_extend_balanced] balanced batches per task ≈ {len(sampler) // 3}")
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
                    batch_size=batch_size, val_batch_size=val_batch_size, eval_only=eval_only
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
            timeout=datetime.timedelta(hours=999),
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

        metrics_file = os.path.join(self.results_dir, stage, "results", "metrics.json")

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
            test_predictions_file = os.path.join(
                self.results_dir, stage, "results", "test_predictions.jsonl"
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

        metrics_file = os.path.join(self.results_dir, stage, "results", "metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        if self.rank == 0:
            print(f"✅ Stage {stage} marked as completed")

    def _get_model(self):
        """Get the underlying model (handles DDP wrapping)."""
        if hasattr(self.model, "module"):
            return self.model.module
        return self.model

    def _checkpoint_exists(self, stage: str) -> bool:
        """Check if a checkpoint exists for a specific stage."""
        checkpoint_path = os.path.join(
            self.results_dir, stage, "checkpoints", "best_model.pt"
        )
        return os.path.exists(checkpoint_path)


    def _enable_lora_if_needed(self, stage_name: str):
        """Enable LoRA for OpenTSLMSP models in stages after stage2."""
        if self.model_type not in ["OpenTSLMSP", "OpenTSLMSPFixPos", "OpenTSLMSPFixPos2D", "OpenTSLMSPFixPos2DTimerXL"]:
            return  # LoRA only for OpenTSLMSP

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
            "joint_phase1_extend_lorallm": (4, 8),
            "joint_phase2":             (16, 32),
            "joint_phase2_open":        (16, 32),
            "joint_phase2_downstream":  (16, 32),
        }

        if stage_name in stages_with_lora:
            if not getattr(model, "lora_enabled", False):
                lora_r, lora_alpha = stages_with_lora[stage_name]
                if self.rank == 0:
                    print(f"🔧 Enabling LoRA for {stage_name} (r={lora_r}, alpha={lora_alpha})")
                try:
                    model.enable_lora(lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0)
                    if self.rank == 0:
                        print(f"✅ LoRA enabled for {stage_name}")
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
        "--use_freq_embed", default=False, action="store_true",
        help="Augment patch embedding with a frequency-domain branch: each patch is "
             "transformed via rfft, embedded independently, then concatenated with the "
             "time-domain patch embedding and projected back to transformer_input_dim.",
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
        '--save_every_n_steps',
        type=int,
        default=10000,
        help='Save a periodic checkpoint every N training steps (0 = disabled). '
             'Saves to periodic_checkpoint.pt so progress is never lost if job hangs.'
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
        default=8,
        help="Number of DataLoader worker processes for async data loading. "
             "Set to 0 to disable (single-process, useful for debugging).",
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

    args = parser.parse_args()

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
