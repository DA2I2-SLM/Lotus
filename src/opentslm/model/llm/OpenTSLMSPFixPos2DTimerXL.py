# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .OpenTSLMSPFixPos2D import OpenTSLMSPFixPos2D
from .TimeSeriesLLM import TimeSeriesLLM
from ..decoder.TransformerCNNDecoder2D import TransformerCNNDecoder2D
from ..encoder.TransformerTimerXLEncoder2D import TransformerTimerXLEncoder2D
from opentslm.model_config import PATCH_SIZE, MAX_FUTURE_PATCH_PE_SIZE
from ..encoder.TransformerCNNEncoder2D import _get_1d_sincos_pos_embed


class OpenTSLMSPFixPos2DTimerXL(OpenTSLMSPFixPos2D):
    """
    OpenTSLM-SP variant that swaps the original 2D encoder for a Timer-XL-style
    encoder while keeping the rest of the LLM/decoder stack unchanged.
    """

    def __init__(
        self,
        args,
        llm_id: str = "meta-llama/Llama-3.2-1B",
        device: str = "cuda",
    ):
        TimeSeriesLLM.__init__(self, device)
        self.args = args

        self.tokenizer = AutoTokenizer.from_pretrained(llm_id, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_id,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            attn_implementation="sdpa",
        )
        self.llm.resize_token_embeddings(len(self.tokenizer))

        print("Length of tokenizer vocab:", len(self.tokenizer))

        special_tokens = {"additional_special_tokens": [
            "<ts>", "</ts>", "<future_patch>", "<ch_sep>",
        ]}
        self.tokenizer.add_special_tokens(special_tokens)
        self.llm.resize_token_embeddings(len(self.tokenizer))

        self.ts_start_token_id     = self.tokenizer.convert_tokens_to_ids("<ts>")
        self.ts_end_token_id       = self.tokenizer.convert_tokens_to_ids("</ts>")
        self.future_patch_token_id = self.tokenizer.convert_tokens_to_ids("<future_patch>")
        self.ch_sep_token_id       = self.tokenizer.convert_tokens_to_ids("<ch_sep>")

        _pe = _get_1d_sincos_pos_embed(
            self.llm.config.hidden_size,
            torch.arange(MAX_FUTURE_PATCH_PE_SIZE, dtype=torch.float32),
        )
        _emb_norm = self.llm.get_input_embeddings().weight.data.float().cpu().norm(dim=1).mean()
        _pe_norm  = _pe.norm(dim=1).mean().clamp(min=1e-8)
        _pe = _pe * (_emb_norm / _pe_norm)
        self.register_buffer("future_patch_pe", _pe)

        print("Length after adding special tokens:", len(self.tokenizer))
        print("Tokenizer special tokens:", self.tokenizer.special_tokens_map)

        encoder_patch_size = int(getattr(args, "patch_size", PATCH_SIZE))
        self.encoder = TransformerTimerXLEncoder2D(
            args=args,
            patch_size=encoder_patch_size,
        ).to(device)

        self.patch_size = int(self.encoder.patch_size)
        self._debug_step = 0
        self.counter = 0
        self.lora_enabled = False
        self.original_llm = None

        for p in self.llm.parameters():
            p.requires_grad = False

        if getattr(args, "gradient_checkpointing", False):
            self.llm.gradient_checkpointing_enable()
            print("🧠 Timer-XL path: LLM gradient checkpointing enabled")
        else:
            self.llm.gradient_checkpointing_disable()
            print("⚡ Timer-XL path: LLM gradient checkpointing disabled")

        enc_dim = self.encoder.output_dim
        llm_dim = self.llm.config.hidden_size

        self.decoder = TransformerCNNDecoder2D(
            input_dim=enc_dim,
            patch_size=self.patch_size,
        ).to(device)

        self.projector = nn.Sequential(
            nn.Linear(enc_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        ).to(device)

        self.decoder_projector = nn.Sequential(
            nn.Linear(llm_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, enc_dim),
        ).to(device)


