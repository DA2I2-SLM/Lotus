"""
Language modeling model for serialized time‑series data.

This module defines a :class:`PlainTextLMModel` that can be used to fine‑tune
large language models on time‑series data that have been converted into plain
text.  It builds on top of Hugging Face’s `AutoModelForCausalLM` and
`AutoTokenizer` and supports optional LoRA fine‑tuning through the `peft`
library.

Unlike the original ``PlainTextModel`` from OpenTSLM, this model does not
require a separate encoder or projector for numeric time‑series data.
Instead, the time series are expected to be provided as part of a single
text sequence (for example, via :class:`llm_timeseries_finetune.plain_text_dataset.PlainTextDataset`).
When computing the loss, the model predicts the next token across the
entire sequence, making it suitable for language‑modeling objectives.

The public API mirrors the relevant parts of ``PlainTextModel`` so that
existing training loops can be reused with minimal changes.
"""

from __future__ import annotations

import torch
from torch import nn
from typing import List, Dict, Tuple, Optional

from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import get_peft_model, LoraConfig, TaskType

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


class PlainTextLMModel(nn.Module):
    """Causal language model for time‑series expressed as text.

    Parameters
    ----------
    llm_id : str, optional
        Hugging Face model identifier for the base language model.  Defaults
        to ``"meta-llama/Llama-3.2-1B"``.  Any causal model supported by
        Hugging Face can be used (e.g. ``"google/gemma-2b"``).
    device : str, optional
        Device on which to place the model.  Defaults to ``"cuda"``.

    Notes
    -----
    - All model parameters are trainable by default.  To perform
      parameter‑efficient fine‑tuning, call :meth:`enable_lora`.
    - The tokenizer’s ``pad_token`` is set to the EOS token if missing.
    """

    def __init__(
        self,
        llm_id: str = "meta-llama/Llama-3.2-1B",
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        # 1) Initialise tokenizer (ensure pad_token exists)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_id, use_fast=True)
        if self.tokenizer.pad_token is None:
            # Many causal LMs use the EOS token as padding
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # 2) Load the base causal language model
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map={"": device},
            attn_implementation="eager",
        )
        # Resize token embeddings in case new special tokens were added
        self.llm.resize_token_embeddings(len(self.tokenizer))

        # LoRA state
        self.lora_enabled: bool = False
        self.original_llm: Optional[nn.Module] = None

    # ------------------------------------------------------------------
    # LoRA fine‑tuning methods
    # ------------------------------------------------------------------
    def enable_lora(
        self,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
    ):
        """Enable parameter‑efficient fine‑tuning via LoRA.

        This method wraps the internal LLM with a LoRA adapter using the
        provided configuration.  Only LoRA weights will be trainable; the
        original weights remain frozen.

        Parameters
        ----------
        lora_r : int, optional
            LoRA rank.  Smaller values require fewer parameters.
        lora_alpha : int, optional
            LoRA alpha scaling factor.
        lora_dropout : float, optional
            Dropout rate applied within the LoRA adapter.
        target_modules : list of str, optional
            Names of submodules within the transformer to which LoRA should
            be applied.  If ``None``, a default list covering common
            projection layers is used.

        Raises
        ------
        RuntimeError
            If the ``peft`` package is not available or LoRA is already enabled.
        """
        if not PEFT_AVAILABLE:
            raise RuntimeError(
                "peft package is required for LoRA fine‑tuning. "
                "Install via `pip install peft`."
            )
        if self.lora_enabled:
            raise RuntimeError(
                "LoRA is already enabled.  Call disable_lora() before reconfiguring."
            )
        # Store a reference to the unfrozen model for disabling later
        self.original_llm = self.llm
        # Default target modules for LoRA if none provided
        if target_modules is None:
            target_modules = [
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        # Apply LoRA; this returns a new model wrapper
        self.llm = get_peft_model(self.llm, lora_config)
        self.lora_enabled = True
        # Print summary of trainable parameters
        lora_params = sum(
            p.numel()
            for name, p in self.llm.named_parameters()
            if p.requires_grad and "lora_" in name
        )
        trainable_params = sum(
            p.numel() for p in self.llm.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.llm.parameters())
        print(f"✅ LoRA enabled:")
        print(f"   LoRA parameters: {lora_params:,}")
        print(f"   Total trainable parameters: {trainable_params:,}")
        print(f"   Total parameters: {total_params:,}")
        print(f"   LoRA %: {100 * lora_params / total_params:.2f}%")
        print(f"   Trainable %: {100 * trainable_params / total_params:.2f}%")

    def disable_lora(self):
        """Disable LoRA and revert to the original model.

        After calling this method, all LoRA adapters are removed and the
        original model parameters are restored.  Any LoRA weights will be
        discarded.

        Raises
        ------
        RuntimeError
            If LoRA was not previously enabled.
        """
        if not self.lora_enabled:
            raise RuntimeError(
                "LoRA is not enabled.  Cannot disable LoRA when it's inactive."
            )
        if self.original_llm is not None:
            self.llm = self.original_llm
            self.original_llm = None
        self.lora_enabled = False
        print("✅ LoRA disabled, reverted to base LLM")

    def get_lora_parameters(self) -> List[torch.nn.Parameter]:
        """Return the list of LoRA parameters for the optimizer."""
        if not self.lora_enabled:
            return []
        return [
            p
            for name, p in self.llm.named_parameters()
            if p.requires_grad and "lora_" in name
        ]

    # ------------------------------------------------------------------
    # Data preparation and inference methods
    # ------------------------------------------------------------------
    def _prepare_batch(self, batch: List[Dict[str, any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a batch of samples for language modeling.

        Parameters
        ----------
        batch : list of dict
            Each dict must contain a ``"text"`` key with the full input
            sequence (prompt + answer + EOS).  Additional keys are
            ignored.

        Returns
        -------
        input_ids : torch.Tensor
            Tensor of shape ``[B, L]`` containing token IDs.
        attention_mask : torch.Tensor
            Tensor of shape ``[B, L]`` where 1 marks tokens to attend to
            and 0 marks padding.
        """
        texts = [sample["text"] for sample in batch]
        tok = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = tok.input_ids.to(self.device, non_blocking=True)
        attention_mask = tok.attention_mask.to(self.device, non_blocking=True)
        return input_ids, attention_mask

    def generate(
        self,
        batch: List[Dict[str, any]],
        max_new_tokens: int = 50,
        **generate_kwargs,
    ) -> List[str]:
        """Generate completions for a batch of input sequences.

        Parameters
        ----------
        batch : list of dict
            Each dict must contain a ``"text"`` key.  This text is fed
            into the model as the prompt.
        max_new_tokens : int, optional
            Maximum number of tokens to generate for each sequence.
        **generate_kwargs :
            Additional arguments passed to ``transformers.AutoModelForCausalLM.generate``.

        Returns
        -------
        list of str
            Decoded outputs for each input sequence.
        """
        input_ids, attention_mask = self._prepare_batch(batch)
        # Delegate to Hugging Face generate
        gen_ids = self.llm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )
        return self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

    def compute_loss(self, batch: List[Dict[str, any]]) -> torch.Tensor:
        """Compute next‑token prediction loss for a batch.

        This method tokenizes the ``"text"`` field of each sample,
        constructs labels identical to the input IDs (i.e. every token
        is a target), and invokes the underlying language model to
        obtain the standard causal language modelling loss.

        Parameters
        ----------
        batch : list of dict
            Each dict must contain a ``"text"`` key.

        Returns
        -------
        torch.Tensor
            A scalar tensor containing the mean loss over the batch.
        """
        input_ids, attention_mask = self._prepare_batch(batch)
        labels = input_ids.clone()
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        return outputs.loss

    # ------------------------------------------------------------------
    # Utility methods for EOS and checkpointing
    # ------------------------------------------------------------------
    def get_eos_token(self) -> str:
        """Return the end‑of‑sequence token for the tokenizer."""
        return self.tokenizer.eos_token

    def save_lora_state_to_checkpoint(self, checkpoint: dict) -> int:
        """Save LoRA adapters into a checkpoint dictionary.

        Parameters
        ----------
        checkpoint : dict
            Dictionary into which LoRA state will be written.  It will be
            modified in place.

        Returns
        -------
        int
            The number of LoRA parameters saved.
        """
        checkpoint["lora_enabled"] = self.lora_enabled
        if self.lora_enabled and hasattr(self.llm, "peft_config"):
            lora_state = {}
            for name, param in self.llm.named_parameters():
                if param.requires_grad and "lora_" in name:
                    lora_state[name] = param.data.clone()
            if lora_state:
                checkpoint["lora_state"] = lora_state
                checkpoint["lora_config"] = self.llm.peft_config
                return len(lora_state)
        return 0

    def load_lora_state_from_checkpoint(
        self,
        checkpoint: dict,
        allow_missing: bool = False,
    ) -> int:
        """Load LoRA adapters from a checkpoint.

        Parameters
        ----------
        checkpoint : dict
            Dictionary containing potential LoRA state.
        allow_missing : bool, optional
            If True, missing LoRA parameters are silently ignored.  If
            False, missing LoRA parameters raise a RuntimeError.

        Returns
        -------
        int
            Number of LoRA parameters loaded.

        Raises
        ------
        RuntimeError
            If mismatches occur between the checkpoint and the current model
            and ``allow_missing`` is ``False``.
        """
        checkpoint_has_lora = checkpoint.get("lora_enabled", False)
        if checkpoint_has_lora and "lora_state" in checkpoint:
            if not self.lora_enabled:
                raise RuntimeError(
                    "Checkpoint contains LoRA adapters but LoRA is not currently enabled. "
                    "Call enable_lora() before loading this checkpoint."
                )
            lora_state = checkpoint["lora_state"]
            loaded_count = 0
            missing_keys = []
            # Determine which LoRA parameters are expected
            expected_lora_params = {
                name
                for name, param in self.llm.named_parameters()
                if param.requires_grad and "lora_" in name
            }
            for name, param in self.llm.named_parameters():
                if name in lora_state and param.requires_grad and "lora_" in name:
                    param.data.copy_(lora_state[name])
                    loaded_count += 1
                elif param.requires_grad and "lora_" in name:
                    missing_keys.append(name)
            if missing_keys and not allow_missing:
                raise RuntimeError(
                    f"Could not find LoRA parameters in checkpoint: {missing_keys[:5]}..."
                )
            return loaded_count
        elif checkpoint_has_lora:
            raise RuntimeError(
                "Checkpoint indicates LoRA was enabled but no LoRA state found."
            )
        # When checkpoint has no LoRA but model expects it
        if not checkpoint_has_lora and self.lora_enabled:
            if not allow_missing:
                raise RuntimeError(
                    "Loading checkpoint from before LoRA was enabled, but LoRA is currently enabled. "
                    "LoRA adapters will be randomly initialized. Set allow_missing=True to allow this."
                )
        return 0

    def store_to_file(self, path: str):
        """Persist model state (including LoRA) to a file.

        Only the LoRA parameters are saved for parameter‑efficient fine‑tuning.
        The underlying base model must be saved separately via Hugging Face
        utilities if full weights are required.
        """
        checkpoint = {}
        # Save LoRA state if applicable
        self.save_lora_state_to_checkpoint(checkpoint)
        torch.save(checkpoint, path)

    def load_from_file(self, path: str):
        """Load LoRA state from a file.

        Parameters
        ----------
        path : str
            Path to the checkpoint file produced by :meth:`store_to_file`.
        """
        ckpt = torch.load(path, map_location=self.device)
        self.load_lora_state_from_checkpoint(ckpt, allow_missing=True)
        print(f"📥 Loaded LoRA state from {path}")
