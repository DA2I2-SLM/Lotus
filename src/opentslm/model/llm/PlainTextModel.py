from __future__ import annotations

import torch
from torch import nn
from typing import List, Dict, Tuple, Optional
import re
from typing import List, Tuple, Sequence

from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
sys.path.append("/home/s223540177/FoundationalTS")
from llm_timeseries_finetune.serialize import serialize_arr, SerializerSettings
from .TimeSeriesLLM import TimeSeriesLLM


try:
    from peft import get_peft_model, LoraConfig, TaskType

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from torch.nn.utils.rnn import pad_sequence



def insert_serialized_timeseries_into_texts(
    all_texts: List[str],
    all_ts: Sequence[str],
    start_placeholder: str = "<ts>",
    end_placeholder: str = "</ts>",
    *,
    strict: bool = True,
) -> List[str]:
    """
    Replace each occurrence of: <ts> ... </ts> (optionally with whitespace inside)
    with: <ts>{serialized_ts}</ts>, consuming serialized_ts from all_ts in order.

    If strict=True:
      - raises if placeholder count != len(all_ts)
      - raises if a <ts> has no matching </ts> (via regex non-match / imbalance symptom)
    """
    # Match "<ts>   </ts>" or "<ts>...</ts>" non-greedily, across newlines
    # We intentionally replace the whole span with start+ts+end.
    pattern = re.compile(
        re.escape(start_placeholder) + r"(.*?)" + re.escape(end_placeholder),
        flags=re.DOTALL,
    )

    ts_i = 0
    out: List[str] = []

    for t in all_texts:
        def _repl(_m: re.Match) -> str:
            nonlocal ts_i
            if ts_i >= len(all_ts):
                if strict:
                    raise ValueError(
                        f"Found more placeholder pairs than provided time series. "
                        f"Need at least {ts_i+1}, have {len(all_ts)}."
                    )
                # If not strict, just keep original
                return _m.group(0)

            s = f"{start_placeholder}{all_ts[ts_i]}{end_placeholder}"
            ts_i += 1
            return s

        new_t, n_subs = pattern.subn(_repl, t)
        out.append(new_t)

    if strict and ts_i != len(all_ts):
        raise ValueError(
            f"Placeholder pairs consumed {ts_i} time series, but all_ts has {len(all_ts)}. "
            f"Either (1) not all samples had <ts></ts> pairs, or (2) ordering mismatch."
        )

    return out




class PlainTextLMModel(TimeSeriesLLM):
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

        # Tokenization settings
        self.max_length = 1024
        self.patch_size = 4

        self.settings = SerializerSettings()


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


    def pad_and_apply_batch(
        self,
        batch: List[Dict[str, any]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TL;DR:
            This function is probably the most crucial part of OpenTSLM-SP, and also the hardest to understand.
            It's where the magic happens and legends are made.

            It batches and embeds all text and time series inputs in parallel,
            then reassembles them per sample to allow efficient GPU execution.
            Praise the PyTorch Wizards: ChatGPT-o4-mini-high, Patrick, and Thomas (listed in strictly descending order of skill).

        Long description:
            Processes a batch of training samples by embedding and aligning text and time series data
            for efficient parallel processing on the GPU.

            This method performs the following steps:

            1. Extracts all text components (pre_prompt, time_series_text, post_prompt) from each sample,
            and embeds them in a single batch using the LLM tokenizer and embedding layer. Padding and attention
            masks are applied to accommodate variable-length sequences.

            2. Gathers all time series segments across the batch and pads them
            into a single tensor of shape [N_ts_total, T_padded, D], where T_padded
            is the smallest multiple of `patch_size` ≥ the longest segment length.
            This tensor is then encoded and projected into the LLM hidden space.

            3. After all embeddings are extracted, the function reconstructs each original sample by interleaving its
            embedded pre_prompt, time series texts and corresponding time series embeddings, and the post_prompt, preserving original order.

            4. Pads all reassembled sequences to a uniform length across the batch to form the final input tensor
                and attention mask.

            5. All of this is only required for efficient processing.

        - pre_prompt: str
        - time_series_text: List[str]
        - time_series: Tensor [N_ts, T] or [N_ts, T, D]
        - post_prompt: str
        Returns (inputs_embeds, attention_mask)
        """
        device = self.device
        H = self.llm.config.hidden_size

        # 1) Gather all texts
        all_texts: List[str] = []
        text_ptrs: List[Tuple[int, int]] = []
        ts_counts: List[int] = []
        all_ts = []

        for sample in batch:
            b_all_ts = []
            start = len(all_texts)
            if sample["pre_prompt"] is not None:
                assert isinstance(sample["pre_prompt"], str), f"Expected pre_prompt to be a string, got: {type(sample['pre_prompt'])}"
                all_texts.append(sample["pre_prompt"]+" "+sample["post_prompt"])
            end = len(all_texts)
            text_ptrs.append((start, end))
            ts_counts.append(len(sample["time_series_text"]))
            for ts in sample["time_series"]:
                serialized_ts = serialize_arr(ts, self.settings)
                b_all_ts.append(serialized_ts)
            all_ts.extend(b_all_ts)


        # convert the timeseries into text and append to all_texts, so they get embedded by the LLM tokenizer and embedding layer
        all_texts_with_ts = insert_serialized_timeseries_into_texts(
            all_texts=all_texts,
            all_ts=all_ts,
            start_placeholder="<ts>",
            end_placeholder="<ts/>",
            strict=True,
        )

        # 2) Tokenize & embed all texts
        tok = self.tokenizer(
            all_texts, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = tok.input_ids.to(device, non_blocking=True)
        attn_mask = tok.attention_mask.to(device, non_blocking=True)
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # [N_all, P_max, H]

        # # 5) Batch-pad the final sequences
        # inputs_embeds = pad_sequence(all_seq_embeds, batch_first=True)  # [B, L_max, H]
        # attention_mask = pad_sequence(all_seq_masks, batch_first=True)  # [B, L_max]

        return text_embeds, attn_mask

    def generate(
        self, batch: List[Dict[str, any]], max_new_tokens: int = 50, **generate_kwargs
    ) -> List[str]:
        inputs_embeds, attention_mask = self.pad_and_apply_batch(batch)
        gen_ids = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )
        return self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)


    def compute_loss(self, batch: List[Dict[str, any]]) -> torch.Tensor:
        """
        batch: same format as generate()
        answers: List[str] of length B
        """

        
        answers = [b["answer"] for b in batch]

        inputs_embeds, attention_mask = self.pad_and_apply_batch(batch)
        B, L, H = inputs_embeds.size()

        # tokenize answers
        ans_tok = self.tokenizer(
            answers, return_tensors="pt", padding=True, truncation=True
        )
        ans_ids = ans_tok.input_ids.to(self.device, non_blocking=True)
        ans_mask = ans_tok.attention_mask.to(self.device, non_blocking=True)
        ans_emb = self.llm.get_input_embeddings()(ans_ids)  # [B, A_max, H]

        # append
        inputs_embeds = torch.cat([inputs_embeds, ans_emb], dim=1)  # [B, L+A, H]
        attention_mask = torch.cat([attention_mask, ans_mask], dim=1)  # [B, L+A]

        # labels: only on the answer tokens
        total_len = attention_mask.size(1)
        labels = torch.full((B, total_len), -100, device=self.device, dtype=torch.long)
        labels[:, L:] = ans_ids

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
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
