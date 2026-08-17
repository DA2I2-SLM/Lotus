# SPDX-License-Identifier: MIT
"""Tokenized-LLM baseline: supervised fine-tuning of a small causal LM on the
Phase-2 five-task mixture, with the time series written into the prompt as text.

This is the control for the central architectural claim of the paper. Our model
projects patch embeddings into the backbone's embedding space; this baseline
instead serialises the numbers and lets the LM read them as tokens -- the
approach taken by TimeOmni-1 and by LLMTime-style work. Same data, same
mixture, same balanced sampling, same selection metric, so the only difference
is the interface.

Two things are deliberately generous to the baseline, so that a win for the
encoder cannot be dismissed as an unfair setup:
  * the backbone is fully fine-tuned (~494M trainable) versus ~4.1M for ours;
  * the series budget is spent on the largest context that trains on a V100.

Usage
    torchrun --nproc_per_node=N tokenized_sft.py --mode train  [flags]
    python  tokenized_sft.py --mode eval --ckpt <dir> --split test --out_dir <d>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

sys.path.insert(0, "/home/s223540177/FoundationalTS/src")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from serialize import serialize_sample  # noqa: E402

TASK_TYPES = {
    "har": "har_cot",
    "sleep": "sleep_cot",
    "ecg": "ecg_cot",
    "pamap2": "pamap2_cot",
    "timemqa": "timemqa_cot",
}

# Tasks that participate in checkpoint selection and early stopping.
# TimeMQA is EXCLUDED, exactly as `_GEN_TYPES` excludes it in curriculum_learning.py:
# its answers are free text with no closed label set, so exact match against them is
# structurally 0.000 at every validation. Leaving it in makes its patience counter
# climb monotonically and early-stops the run on a metric that can never improve --
# the same failure that killed the official baselines via ECG. Caught by the smoke test.
SELECT_TASKS = ["har", "sleep", "ecg", "pamap2"]


# --------------------------------------------------------------------------- data
class SerializedDataset(Dataset):
    """Wraps Phase2DownstreamDataset and renders each sample to text."""

    def __init__(self, base, tokenizer, max_len: int, max_values: int,
                 serializer: str = "ours"):
        self.base = base
        self.tok = tokenizer
        self.max_len = max_len
        self.max_values = max_values
        self.serializer = serializer

    def __len__(self) -> int:
        return len(self.base.samples)

    def _encode_prompt(self, text: str) -> List[int]:
        """Tokenise the user turn.

        Instruct backbones (Qwen2.5-*-Instruct) carry a chat template and are
        used as intended. Base backbones do not -- notably google/gemma-3-270m,
        which is the backbone our own model wraps. There we concatenate raw
        text, which is exactly what OpenTSLMSPFixPos2D does with its
        pre_prompt / post_prompt fields: no chat turns, no role markers. So the
        backbone-matched run compares two interfaces to the *same* prompt
        string, and the only difference left is projected patches vs digits.
        """
        if getattr(self.tok, "chat_template", None):
            out = self.tok.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True, add_generation_prompt=True,
            )
            # transformers 4.x returns a plain id list; 5.x returns a
            # BatchEncoding (and sometimes batches it). Normalise to list[int],
            # or `prompt_ids + target_ids` dies with
            # "unsupported operand type(s) for +: 'BatchEncoding' and 'list'".
            if hasattr(out, "keys"):
                out = out["input_ids"]
            if out and isinstance(out[0], list):
                out = out[0]
            return list(out)
        return self.tok(text + "\n", add_special_tokens=True).input_ids

    def __getitem__(self, i: int) -> Dict[str, Any]:
        row = self.base[i]
        target_ids = self.tok(
            str(row.get("answer", "")).replace("<eos>", "").strip(),
            add_special_tokens=False,
        ).input_ids + [self.tok.eos_token_id]
        budget = self.max_len - len(target_ids)

        # Fit the SERIES to the token budget by decimating harder, rather than
        # letting the prompt be truncated. Truncation would silently delete the
        # head of the prompt -- which is exactly where the question lives -- so a
        # 12-lead ECG sample would arrive as numbers with no question attached.
        self._fit_calls = getattr(self, "_fit_calls", 0) + 1
        s = serialize_sample(row, max_values_total=self.max_values, serializer=self.serializer)
        prompt_ids = self._encode_prompt(s["prompt"])

        if len(prompt_ids) > budget:
            # Bisect for the LARGEST value count that fits. Token count is monotone
            # in the number of serialised values, so plain bisection is exact.
            #
            # Two earlier attempts were both wrong, in opposite directions. Shaving
            # by the estimated overflow alone converged too slowly and fell through
            # to text truncation on 3.5% of ECG samples (gemma) / 1.5% (Qwen).
            # Forcing at least a halving fixed that but overshot: ECG dropped
            # straight to 256 values / stride 48, leaving ~1.1k tokens of the window
            # unused on EVERY sample -- degrading the baseline's hardest column to
            # fix a 3.5% tail. Bisection gives 0% truncation at maximal resolution,
            # which is what "generous to the baseline" in the header requires.
            #
            # Only long tasks enter this branch; har/pamap2/timemqa cost one call.
            lo, hi = 32, self.max_values          # lo assumed to fit, hi known not to
            for _ in range(7):                    # 480-wide range -> within ~4 values
                mid = (lo + hi) // 2
                if mid <= lo:
                    break
                s_m = serialize_sample(row, max_values_total=mid, serializer=self.serializer)
                ids_m = self._encode_prompt(s_m["prompt"])
                if len(ids_m) <= budget:
                    lo, s, prompt_ids = mid, s_m, ids_m
                else:
                    hi = mid
            if len(prompt_ids) > budget:          # never found a fit; try the floor
                s = serialize_sample(row, max_values_total=32, serializer=self.serializer)
                prompt_ids = self._encode_prompt(s["prompt"])

        # Last resort only: if even 32 values will not fit, the non-series text
        # alone exceeds the window. Keep the HEAD (question) and drop the tail.
        if len(prompt_ids) > budget:
            self._truncated = getattr(self, "_truncated", 0) + 1
            if self._truncated in (1, 10, 100, 1000):
                print(f"[tokenized] WARNING: prompt truncated even at "
                      f"max_values=32 ({self._truncated} so far of {self._fit_calls} "
                      f"samples, task={s['type']}) -- the non-series text alone "
                      f"exceeds the window.", flush=True)
            prompt_ids = prompt_ids[: budget - 1] + prompt_ids[-1:]

        ids = prompt_ids + target_ids
        # -100 on the prompt: supervise the answer only, matching our CE setup.
        labels = [-100] * len(prompt_ids) + target_ids[:]
        return {
            "input_ids": ids,
            "labels": labels,
            "type": s["type"],
            "target_text": s["target"],
            "label": s.get("label"),
            "decimated": s["decimated"],
            "max_stride": s["max_stride"],
            "idx": i,
        }


@torch.no_grad()
def val_loss(model, tok, ds_va, device, n: int, pad_id: int) -> float:
    """Teacher-forced mean loss on the answer tokens of `n` validation samples.

    Exists because --select_by val_loss needs it and nothing here computed a
    validation loss at all: selection was gen-EM only. Deterministic subset (the
    first n) so the number is comparable across evals and across runs.
    """
    was_training = model.training
    model.eval()
    tot, cnt = 0.0, 0
    for i in range(min(n, len(ds_va))):
        b = ds_va[i]
        ids = torch.tensor([b["input_ids"]], device=device)
        lab = torch.tensor([b["labels"]], device=device)
        att = torch.ones_like(ids)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(input_ids=ids, attention_mask=att, labels=lab)
        if torch.isfinite(out.loss):
            tot += float(out.loss); cnt += 1
    if was_training:
        model.train()
    return tot / cnt if cnt else float("inf")


def collate(batch, pad_id: int):
    n = max(len(b["input_ids"]) for b in batch)
    ids, labels, attn = [], [], []
    for b in batch:
        k = n - len(b["input_ids"])
        ids.append(b["input_ids"] + [pad_id] * k)
        labels.append(b["labels"] + [-100] * k)
        attn.append([1] * len(b["input_ids"]) + [0] * k)
    return (
        torch.tensor(ids), torch.tensor(labels), torch.tensor(attn),
        [b["type"] for b in batch],
    )


def build_base(mode: str, args):
    from opentslm.time_series_datasets.Phase2DownstreamDataset import Phase2DownstreamDataset
    return Phase2DownstreamDataset(
        mode=mode, EOS_TOKEN="<eos>", patch_size=32, windows_per_col_train=0
    )


# ------------------------------------------------------------------- generative EM
@torch.no_grad()
def gen_em(model, tok, base, ds, device, n_per_task: int, max_new: int = 200):
    """Mean exact-match over a fixed val subset, mirroring the selection metric
    used by the encoder runs so the comparison is like-for-like."""
    from opentslm.time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset
    from opentslm.time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset
    from opentslm.time_series_datasets.ecg_qa.ECGQACoTQADataset import ECGQACoTQADataset
    from opentslm.time_series_datasets.pamap2.PAMAP2CoTQADataset import PAMAP2CoTQADataset

    cands = {
        "har": HARCoTQADataset.get_labels(),
        "sleep": SleepEDFCoTQADataset.get_labels(),
        "ecg": ECGQACoTQADataset.get_labels(),
        "pamap2": PAMAP2CoTQADataset.get_labels(),
    }
    m = model.module if hasattr(model, "module") else model
    m.eval()
    out = {}
    for task, ttype in TASK_TYPES.items():
        idxs = [i for i, s in enumerate(base.samples)
                if isinstance(s, dict) and s.get("type") == ttype][:n_per_task]
        if not idxs:
            out[task] = float("nan")
            continue
        correct = total = 0
        for i in idxs:
            item = ds[i]
            plen = item["labels"].count(-100)
            inp = torch.tensor([item["input_ids"][:plen]], device=device)
            gen = m.generate(inp, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
            pred = tok.decode(gen[0][plen:], skip_special_tokens=True)
            gold = extract_answer(item["target_text"]) or (item.get("label") or "")
            pl = extract_label(pred, cands.get(task, []))
            total += 1
            correct += int(norm(pl) == norm(gold) and bool(gold))
        out[task] = correct / total if total else float("nan")
    m.train()
    return out


def norm(s: str) -> str:
    import re
    s = re.sub(r"[^\w\s]", " ", str(s).lower())
    return " ".join(t for t in s.split() if t not in {"eos", "bos", "pad", "unk", "ts"})


def extract_answer(text: str) -> str:
    import re
    m = re.search(r"answer\s*[:\-]\s*([^\n.<]+)", str(text), re.I)
    if m:
        return m.group(1).rstrip(".").strip()
    lines = [l.strip() for l in str(text).splitlines() if l.strip()]
    return lines[-1].rstrip(".").strip() if lines and len(lines[-1]) <= 40 else ""


def extract_label(text: str, cands: List[str]) -> str:
    suf = extract_answer(text)
    ns = norm(suf)
    if suf and cands:
        for c in cands:
            if norm(c) == ns:
                return c
        for c in cands:
            if norm(c) in ns or ns in norm(c):
                return c
        return suf
    if suf:
        return suf
    nt = norm(text)
    for c in cands:
        if norm(c) in nt:
            return c
    return ""


# ------------------------------------------------------------------------- train
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--model_id", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--tasks", default=None,
                   help="eval only: comma-separated subset of har,sleep,ecg,pamap2,timemqa "
                        "(default all). Combine with EVAL_SHARD_START/END to shard a task.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=32)
    p.add_argument("--max_len", type=int, default=4096)
    p.add_argument("--max_values", type=int, default=512)
    p.add_argument("--gen_em_n", type=int, default=192)
    p.add_argument("--eval_every", type=int, default=500, help="optimizer steps")
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    # 0 = off = full fine-tune, the historical behaviour of every existing row.
    # >0 gives the matched-capacity arm against our encoder model (r=8, alpha=16).
    p.add_argument("--lora_r", type=int, default=0,
                   help="0 = full fine-tune (default). >0 wraps the backbone in "
                        "LoRA of this rank, matching our model's adapter.")
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--serializer", choices=["ours", "llmtime"], default="ours",
                   help="'ours' = %%.2f comma-separated (default, all existing rows). "
                        "'llmtime' = Gruver et al. (2023) digit-level encoding.")
    p.add_argument("--select_by", choices=["gen_em", "val_loss"], default="gen_em",
                   help="'gen_em' = generative exact-match basket (default, matches "
                        "our encoder runs). 'val_loss' = teacher-forced validation "
                        "loss, as used by the LLMTime-style published baselines.")
    p.add_argument("--lora_targets", type=str, default="",
                   help="comma-separated target module names; empty = the seven "
                        "our encoder model uses. Override only when the backbone "
                        "names its projections differently (see the note above).")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world > 1
    if ddp:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); random.seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    src = args.ckpt or args.model_id
    _adapter_cfg = os.path.join(src, "adapter_config.json")
    if os.path.exists(_adapter_cfg):
        # A LoRA run's "best" dir holds only the adapter. Rebuild base+adapter,
        # then merge so everything downstream sees an ordinary causal LM.
        import json as _json
        from peft import PeftModel
        _base = _json.load(open(_adapter_cfg))["base_model_name_or_path"]
        if rank == 0:
            print(f"[tokenized] adapter dir detected; base={_base}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(_base, torch_dtype=torch.float32)
        model = PeftModel.from_pretrained(model, src).merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.float32)

    # ── optional LoRA arm ────────────────────────────────────────────────────
    # By default this baseline FULLY fine-tunes, which is deliberately generous
    # (see the module docstring). But "tokenized wins" is then ambiguous: it has
    # ~90x the trainable parameters of our encoder model, so the interface and
    # the capacity are confounded. --lora_r turns this into the matched-capacity
    # arm: same backbone, same data, same mixture, same selection metric, an
    # adapter of the SAME rank/alpha/target-modules our model uses, and only the
    # interface (serialised digits vs projected patches) differing.
    # Defaults to 0 = off, so every existing tokenized row reproduces unchanged.
    if getattr(args, "lora_r", 0):
        from peft import LoraConfig, TaskType, get_peft_model
        # Default = the seven names our encoder model's adapter uses, so gemma /
        # Qwen rows are unchanged. Overridable because "the same seven names" is
        # not the same COVERAGE on every architecture: Qwen3.5 (TimeOmni-1) is a
        # hybrid with full_attention_interval=4, so only 8 of its 32 layers have
        # q/k/v/o_proj at all -- the other 24 use linear_attn with in_proj_qkv /
        # in_proj_z / out_proj. Passing the bare 7 names there would adapt just
        # 8/32 attention blocks (10.62M params, 71.7% of linears) versus every
        # block on gemma, silently handicapping the baseline.
        targets = (args.lora_targets.split(",") if getattr(args, "lora_targets", "")
                   else ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
            target_modules=targets, bias="none", task_type=TaskType.CAUSAL_LM))
        if rank == 0:
            print(f"[tokenized] LoRA r={args.lora_r} alpha={args.lora_alpha} "
                  f"targets={targets}", flush=True)

    if getattr(args, "lora_r", 0):
        # use_reentrant=False is REQUIRED with a frozen base: the reentrant
        # variant treats checkpointed block outputs as non-differentiable when
        # the inputs do not require grad, silently zeroing LoRA gradients. Same
        # failure mode documented in OpenTSLMSPFixPos2D.enable_lora.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    else:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.to(device)

    if rank == 0:
        n = sum(q.numel() for q in model.parameters() if q.requires_grad)
        tot = sum(q.numel() for q in model.parameters())
        print(f"[tokenized] {args.model_id} trainable params: {n:,} of {tot:,} "
              f"({100 * n / tot:.2f}%)", flush=True)

    if args.mode == "eval":
        run_eval(model, tok, args, device)
        return

    base_tr = build_base("train", args)
    base_va = build_base("validation", args)
    ds_tr = SerializedDataset(base_tr, tok, args.max_len, args.max_values, args.serializer)
    ds_va = SerializedDataset(base_va, tok, args.max_len, args.max_values, args.serializer)

    if rank == 0:
        report_decimation(ds_tr, base_tr)

    samp = DistributedSampler(ds_tr, shuffle=True) if ddp else None
    dl = DataLoader(ds_tr, batch_size=args.batch_size, sampler=samp,
                    shuffle=(samp is None), num_workers=0, drop_last=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))

    if ddp:
        model = DDP(model, device_ids=[local_rank],
                    find_unused_parameters=bool(getattr(args, "lora_r", 0)))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")   # V100: fp16 AMP

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    best = float("inf")
    # Per-task bests, mirroring best_val_cot in curriculum_learning.py: a task's
    # patience resets only when THAT task improves on its own history, not when
    # the pooled mean happens to improve.
    best_task = {t: float("inf") for t in SELECT_TASKS}
    no_improve = {t: 0 for t in SELECT_TASKS}
    step = 0

    for epoch in range(1, args.epochs + 1):
        if samp:
            samp.set_epoch(epoch)
        model.train()
        # Patience is assessed at the EPOCH boundary (block after this loop), not
        # per eval. This flag records whether a task beat its own history at any
        # eval inside the epoch.
        improved_this_epoch = {t: False for t in SELECT_TASKS}
        for i, (ids, labels, attn, _types) in enumerate(dl):
            ids, labels, attn = ids.to(device), labels.to(device), attn.to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=ids, attention_mask=attn, labels=labels).loss
            scaler.scale(loss / args.grad_accum).backward()
            if (i + 1) % args.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % args.eval_every == 0:
                    if args.select_by == "val_loss":
                        # Single scalar drives both best-model and patience, so
                        # the per-task basket is bypassed entirely. This is the
                        # protocol the published LLMTime-style baselines use.
                        vl = val_loss(model, tok, ds_va, device,
                                      args.gen_em_n, tok.pad_token_id)
                        if rank == 0:
                            print(f"  val-loss {vl:.4f}", flush=True)
                            if vl + 1e-4 < best:
                                best = vl
                                for t in SELECT_TASKS:
                                    improved_this_epoch[t] = True
                                (model.module if ddp else model).save_pretrained(
                                    Path(args.out_dir) / "best")
                                tok.save_pretrained(Path(args.out_dir) / "best")
                                print("  New best model saved (val_loss).", flush=True)
                        if ddp:
                            dist.barrier()
                        continue
                    em = gen_em(model, tok, base_va, ds_va, device, args.gen_em_n)
                    # Selection basket excludes timemqa (see SELECT_TASKS).
                    present = [1 - em[t] for t in SELECT_TASKS
                               if t in em and em[t] == em[t]]
                    mean = sum(present) / len(present) if present else float("inf")
                    if rank == 0:
                        print("  gen-EM — " + "  ".join(
                            f"{t}={v:.3f}" for t, v in em.items() if v == v)
                            + "   [timemqa reported, not selected on]", flush=True)
                        for t in SELECT_TASKS:
                            v = em.get(t, float("nan"))
                            if v != v:
                                continue
                            if (1 - v) + 1e-4 < best_task[t]:
                                best_task[t] = 1 - v
                                improved_this_epoch[t] = True
                        if mean + 1e-4 < best:
                            best = mean
                            (model.module if ddp else model).save_pretrained(
                                Path(args.out_dir) / "best")
                            tok.save_pretrained(Path(args.out_dir) / "best")
                            print("✔️  New best model saved.", flush=True)

        # ── Early stopping (epoch-level) — mirrors curriculum_learning.py:3653 ──
        # This MUST be per-epoch, not per-eval. The encoder runs increment their
        # counters once per epoch, so `patience=8` there means 8 epochs. Counting
        # per eval here meant 8 evals = 2 epochs (evals fire 4x/epoch at
        # eval_every=500), giving the tokenized baselines a 4x shorter budget than
        # the model they are the control for. It killed the Qwen run at epoch 3.84
        # on the very eval where it saved a new best and HAR was still improving.
        #
        # The decision is made on rank 0 and BROADCAST. Previously the counters
        # were mutated under `if rank == 0` but the stop test ran on every rank,
        # so rank 0 returned alone while the others kept waiting on collectives —
        # that is what aborted the Qwen job with SIGABRT rather than exiting clean.
        stop = 0
        if rank == 0:
            active = [t for t in SELECT_TASKS if best_task[t] != float("inf")]
            for t in active:
                if improved_this_epoch[t]:
                    no_improve[t] = 0
                else:
                    no_improve[t] += 1
            print(f"Epoch {epoch} done (opt steps={step})", flush=True)
            print("  CoT patience: " + "  ".join(
                f"{t}={no_improve[t]}/{args.patience}" for t in active), flush=True)
            stag = [f"{t} ({no_improve[t]} ep)" for t in active
                    if no_improve[t] >= args.patience]
            if stag:
                print(f"\nEarly stopping (per-task): {', '.join(stag)} "
                      f"stagnated for {args.patience} epochs.", flush=True)
                stop = 1
        if ddp:
            _s = torch.tensor([stop], device=device, dtype=torch.int32)
            dist.broadcast(_s, src=0)
            stop = int(_s.item())
        if stop:
            return


def report_decimation(ds: SerializedDataset, base) -> None:
    """State plainly how much of each signal survives serialisation."""
    print("[tokenized] serialisation budget report "
          f"(max_values={ds.max_values}, max_len={ds.max_len}):", flush=True)
    for task, ttype in TASK_TYPES.items():
        idxs = [i for i, s in enumerate(base.samples)
                if isinstance(s, dict) and s.get("type") == ttype][:8]
        if not idxs:
            print(f"  {task:9s} absent", flush=True); continue
        row = ds[idxs[0]]
        n_tok = len(row["input_ids"])
        print(f"  {task:9s} tokens={n_tok:6d}  decimated={row['decimated']}  "
              f"stride={row['max_stride']}", flush=True)


@torch.no_grad()
def run_eval(model, tok, args, device):
    """Write predictions in the same jsonl schema the encoder runs produce, so
    the existing scorers and the results table need no special-casing."""
    base = build_base(args.split, args)
    ds = SerializedDataset(base, tok, args.max_len, args.max_values)
    from opentslm.time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset
    from opentslm.time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset
    from opentslm.time_series_datasets.ecg_qa.ECGQACoTQADataset import ECGQACoTQADataset
    from opentslm.time_series_datasets.pamap2.PAMAP2CoTQADataset import PAMAP2CoTQADataset
    cands = {"har": HARCoTQADataset.get_labels(), "sleep": SleepEDFCoTQADataset.get_labels(),
             "ecg": ECGQACoTQADataset.get_labels(), "pamap2": PAMAP2CoTQADataset.get_labels()}
    model.eval()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Task filter + shard window, mirroring EVAL_SHARD_START/END in
    # evaluation/cot/eval_cot_phase2.py. Without these the whole test set is
    # 16,262 single-sample generations in one job, which times out -- the same
    # trap the OpenTSLM-SP eval hit on 7 Aug.
    want = [t.strip() for t in (args.tasks or "").split(",") if t.strip()] or list(TASK_TYPES)
    _ss = int(os.environ.get("EVAL_SHARD_START", "0"))
    _se = int(os.environ.get("EVAL_SHARD_END", "-1"))

    per_task: Dict[str, List[Dict[str, Any]]] = {t: [] for t in TASK_TYPES}
    for task, ttype in TASK_TYPES.items():
        if task not in want:
            continue
        idxs = [i for i, s in enumerate(base.samples)
                if isinstance(s, dict) and s.get("type") == ttype]
        total = len(idxs)
        lo = max(0, _ss)
        hi = total if _se < 0 else min(_se, total)
        # `j` stays the GLOBAL position within the task so shards carry disjoint,
        # stitchable idx values that line up with the encoder runs' predictions.
        window = list(enumerate(idxs))[lo:hi]
        print(f"[tokenized] {task}: samples [{lo},{hi}) of {total}", flush=True)
        for j, i in window:
            item = ds[i]
            plen = item["labels"].count(-100)
            inp = torch.tensor([item["input_ids"][:plen]], device=device)
            gen = model.generate(inp, max_new_tokens=200, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            pred = tok.decode(gen[0][plen:], skip_special_tokens=True)
            gold = item["target_text"]
            per_task[task].append({
                "idx": j, "gold": gold, "pred": pred,
                "label_gold": extract_answer(gold) or (item.get("label") or ""),
                "label_pred": extract_label(pred, cands.get(task, [])),
                "decimated": item["decimated"], "max_stride": item["max_stride"],
            })
        d = out_dir / task; d.mkdir(exist_ok=True)
        with open(d / f"{task}_test_predictions.jsonl", "w") as f:
            for r in per_task[task]:
                f.write(json.dumps(r) + "\n")
        print(f"[tokenized] wrote {len(per_task[task])} preds for {task}", flush=True)


if __name__ == "__main__":
    main()
