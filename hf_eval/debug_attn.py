#!/usr/bin/env python3
"""Test whether forcing attn_implementation='eager' (and/or fp32) fixes the
degenerate-generation bug on the gemma-3-270m SP checkpoint.

gemma-3 is documented to require eager attention for numerically correct
generation; sdpa+bf16 can produce garbage. OpenTSLMSP hardcodes sdpa+bf16, so
we monkey-patch AutoModelForCausalLM.from_pretrained (no repo edits) to override
attn_implementation/dtype before the model is built.
"""
import getpass
import sys

USER = getpass.getuser()
sys.path.insert(0, f"/home/{USER}/FoundationalTS/src")

import torch
import transformers
from transformers import AutoModelForCausalLM as _AMC

_ORIG_FROM_PRETRAINED = _AMC.from_pretrained
OVERRIDE = {"attn_implementation": "eager", "torch_dtype": None}


def _patched_from_pretrained(cls, *args, **kwargs):
    if OVERRIDE.get("attn_implementation"):
        kwargs["attn_implementation"] = OVERRIDE["attn_implementation"]
    if OVERRIDE.get("torch_dtype") is not None:
        kwargs["torch_dtype"] = OVERRIDE["torch_dtype"]
    return _ORIG_FROM_PRETRAINED.__func__(cls, *args, **kwargs)


_AMC.from_pretrained = classmethod(_patched_from_pretrained)

from torch.utils.data import DataLoader
from opentslm.model.llm.OpenTSLM import OpenTSLM
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)
from opentslm.time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset

REPO = "OpenTSLM/gemma-3-270m-pt-sleep-sp"
device = "cuda" if torch.cuda.is_available() else "cpu"


def looks_garbage(t):
    low = (t or "").lower()
    bad = ("penurunan", "qualquer", "<unused", "attempt attempt", "тельное",
           "linklabel", "filedialog", "ramo ", "thụ", "chesterchester", "підви")
    return (not t) or any(b in low for b in bad)


def evaluate(model, n=12, max_new=220):
    eos = model.get_eos_token()
    ds = SleepEDFCoTQADataset("test", EOS_TOKEN=eos)
    ps = getattr(model, "patch_size", 4)
    sub = torch.utils.data.Subset(ds, list(range(n)))
    loader = DataLoader(sub, batch_size=1, shuffle=False,
                        collate_fn=lambda b: extend_time_series_to_match_patch_size_and_aggregate(b, patch_size=ps))
    coh = corr = 0
    samples = []
    with torch.no_grad():
        for batch in loader:
            preds = model.generate(batch, max_new_tokens=max_new)
            for s, p in zip(batch, preds):
                g = looks_garbage(p)
                if not g and "answer:" in p.lower():
                    coh += 1
                gold = (s.get("label") or "").strip().lower()
                if gold and "answer:" in p.lower() and gold in p.lower().split("answer:")[-1]:
                    corr += 1
                if len(samples) < 3:
                    samples.append(("GARB" if g else "OK", p[:110]))
    return coh, corr, n, samples


for cfg_name, attn, dt in [
    ("eager+bf16", "eager", torch.bfloat16),
    ("eager+fp32", "eager", torch.float32),
    ("sdpa+fp32", "sdpa", torch.float32),
]:
    OVERRIDE["attn_implementation"] = attn
    OVERRIDE["torch_dtype"] = dt
    print(f"\n===== CONFIG {cfg_name} (attn={attn}, dtype={dt}) =====", flush=True)
    model = OpenTSLM.load_pretrained(REPO, enable_lora=True, device=device)
    model.eval()
    coh, corr, n, samples = evaluate(model)
    print(f"RESULT {cfg_name}: coherent {coh}/{n}  correct {corr}/{n}", flush=True)
    for tag, txt in samples:
        print(f"   [{tag}] {txt!r}", flush=True)
    del model
    torch.cuda.empty_cache()

print("\nDONE_ATTN")
