#!/usr/bin/env python3
"""Diagnostic: load the OpenTSLM Flamingo checkpoint with the compat shim,
report state_dict load coverage, and generate a few sleep samples to eyeball
whether outputs are coherent (i.e. weights really loaded)."""
import getpass
import os
import sys

USER = getpass.getuser()
sys.path.insert(0, f"/home/{USER}/FoundationalTS/src")
sys.path.insert(0, f"/home/{USER}/FoundationalTS/hf_eval")

import torch
import flamingo_compat  # noqa: F401  (applies the monkey-patch on import)

# ---- Intercept load_state_dict to capture missing/unexpected key counts ----
from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo

_orig_lsd = OpenTSLMFlamingo.load_state_dict
_capture = {}


def _wrapped_lsd(self, state_dict, strict=True):
    out = _orig_lsd(self, state_dict, strict=False)
    try:
        missing, unexpected = out
    except Exception:
        missing, unexpected = [], []
    _capture["missing"] = list(missing)
    _capture["unexpected"] = list(unexpected)
    _capture["ckpt_keys"] = len(state_dict)
    return out


OpenTSLMFlamingo.load_state_dict = _wrapped_lsd

from opentslm.model.llm.OpenTSLM import OpenTSLM
from opentslm.model_config import PATCH_SIZE
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)
from torch.utils.data import DataLoader

REPO = "OpenTSLM/gemma-3-270m-pt-sleep-flamingo"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}")
model = OpenTSLM.load_pretrained(REPO, enable_lora=False, device=device)
model.eval()

print("\n===== STATE_DICT LOAD COVERAGE =====")
miss = _capture.get("missing", [])
unexp = _capture.get("unexpected", [])
print(f"ckpt model_state keys : {_capture.get('ckpt_keys')}")
print(f"missing keys (in model, not in ckpt) : {len(miss)}")
print(f"unexpected keys (in ckpt, not model) : {len(unexp)}")
# How many vision_encoder keys are missing/unexpected (the critical encoder)?
miss_ve = [k for k in miss if "vision_encoder" in k]
unexp_ve = [k for k in unexp if "vision_encoder" in k]
print(f"  vision_encoder missing   : {len(miss_ve)}  e.g. {miss_ve[:5]}")
print(f"  vision_encoder unexpected : {len(unexp_ve)}  e.g. {unexp_ve[:5]}")
print(f"  first 15 missing: {miss[:15]}")
print(f"  first 15 unexpected: {unexp[:15]}")

print("\n===== SAMPLE GENERATIONS (Sleep) =====")
from opentslm.time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset
ds = SleepEDFCoTQADataset("test", EOS_TOKEN=model.get_eos_token())
loader = DataLoader(ds, batch_size=2, shuffle=False,
                    collate_fn=lambda b: extend_time_series_to_match_patch_size_and_aggregate(b, patch_size=PATCH_SIZE))
with torch.no_grad():
    for i, batch in enumerate(loader):
        preds = model.generate(batch, max_new_tokens=200)
        for s, p in zip(batch, preds):
            print("-" * 60)
            print("GOLD :", s.get("answer", "")[-120:])
            print("PRED :", repr(p[:400]))
        if i >= 2:
            break
print("\nDONE_DEBUG")
