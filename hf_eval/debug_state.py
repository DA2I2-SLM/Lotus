#!/usr/bin/env python3
"""Diagnose the 'first sample coherent, rest garbage' bug.

Hypothesis: Gemma3 reuses a static/hybrid KV cache (self._cache) across
generate() calls, corrupting every call after the first. Test by varying call
order and cache settings on a fixed sample.
"""
import getpass
import sys

USER = getpass.getuser()
sys.path.insert(0, f"/home/{USER}/FoundationalTS/src")

import torch
from torch.utils.data import DataLoader
from opentslm.model.llm.OpenTSLM import OpenTSLM
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)
from opentslm.time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset

REPO = "OpenTSLM/gemma-3-270m-pt-sleep-sp"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = OpenTSLM.load_pretrained(REPO, enable_lora=True, device=device)
model.eval()
PS = getattr(model, "patch_size", 4)
print("patch_size:", PS)

eos = model.get_eos_token()
ds = SleepEDFCoTQADataset("test", EOS_TOKEN=eos)


def collate1(idx):
    b = [ds[idx]]
    return extend_time_series_to_match_patch_size_and_aggregate(b, patch_size=PS)


def gen(idx, **kw):
    batch = collate1(idx)
    with torch.no_grad():
        out = model.generate(batch, max_new_tokens=200, **kw)
    return out[0]


def garbage(t):
    low = (t or "").lower()
    return any(b in low for b in ("penurunan", "qualquer", "<unused", "attempt attempt", "тельное"))


def tag(t):
    return f"{'GARBAGE' if garbage(t) else 'OK'} | has_answer={'answer:' in (t or '').lower()} | {repr((t or '')[:90])}"


print("\n--- T1: sample 3 as the VERY FIRST call (default) ---")
print("idx3 first:", tag(gen(3)))

print("\n--- T2: now sample 0, then sample 3 again (default) ---")
print("idx0     :", tag(gen(0)))
print("idx3 2nd :", tag(gen(3)))

print("\n--- T3: sample 3 with cache_implementation='dynamic' ---")
try:
    print("idx3 dyn :", tag(gen(3, cache_implementation="dynamic")))
except Exception as e:
    print("dynamic failed:", repr(e))

print("\n--- T4: sample 3 with use_cache=False ---")
try:
    print("idx3 nocache:", tag(gen(3, use_cache=False)))
except Exception as e:
    print("use_cache=False failed:", repr(e))

print("\n--- T5: try resetting model._cache between calls (default kwargs) ---")
for idx in (0, 3, 5, 7):
    if hasattr(model.llm, "_cache"):
        model.llm._cache = None
    # also clear any cached generation state
    print(f"idx{idx} reset:", tag(gen(idx)))

print("\n--- T6: loop 6 samples with dynamic cache ---")
ok = 0
for idx in range(6):
    t = gen(idx, cache_implementation="dynamic")
    ok += 0 if garbage(t) else 1
print(f"dynamic-cache coherent: {ok}/6")

print("\nDONE_STATE")
