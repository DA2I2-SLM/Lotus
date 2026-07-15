#!/usr/bin/env python3
"""Pin down the correct eval config for the OpenTSLM SP checkpoint.

Tests batch_size x patch_size combinations on a few Sleep test samples and
reports, for each, how many of N outputs are coherent (contain 'Answer:' and
no obvious garbage) plus the running accuracy. Also prints the model's OWN
encoder patch_size so we can match the collate to it.
"""
import getpass
import sys

USER = getpass.getuser()
sys.path.insert(0, f"/home/{USER}/FoundationalTS/src")

import torch
from torch.utils.data import DataLoader
from opentslm.model.llm.OpenTSLM import OpenTSLM
from opentslm.model_config import PATCH_SIZE as CFG_PATCH
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)
from opentslm.time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset

REPO = "OpenTSLM/gemma-3-270m-pt-sleep-sp"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}  CFG_PATCH(model_config)={CFG_PATCH}")

model = OpenTSLM.load_pretrained(REPO, enable_lora=True, device=device)
model.eval()


def detect_patch_size(m):
    cands = [m, getattr(m, "encoder", None),
             getattr(getattr(m, "model", None), "vision_encoder", None),
             getattr(m, "vision_encoder", None)]
    for obj in cands:
        if obj is None:
            continue
        ps = getattr(obj, "patch_size", None)
        if isinstance(ps, int) and ps > 0:
            return ps, f"{type(obj).__name__}.patch_size"
        pe = getattr(obj, "patch_embed", None)
        if pe is not None and hasattr(pe, "kernel_size"):
            ks = pe.kernel_size
            ks = ks[0] if isinstance(ks, (tuple, list)) else ks
            return int(ks), f"{type(obj).__name__}.patch_embed.kernel_size"
    return None, "not found"


ps_detected, src = detect_patch_size(model)
print(f"DETECTED encoder patch_size = {ps_detected}  (via {src})")

eos = model.get_eos_token()
ds = SleepEDFCoTQADataset("test", EOS_TOKEN=eos)

# inspect raw lengths of first 16 test series
raw_lens = []
for i in range(16):
    ts = ds[i]["time_series"]
    try:
        raw_lens.append(int(torch.as_tensor(ts[0]).shape[0]))
    except Exception:
        raw_lens.append(-1)
print("first16 raw series lengths:", raw_lens)


def looks_garbage(txt: str) -> bool:
    if not txt:
        return True
    low = txt.lower()
    bad = ("penurunan", "qualquer", "<unused", "attempt attempt", "тельное", "thụ")
    return any(b in low for b in bad)


def run_cfg(bs, ps, n=12, max_new=256):
    sub = torch.utils.data.Subset(ds, list(range(n)))
    loader = DataLoader(sub, batch_size=bs, shuffle=False,
                        collate_fn=lambda b: extend_time_series_to_match_patch_size_and_aggregate(b, patch_size=ps))
    coh = 0
    corr = 0
    tot = 0
    with torch.no_grad():
        for batch in loader:
            preds = model.generate(batch, max_new_tokens=max_new)
            for s, p in zip(batch, preds):
                tot += 1
                if (not looks_garbage(p)) and ("answer:" in p.lower()):
                    coh += 1
                gold = (s.get("label") or "").strip().lower()
                pl = p.lower()
                if gold and ("answer:" in pl) and gold in pl.split("answer:")[-1]:
                    corr += 1
    return coh, corr, tot


print("\n==== CONFIG SWEEP (n=12 sleep samples) ====")
for bs in (1, 8):
    for ps in sorted({4, CFG_PATCH, ps_detected or 4}):
        coh, corr, tot = run_cfg(bs, ps)
        print(f"  batch_size={bs:>2}  patch_size={ps:>2}  ->  coherent {coh}/{tot}   correct {corr}/{tot}", flush=True)
print("\nDONE_SWEEP")
