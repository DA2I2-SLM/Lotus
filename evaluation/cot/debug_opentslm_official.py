#!/usr/bin/env python3
"""Debug: load official OpenTSLM SP ckpt, try generate(), print full error."""
import sys, getpass, traceback
USER = getpass.getuser()
sys.path.insert(0, f"/home/{USER}/FoundationalTS/src")

import torch
from opentslm.model.llm.OpenTSLM import OpenTSLM
from opentslm.time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

print("\n=== Loading model ===")
model = OpenTSLM.load_pretrained("OpenTSLM/gemma-3-270m-sleep-sp",
                                  device=device, enable_lora=True)
model.eval()
print(f"Model type: {type(model).__name__}")
print(f"self.llm type: {type(model.llm).__name__}")
print(f"self.llm.generate exists: {hasattr(model.llm, 'generate')}")
print(f"self.llm.get_base_model exists: {hasattr(model.llm, 'get_base_model')}")

EOS = model.get_eos_token() if hasattr(model, "get_eos_token") else ""
ds = HARCoTQADataset(split="test", EOS_TOKEN=EOS)
row = ds[0]
print(f"\n=== Row 0 keys: {list(row.keys())} ===")
print(f"  pre_prompt len: {len(row['pre_prompt'])}")
print(f"  ts segments: {len(row['time_series'])}")

print("\n=== Trying model.generate([row], max_new_tokens=32) ===")
try:
    with torch.no_grad():
        out = model.generate([row], max_new_tokens=32)
    print(f"Return type: {type(out).__name__}")
    print(f"Return value (truncated): {str(out)[:200]}")
except Exception as e:
    print(f"!! Exception: {type(e).__name__}: {e}")
    traceback.print_exc()
