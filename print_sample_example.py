"""Print one joint_phase2_downstream training sample for presentation."""
import os
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import pandas as pd

# ── Config ──────────────────────────────────────────────────────────────────
CSV_PATH   = os.path.join(
    os.environ.get("FTS_ETT_DIR", f"/scratch/{os.environ.get('USER','')}/TimeSeriesECM/dataset"),
    "ETTh1.csv",
)
PATCH_SIZE = 32
HIST_LEN   = 512
PRED_LEN   = 96
N_SHOW_CH  = 2   # how many channels to display (ETTh1 has 7)

from opentslm.time_series_datasets.Phase2DownstreamDataset import (
    _make_sample, _split_indices
)

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH).select_dtypes(include="number")
N  = len(df)
tr_s, tr_e = _split_indices(N, "ett", "train")
train_df    = df.iloc[tr_s:tr_e]
global_mean = torch.tensor(train_df.mean().values, dtype=torch.float32)
global_std  = torch.tensor(train_df.std().values,  dtype=torch.float32).clamp(min=1e-5)

# Take the first window from the test split
te_s, te_e = _split_indices(N, "ett", "test")
window      = torch.tensor(
    df.iloc[te_s : te_s + HIST_LEN + PRED_LEN].values, dtype=torch.float32
)

N_PER_CH = PRED_LEN // PATCH_SIZE
fp_tokens = ["<future_patch>"] * N_PER_CH

sample = _make_sample(
    df_window    = window,
    col_names    = list(df.columns),
    history_len  = HIST_LEN,
    pred_len     = PRED_LEN,
    ds_name      = "ETTh1",
    eos_token    = "</s>",
    ch_sep_token = "<ch_sep>",
    fp_tokens    = fp_tokens,
    patch_size   = PATCH_SIZE,
    global_mean  = global_mean,
    global_std   = global_std,
)

SEP = "=" * 70

# ── Print ────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  JOINT PHASE-2 DOWNSTREAM  —  example training sample (ETTh1)")
print(SEP)

print(f"\n[PROMPT]")
print(f"  pre_prompt  : {sample['pre_prompt']}")
for c in range(min(N_SHOW_CH, sample['n_channels'])):
    ts   = sample['time_series'][c]
    txt  = sample['time_series_text'][c]
    mean, std = sample['time_series_norm_stats'][c]
    print(f"\n  channel {c}   : {txt}")
    print(f"  history     : len={len(ts)}  "
          f"first_5={[round(v,4) for v in ts[:5]]}  "
          f"last_5={[round(v,4) for v in ts[-5:]]}")
    print(f"  norm stats  : mean={mean:.4f}  std={std:.4f}")
if sample['n_channels'] > N_SHOW_CH:
    print(f"\n  ... ({sample['n_channels'] - N_SHOW_CH} more channels)")
print(f"\n  post_prompt : {sample['post_prompt']}")

print(f"\n[ANSWER TEMPLATE]")
print(f"  {sample['answer']}")
print(f"\n  Structure: <ts> [{N_PER_CH} × {sample['n_channels']} = {N_PER_CH * sample['n_channels']} <future_patch> tokens, time-major] </ts> </s>")
for t in range(N_PER_CH):
    ch_str = " ".join([f"enc[c{c},t{t}]" for c in range(min(sample['n_channels'], 4))])
    ellipsis = " ..." if sample['n_channels'] > 4 else ""
    print(f"    t={t}: {ch_str}{ellipsis}")

print(f"\n[FUTURE TARGETS  (ts_future)]")
ts_future = sample['ts_future']   # [C, pred_len]
print(f"  shape: {list(ts_future.shape)}  (channels × pred_len)")
for c in range(min(N_SHOW_CH, sample['n_channels'])):
    patches = ts_future[c].reshape(N_PER_CH, PATCH_SIZE)
    print(f"\n  channel {c}:")
    for k in range(N_PER_CH):
        p = patches[k]
        print(f"    patch {k}  (fp_{k}): [{', '.join(f'{v:.4f}' for v in p[:4].tolist())} ...]")

print(f"\n[SAMPLE META]")
print(f"  type       : {sample['type']}")
print(f"  dataset    : {sample['_dataset']}")
print(f"  history    : {sample['_history_len']} timesteps")
print(f"  pred       : {sample['_pred_len']} timesteps  ({N_PER_CH} patches × patch_size={PATCH_SIZE})")
print(f"  n_channels : {sample['n_channels']}")
print(f"  normalized : {sample['ts_already_normalized']}")
print(f"\n{SEP}\n")
