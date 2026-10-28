<!--
SPDX-FileCopyrightText: 2026 Deakin University and the LOTUS authors
SPDX-License-Identifier: MIT
-->

# scripts/analysis/memory — memory-scaling analysis

Backs the paper's **Fig. 7** and the Appendix section **"Computational Cost"**
(arXiv 2610.18877): how the peak training memory of the soft-prompt interface
grows with the length `L` of each series and the number of series `N` fed to
the encoder, for each backbone.

## What is measured

`get_memory_use.py` builds one model, one dataset, runs `--steps` training
steps at batch size 1 (forward, backward, AdamW step on the trainable
parameters) and appends one row to a CSV with

| column | meaning |
|---|---|
| `peak_cuda_gb` | `torch.cuda.max_memory_allocated()` over the steps |
| `peak_cuda_reserved_gb` | `torch.cuda.max_memory_reserved()` (what the plots use) |
| `nvml_peak_gb` | per-process usage seen by the NVIDIA driver (needs `pynvml`; `-1` otherwise) |
| `status` / `error` | `ok`, or `error` with the exception text (an OOM is recorded, not fatal) |

The synthetic `SimulationQADataset` (`--dataset SimulationQADataset --length L
--num_series N`) gives the controlled `(L, N)` grid; the real CoT datasets
(`HARCoTQADataset`, `SleepEDFCoTQADataset`, `ECGQACoTQADataset`, `TSQADataset`)
can be measured the same way.

Model types:

* `OpenTSLMSPFixPos2D` — LOTUS. Defaults are the released 270M configuration
  (MLP encoder 64/256, frequency branch, patch 32, `--pe_base_c 10`, Phase-2
  LoRA r=4/alpha=8). For the 1B / Qwen3-0.6B runs pass the transformer flags
  from `configs/gemma3_1b.env` / `configs/qwen3_0_6b.env`
  (`--enc_dec_type transformer --lora_r 8 --lora_alpha 16`; encoder dims default
  to the paper's 192 x 6 x 8 / ff 896).
* `OpenTSLMSP` — the OpenTSLM soft-prompt baseline (encoder + projector trained).
* `OpenTSLMFlamingo` — the OpenTSLM cross-attention baseline (needs `open_flamingo`).

## How to run

```bash
# one cell
"${FTS_PYTHON}" scripts/analysis/memory/get_memory_use.py \
    --llm_id google/gemma-3-270m --model OpenTSLMSPFixPos2D \
    --dataset SimulationQADataset --length 1000 --num_series 3 --device cuda

# full sweep (SLURM; backbones x models x L x N, see the header for MEM_* overrides)
sbatch scripts/analysis/memory/run_all_memory.sh
MEM_LLM_IDS="google/gemma-3-1b-pt" MEM_EXTRA="--enc_dec_type transformer --lora_r 8 --lora_alpha 16" \
    sbatch scripts/analysis/memory/run_all_memory.sh

# plots (png + pdf)
"${FTS_PYTHON}" scripts/analysis/memory/plot_memory_scaling.py \
    --csv "$FTS_RESULTS_ROOT/memory/memory_use.csv" --out memory_scaling          # Fig. 7: memory vs N, one column per L
"${FTS_PYTHON}" scripts/analysis/memory/plot_memory_total_length.py \
    --csv "$FTS_RESULTS_ROOT/memory/memory_use.csv" --out memory_total_length     # memory vs N x L, one panel per backbone
```

The CSV defaults to `$FTS_RESULTS_ROOT/memory/memory_use.csv`
(`bashfiles/_paths.sh` convention); runs append, and the plots keep the latest
measurement of each `(family, backbone, L, N)` cell. Runs whose `status` is not
`ok` or whose peak exceeds `--oom_threshold` (180 GB) are drawn as a dashed
extension to a red "OOM" marker.

## Provenance

These files are the OpenTSLM `evaluation/memory/` harness (removed from this
fork in the release cleanup) restored under `scripts/analysis/memory/` with:
repo-relative imports, `FTS_*` paths and the `_paths.sh` preamble, the LOTUS
model as a measurable `--model`, lazy imports so the Flamingo baseline's
`open_flamingo` dependency is only needed when that model is requested, and a
recorded-not-fatal OOM. Upstream's `plot_memory_scaling.py` was truncated
mid-statement in every branch, so the scaling plot here is the complete sibling
`plot_memory_simulation_per_length.py`, generalised. The upstream
`plot_memory_usage.py` (bar chart over the real datasets) was not restored; it is
still available via `git show origin/main:evaluation/memory/plot_memory_usage.py`.
