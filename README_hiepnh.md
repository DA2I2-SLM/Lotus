# LangTimeFoundationModel — Training & Evaluation Guide

This guide covers how to train and evaluate the **OpenTSLMSPFixPos2D** model from scratch on the Deakin HPC cluster (Slurm / V100 GPUs).

Pre-trained checkpoints are available on Hugging Face:

| Repo | Description |
|------|-------------|
| **[hiepnh137/LangTimeFoundationModel](https://huggingface.co/hiepnh137/LangTimeFoundationModel)** | All runs with optional LoRA adapters |
| **[hiepnh137/LangTimeFoundationModel-nolora](https://huggingface.co/hiepnh137/LangTimeFoundationModel-nolora)** | Best checkpoint — Transformer encoder, freq embed, mv-pos, **no LoRA** |

---

## Architecture overview

- **Backbone LLM:** `google/gemma-3-270m` (frozen by default; optional LoRA adapters)
- **Encoder/decoder variants:** MLP (`--enc_dec_type mlp`) or Transformer (`--enc_dec_type transformer`)
- **Patch size:** 32 (default for all production runs)
- **Positional encoding:** sinusoidal 2-D (`--pe_type sincos`)
- **Optional frequency branch:** `--use_freq_embed` adds a frequency-domain path to the MLP encoder

---

## Environment setup

All training and evaluation jobs activate the project-local conda env:

```bash
conda activate opentslm      # or: conda activate ./.env  (see bashfiles/_paths.sh)
```

The interpreter used inside Slurm scripts is:

```bash
ENV_PY="$(pwd)/.env/bin/python"
```

---

## Training

All scripts live under `bashfiles/joint_phase1/`. Submit them from the repo root with `sbatch`.

### Step 0 — build the data cache (CPU job, run once per stage variant)

Data pre-processing must finish before any GPU training job starts.

```bash
# For phase1_chronos_mse (nofreq or freq — data is identical)
sbatch bashfiles/joint_phase1/build_cache/build_cache_phase1_chronos_mse_nofreq.sh
sbatch bashfiles/joint_phase1/build_cache/build_cache_phase1_chronos_mse_freq.sh

# For joint_phase1_extend_balanced
sbatch bashfiles/joint_phase1/build_cache/build_cache_joint_phase1_extbal_nofreq.sh
sbatch bashfiles/joint_phase1/build_cache/build_cache_joint_phase1_extbal_freq.sh
```

The output is written to `_preprocessed_chronos_uv_synth_*` directories in the repo root. These directories must exist before the corresponding GPU job reads them.

> **Note:** always delete the checkpoint directory before re-submitting a training job to guarantee a clean start (the scripts do this automatically unless `KEEP_CKPT=1` is set).

---

### MLP encoder/decoder runs

#### Phase-1 Chronos MSE pretraining

| Variant | Script | Stage flag | Nodes x GPUs |
|---------|--------|-----------|--------------|
| No freq embed | `mlp/train_phase1_chronos_mse_nofreq.sh` | `phase1_chronos_mse` | 2 x 2 |
| With freq embed | `mlp/train_phase1_chronos_mse_freq.sh` | `phase1_chronos_mse` | 2 x 2 |
| No freq (full epochs) | `mlp/train_phase1_chronos_mse_nofreq_full.sh` | `phase1_chronos_mse` | 2 x 2 |

```bash
sbatch bashfiles/joint_phase1/mlp/train_phase1_chronos_mse_nofreq.sh
sbatch bashfiles/joint_phase1/mlp/train_phase1_chronos_mse_freq.sh
```

Checkpoints land in `results/sincos_patch32_mlp/gemma_3_270m/OpenTSLMSPFixPos2D/phase1_chronos_mse_{nofreq,freq}/checkpoints/`.

#### Joint phase-1 (extend-balanced, Chronos + TSQA + M4)

| Variant | Script | Nodes x GPUs |
|---------|--------|--------------|
| No freq | `mlp/train_mlp_joint_phase1_proc_synth.sh` | 2 x 4 |
| With freq | `mlp/train_mlp_joint_phase1_proc_synth_freq.sh` | 2 x 4 |
| No freq + LoRA | `mlp/train_mlp_joint_phase1_proc_synth_lora.sh` | 2 x 4 |
| With freq + LoRA | `mlp/train_mlp_joint_phase1_proc_synth_freq_lora.sh` | 2 x 4 |

```bash
sbatch bashfiles/joint_phase1/mlp/train_mlp_joint_phase1_proc_synth.sh
sbatch bashfiles/joint_phase1/mlp/train_mlp_joint_phase1_proc_synth_freq.sh
```

Checkpoints land in `results/sincos_patch32_mlp/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase1_extend_balanced_{nofreq,freq}/checkpoints/`.

---

### Transformer encoder/decoder runs

#### Phase-1 Chronos MSE

```bash
sbatch bashfiles/joint_phase1/trans_enc/train_trans_phase1_chronos_mse_nofreq.sh
sbatch bashfiles/joint_phase1/trans_enc/train_trans_phase1_chronos_mse_freq.sh
# Best checkpoint — freq embed + multivariate positional encoding, no LoRA (★ published on HF)
sbatch bashfiles/joint_phase1/trans_enc/train_trans_phase1_chronos_mse_freq_1p5x_mvpos.sh
```

Checkpoints for the `_mvpos` run land in `results/sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/phase1_chronos_mse_freq_mvpos/checkpoints/`.

#### Joint phase-1 extend-balanced

| Variant | Script |
|---------|--------|
| No freq | `trans_enc/train_trans_joint_phase1_extbal_nofreq.sh` |
| With freq | `trans_enc/train_trans_joint_phase1_extbal_freq.sh` |
| No freq + LoRA | `trans_enc/train_trans_joint_phase1_extbal_nofreq_lora.sh` |
| With freq + LoRA | `trans_enc/train_trans_joint_phase1_extbal_freq_lora.sh` |

```bash
sbatch bashfiles/joint_phase1/trans_enc/train_trans_joint_phase1_extbal_nofreq.sh
sbatch bashfiles/joint_phase1/trans_enc/train_trans_joint_phase1_extbal_freq.sh
```

Checkpoints land in `results/sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase1_extend_balanced_{nofreq,freq}/checkpoints/`.

---

### LoRA LLM variants (patch size 16)

Higher-resolution patch runs (patch_size=16) with full LLM LoRA fine-tuning:

```bash
sbatch bashfiles/joint_phase1/train_extend_patch16.sh          # patch16, frozen LLM
sbatch bashfiles/joint_phase1/train_extend_lorallm_patch16.sh  # patch16, LLM LoRA
sbatch bashfiles/joint_phase1/train_patch16_lora.sh            # patch16, LoRA joint
```

---

## Evaluation

All evaluation scripts live under `bashfiles/evaluate/joint_phase1/`. To run all transformer-encoder evaluations at once:

```bash
bash bashfiles/evaluate/joint_phase1/run_all.sh
```

### Time-series forecasting eval

Runs `evaluation/forecasting/fixed_inference.py` in autoregressive mode.
Evaluates on two data sources separately: **chronos_uv** (univariate Chronos) and **synthetic** (multivariate arrow files).

#### MLP encoder

```bash
# phase1_chronos_mse checkpoints
sbatch bashfiles/evaluate/joint_phase1/mlp/eval_time_series_phase1_chronos_mse_nofreq.sh
sbatch bashfiles/evaluate/joint_phase1/mlp/eval_time_series_phase1_chronos_mse_freq.sh

# joint_phase1_extend_balanced checkpoints
sbatch bashfiles/evaluate/joint_phase1/mlp/eval_time_series_joint_phase1_extbal_nofreq.sh
sbatch bashfiles/evaluate/joint_phase1/mlp/eval_time_series_joint_phase1_extbal_freq.sh
```

#### Transformer encoder

```bash
sbatch bashfiles/evaluate/joint_phase1/trans_enc/eval_time_series_trans_phase1_chronos_mse_nofreq.sh
sbatch bashfiles/evaluate/joint_phase1/trans_enc/eval_time_series_trans_phase1_chronos_mse_freq.sh
sbatch bashfiles/evaluate/joint_phase1/trans_enc/eval_time_series_trans_joint_phase1_extbal_nofreq.sh
sbatch bashfiles/evaluate/joint_phase1/trans_enc/eval_time_series_trans_joint_phase1_extbal_freq.sh
```

Results are written to `eval_results/<run_name>/time_series/{chronos_uv,synthetic}/metrics.json`.

---

### Language eval (TSQA MCQ + M4 captioning)

Runs `evaluation/baseline/evaluate_tsqa_opentslm.py` (multiple-choice accuracy) and `evaluation/captioning/m4_eval.py` (ROUGE).

#### MLP encoder

```bash
sbatch bashfiles/evaluate/joint_phase1/mlp/eval_language_joint_phase1_extbal_nofreq.sh
sbatch bashfiles/evaluate/joint_phase1/mlp/eval_language_joint_phase1_extbal_freq.sh
```

#### Transformer encoder

```bash
sbatch bashfiles/evaluate/joint_phase1/trans_enc/eval_language_trans_joint_phase1_extbal_nofreq.sh
sbatch bashfiles/evaluate/joint_phase1/trans_enc/eval_language_trans_joint_phase1_extbal_freq.sh
```

Results are written to `eval_results/<run_name>/language/{tsqa,m4}/`.

---

## Checkpoint paths summary

| Run | Encoder | Freq | Checkpoint dir |
|-----|---------|------|---------------|
| phase1_chronos_mse | MLP | No | `results/sincos_patch32_mlp/.../phase1_chronos_mse_nofreq/` |
| phase1_chronos_mse | MLP | Yes | `results/sincos_patch32_mlp_seed_42_freq/.../phase1_chronos_mse_freq/` |
| joint_phase1_extbal | MLP | No | `results/sincos_patch32_mlp/.../joint_phase1_extend_balanced_nofreq/` |
| joint_phase1_extbal | MLP | Yes | `results/sincos_patch32_mlp_seed_42_freq/.../joint_phase1_extend_balanced_freq/` |
| joint_phase1_extbal | Transformer | No | `results/sincos_patch32/.../joint_phase1_extend_balanced_nofreq/` |
| joint_phase1_extbal | Transformer | Yes | `results/sincos_patch32_seed_42_freq/.../joint_phase1_extend_balanced_freq/` |
| **phase1_chronos_mse_freq_mvpos** ★ | **Transformer** | **Yes + mv-pos** | `results/sincos_patch32/gemma_3_270m/OpenTSLMSPFixPos2D/phase1_chronos_mse_freq_mvpos/` |

Pre-trained versions of all the above are published at **[hiepnh137/LangTimeFoundationModel](https://huggingface.co/hiepnh137/LangTimeFoundationModel)**.

The ★ run (no LoRA) is also separately published at **[hiepnh137/LangTimeFoundationModel-nolora](https://huggingface.co/hiepnh137/LangTimeFoundationModel-nolora)** — this is the best-performing checkpoint evaluated on GIFT-Eval (see below).

---

## GIFT-Eval benchmark results

The `phase1_chronos_mse_freq_mvpos` checkpoint was evaluated on all **97 official GIFT-Eval configurations** (zero-shot, no fine-tuning). Evaluation script: `evaluation/forecasting/gift_eval_eval.py`.

### Summary (lower is better)

| Model | Short MASE | Medium MASE | Long MASE | Overall MASE | Overall WQL |
|-------|-----------|------------|----------|-------------|------------|
| Moirai-Large | 2.000 | 1.635 | 1.678 | 1.850 | 0.269 |
| Chronos-Large | 2.015 | 1.791 | 1.756 | 1.910 | 0.291 |
| Chronos-Base | 2.017 | 1.838 | 1.781 | 1.927 | 0.291 |
| Chronos-Small | 2.027 | 1.847 | 1.836 | 1.947 | 0.294 |
| Moirai-Small | 2.166 | 1.680 | 1.691 | 1.958 | 0.289 |
| SeasonalNaive | 2.377 | 1.425 | 1.422 | 1.964 | 0.352 |
| Moirai-Base | 2.021 | 2.005 | 2.028 | 2.019 | 0.295 |
| TTM-R2-PT | 2.641 | 1.814 | 1.838 | 2.288 | 0.337 |
| **Ours (nolora)** | **2.946** | **2.200** | **2.233** | **2.630** | **0.354** |
| TimesFM-1.0 | 2.359 | 5.113 | 8.002 | 4.177 | 0.491 |

Full per-config results are in `results/OpenTSLMSPFixPos2D/all_results.csv`. Comparison Excel: `results/gift_eval_comparison.xlsx` and `results/comparison_tables.xlsx` (sheets C1–C3).

To reproduce:

```bash
# 5-task SLURM array (V100 GPUs, ~12 h total)
sbatch bashfiles/evaluate/joint_phase1/eval_gift_eval.sh
# After jobs finish, merge and summarise:
python evaluation/forecasting/merge_gift_eval_results.py
# Compare against paper baselines:
python evaluation/forecasting/compare_gift_eval.py
```

---

## Common flags reference

| Flag | Description |
|------|-------------|
| `--model OpenTSLMSPFixPos2D` | Model class (used in all production runs) |
| `--llm_id google/gemma-3-270m` | LLM backbone |
| `--stages <name>` | Training stage / dataset schedule |
| `--enc_dec_type mlp\|transformer` | Encoder/decoder architecture |
| `--patch_size 32` | Patch size in time-series tokens |
| `--pe_type sincos` | 2-D sinusoidal positional encoding |
| `--use_freq_embed` | Add frequency-domain branch to MLP encoder |
| `--num_epochs 100` | Number of training epochs |
| `--val_batch_size 4` | Validation batch size per GPU |
| `--grad_accum_steps 8` | Gradient accumulation steps |

---

## HF authentication

Training scripts check for `HF_TOKEN` in the environment first, then fall back to `~/.cache/huggingface/token`. For gated models (Gemma), set your token before submitting:

```bash
export HF_TOKEN=<your_token>
sbatch bashfiles/joint_phase1/mlp/train_phase1_chronos_mse_nofreq.sh
```
