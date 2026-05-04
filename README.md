# Training with OpenTSLM

## Environment Setup

The training environment is managed with Conda. Activate it before running any commands:

```bash
conda create -n tsfm python=3.12
conda activate tsfm

```

Install dependencies from scratch:

```bash
pip install -r requirements.txt
```

You will also need a Hugging Face account with access to the base LLM (e.g. `google/gemma-3-270m`). Log in once with:

```bash
huggingface-cli login
```

---

## Training Entry Point

All training is driven by `curriculum_learning.py`. The key arguments are:

| Argument | Required | Description |
|---|---|---|
| `--model` | Yes | Model architecture. Use `OpenTSLMSPFixPos2D` for the standard setup. |
| `--stages` | Yes | Training stage(s) to run, e.g. `joint_phase1_extend`. |
| `--llm_id` | Yes | HuggingFace ID of the base LLM, e.g. `google/gemma-3-270m`. |
| `--num_epochs` | Yes | Number of training epochs (20 is the standard). |
| `--patch_size` | No | Patch size for time-series tokenisation (default: 32). Tested values: 16, 32, 96. |
| `--pe_type` | No | Positional encoding type. Use `sincos` for 2D sin-cos encoding. |
| `--batch_size` | No | Per-GPU batch size (default from `model_config.py`). |
| `--val_batch_size` | No | Validation batch size (default: same as train). |
| `--grad_accum_steps` | No | Gradient accumulation steps (default: 4). |
| `--warm_start_from` | No | Stage name to load weights from before training begins. |
| `--mse_alpha` | No | Fixed weight for MSE loss relative to CE loss (default: 1.0). |
| `--dynamic_loss_weighting` | No | Dynamically scale MSE to match CE magnitude each step. |
| `--max_samples_per_source` | No | Cap each dataset source (useful for smoke tests, e.g. `100`). |

---

## Single-GPU Quick Start

To run a quick smoke test on a single GPU:

```bash
python curriculum_learning.py \
  --model OpenTSLMSPFixPos2D \
  --stages joint_phase1_extend \
  --llm_id google/gemma-3-270m \
  --num_epochs 1 \
  --pe_type sincos \
  --patch_size 32 \
  --max_samples_per_source 100 \
  --grad_accum_steps 1
```

---

## Multi-GPU Training (SLURM)

Pre-written SLURM scripts are in `bashfiles/`. The standard `joint_phase1_extend` run uses 4 nodes x 2 GPUs:

```bash
sbatch bashfiles/joint_phase1/train_extend_patch96.sh
```

To pass a run tag:

```bash
sbatch bashfiles/joint_phase1/train_extend_patch96.sh my_run_tag
```

Logs are written to `output_job/<job-name>_<job-id>.out/.err`.

For a torchrun launch directly (without SLURM):

```bash
torchrun --nproc_per_node=2 curriculum_learning.py \
  --model OpenTSLMSPFixPos2D \
  --stages joint_phase1_extend \
  --llm_id google/gemma-3-270m \
  --num_epochs 20 \
  --pe_type sincos \
  --patch_size 32 \
  --val_batch_size 4 \
  --grad_accum_steps 8
```

---

## Available Training Stages

The `--stages` argument accepts one or more of the following. They can be chained (run sequentially) by listing them in order.

| Stage | Tasks | Notes |
|---|---|---|
| `phase1_chronos_mse` | Chronos forecasting only | MSE loss, LLM frozen |
| `joint_phase1_extend` | Chronos + TSQA + M4 | Three-task joint training, standard setup |
| `joint_phase1_extend_balanced` | Chronos + TSQA + M4 | Equal task weights (1:1:1) regardless of dataset size |
| `joint_phase1_extend_lorallm` | Chronos + TSQA + M4 | Adds LoRA to LLM, detaches LLM from MSE gradient |
| `joint_phase2` | Downstream fine-tuning | Encoder/projector frozen, only decoder head trains |

---

## Resuming and Warm-Starting

Training automatically resumes from the latest checkpoint if one exists for the requested stage. To explicitly warm-start from a previously trained stage:

```bash
--warm_start_from phase1_chronos_mse
```

---

## Evaluation Only

To run evaluation without training (requires an existing checkpoint):

```bash
python curriculum_learning.py \
  --model OpenTSLMSPFixPos2D \
  --stages joint_phase1_extend \
  --llm_id google/gemma-3-270m \
  --num_epochs 20 \
  --pe_type sincos \
  --patch_size 32 \
  --eval_only
```
