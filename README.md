# Training with OpenTSLM

> **Evaluating the released baseline checkpoints?** Jump to
> [Baseline checkpoints: mechanisms and evaluation guide](#baseline-checkpoints-mechanisms-and-evaluation-guide)
> — the entry point is `bashfiles/evaluate/eval_baselines.sh`.

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
sbatch bashfiles/joint_phase1/train_extend_patch32.sh
```

To pass a run tag:

```bash
sbatch bashfiles/joint_phase1/train_extend_patch32.sh my_run_tag
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

---

# Baseline checkpoints: mechanisms and evaluation guide

Reproduction instructions for the FoundationalTS results table. The checkpoint
files themselves are hosted at https://huggingface.co/daidv1112/FoundationalTS.
All `opentslm-*` files derive from `google/gemma-3-270m` and the released
OpenTSLM checkpoints (`OpenTSLM/gemma-3-270m-m4-{sp,flamingo}`, apache-2.0);
use is subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

---

## 1. How each baseline works

### OpenTSLM-SP (Soft Prompt) — `opentslm-sp_*.pt`
A **frozen** gemma-3-270m reads a text prompt in which each time series appears
as a span of continuous "soft tokens" between two added special tokens
`<ts>...</ts>`. The soft tokens come from a small patch encoder
(patch size **4**, embed dim **128**, learned positional embedding) followed by
a projector into the LLM embedding space (128 -> 640). Trainable in phase 2:
encoder + projector + LoRA r=4/alpha=8 on the backbone = **3.1M** parameters.
The checkpoint stores ONLY encoder + projector + LoRA state + the resized input
embeddings — the backbone itself is not included (see §3).

### OpenTSLM-Flamingo — `opentslm-flamingo_*.pt`
The series is encoded by the same patch encoder, but instead of splicing soft
tokens into the prompt, a **perceiver resampler** compresses the patch sequence
and **gated cross-attention layers** interleaved with the frozen LLM's blocks
attend to it (open-flamingo style). Trainable: cross-attention + perceiver
= **317M** parameters. The checkpoint stores the full `model_state` (~3.3 GB).

### The two training protocols (the suffix on every filename)
* **`joint-5task-finetune`** (our recipe): ONE phase-2 stage on all five tasks
  simultaneously — HAR, Sleep-EDF, ECG-QA, PAMAP2, TimeMQA — each resampled to
  40,000 rows/epoch; best model by mean generative exact-match on a
  192-sample/task validation basket; early stop when any task stagnates
  8 epochs. Warm-started from `OpenTSLM/gemma-3-270m-m4-*`.
* **`sequential-curriculum`** (OpenTSLM's published recipe): one task at a
  time (released ECG checkpoint -> PAMAP2 -> TimeMQA), patience 5, best model
  by validation loss. Included to demonstrate catastrophic forgetting: the
  terminal model retains essentially only its final stage.

### Tokenized time-series, zero-shot (no checkpoint — by construction)
The **unmodified base LLM** receives the series serialized as text with
Gruver et al. (2023)'s digit-level encoding (` 1 2 3 , -4 5 0 ,` — digit-spaced,
no decimal point, per-series rescale) and is evaluated directly. There are no
weights to distribute; reproduction = base model + the eval code.

### Tokenized finetuned (LoRA) — adapters to be added when training completes
Same Gruver serialization, backbone fine-tuned with LoRA r=8/alpha=16
(q,k,v,o,gate,up,down projections), best model by **validation loss**.
This is the OpenTSLM paper's "Tokenized finetuned" baseline.

---

## 2. Environment and code (Deakin cluster)

```bash
module load Anaconda3 && conda activate opentslm       # transformers 4.57.3, peft 0.17.1
CODE=/home/s223540177/FoundationalTS                    # evaluation code
DATA=/scratch/s223540177/FoundationalTS/data            # HAR/Sleep/ECG/PAMAP2 CoT csvs
TIMEMQA=/scratch/s223540177/timemqa_data                # TimeMQA csv
MANIFEST=/vast/s223540177/cik_work/eval_manifests/ecg_test_first5000.json
```

The ECG test split is pinned to a **5000-sample manifest** with per-sample gold
sha1 hashes; the eval hard-fails on any mismatch, so all models are guaranteed
to be scored on identical rows. Do not evaluate ECG without it.

## 3. Evaluating each checkpoint

**The evaluation entry point is one bash file:**
[`bashfiles/evaluate/eval_baselines.sh`](https://github.com/Davido111200/FoundationalTS/blob/main/bashfiles/evaluate/eval_baselines.sh)
in the code repo (`git clone https://github.com/Davido111200/FoundationalTS` —
private; ask Dai for access).

```bash
cd FoundationalTS
# Soft-Prompt checkpoint on HAR:
sbatch --job-name=sp.har --time=06:00:00 bashfiles/evaluate/eval_baselines.sh \
  sp ./opentslm-sp_gemma-3-270m_joint-5task-finetune.pt ./eval_out/sp_har har

# Flamingo checkpoint on ECG (pinned 5000-sample manifest applied automatically):
sbatch --job-name=fl.ecg --time=06:00:00 bashfiles/evaluate/eval_baselines.sh \
  fl ./opentslm-flamingo_gemma-3-270m_joint-5task-finetune.pt ./eval_out/fl_ecg ecg

# SP is slow (~10 s/sample) -- shard long tasks by row range:
sbatch --export=ALL,EVAL_SHARD_START=0,EVAL_SHARD_END=1000 --time=06:00:00 \
  bashfiles/evaluate/eval_baselines.sh sp <ckpt> ./eval_out/sp_har_s0 har
```

Runner argument: `sp` for `opentslm-sp_*.pt`, `fl` for `opentslm-flamingo_*.pt`.
The script sets the required encoder geometry itself, applies the pinned ECG
manifest when the task is `ecg`, and writes
`<out_dir>/<task>_test_predictions.jsonl`.

### Tokenized zero-shot (no checkpoint)
```bash
python $CODE/baselines/tokenized_llm/tokenized_sft.py \
  --mode eval --model_id google/gemma-3-1b-pt \
  --out_dir ./eval_out/zs_har --split test --tasks har \
  --max_len 4096 --max_values 512 --serializer llmtime
```
`--serializer llmtime` selects the Gruver encoding; the port is
`baselines/tokenized_llm/serialize_llmtime.py`, verified against the original
docstring examples. Long series are decimated to fit the token budget and the
stride is recorded per sample.

### Outputs and scoring
Every eval writes `<task>_test_predictions.jsonl` with
`idx / gold / pred / label_gold / label_pred`. Scoring:
* label accuracy = case-insensitive exact match of `label_pred` vs `label_gold`
  (the extractor takes the LONGEST-candidate prefix after the final `Answer:`;
  beware: greedy regexes glue trailing text onto answers and cost one baseline
  28pp before we fixed it);
* TimeMQA is scored by its own `question_format` column (MCQ letter accuracy
  over 469 lettered golds; true/false stance accuracy over 329), joined on gold
  TEXT, not idx;
* we additionally report an LLM-judge score per cell (Claude Haiku, T=0,
  closed label set, commit-or-fail rubric) — judged and exact-match numbers
  agree within ~3pp after the extractor fix.

## 4. Expected results (verify your reproduction)

| checkpoint | HAR | Sleep | ECG | PAMAP2 | MCQ | T/F |
|---|---:|---:|---:|---:|---:|---:|
| sp joint-5task | 54.4 | 72.7 | 37.6 | 52.1 | 38.6 | 68.4 |
| flamingo joint-5task | 62.6 | 70.4 | 31.1 | 54.7 | 12.4 | 3.0 |
| sp sequential-curriculum | 1.5 | 0.0 | 17.0 | 0.0 | 41.4 | 70.5 |
| flamingo sequential-curriculum | 3.8 | 0.0 | 5.4 | 10.3 | 20.0 | 4.9 |
| tokenized zero-shot (1b) | 0.0 | 0.0 | 0.0* | 0.0 | 5.8 | 10.3 |

\*exact-match may report up to ~3.5 on ECG from label words echoed inside
degenerate output; the judge scores it 0.04. Majority-class floors:
HAR 28.3 / Sleep 45.3 / ECG 19.3 / PAMAP2 12.8 / MCQ 43.3 / T-F 59.3.

## 5. Where things live

| what | where |
|---|---|
| checkpoint files | https://huggingface.co/daidv1112/FoundationalTS |
| evaluation entry point | `bashfiles/evaluate/eval_baselines.sh` (this repo) |
| pinned ECG manifest | `eval_manifests/ecg_test_first5000.json` (this repo) |
| datasets (cluster) | `/scratch/s223540177/FoundationalTS/data`, `/scratch/s223540177/timemqa_data` |

Questions: Dai (daidv1112) — training logs, per-epoch curves, launch scripts
and all per-cell predictions (both metrics) are archived and available.
