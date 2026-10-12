# LOTUS: Learning to Understand Time-series by Learning to Forecast

LOTUS is a time-series language model trained under **forecast grounding**: a pretraining
principle that requires language-aligned time-series representations to remain predictive
of the future evolution of the signal before they are adapted to downstream tasks.

Time-series language models are usually aligned through language supervision alone —
labels, captions, instructions, question–answer pairs. That supervision encourages the
representation to preserve whatever is needed to produce the correct text, but it does not
require it to retain *how the underlying signal evolves*. Two motion segments may both be
"walking" while differing in speed, local dynamics, and likely continuation. Forecast
grounding adds that missing constraint: during pretraining, the representation handed to
the language model must also predict the signal's next patch.

Across three language-model backbones and six downstream evaluation splits, LOTUS attains
the best average accuracy in every matched backbone group while training only a small
number of parameters, reaching **59.91%** six-task average accuracy and outperforming the
strongest matched baseline by **3.36%**.

---

## Method

### Architecture

A multivariate series `S ∈ R^(C×L)` is normalized per channel over its observed extent,
zero-padded to a multiple of the patch length `P`, and split into non-overlapping patches
`x_{c,n} ∈ R^P`, giving `C·N` patch positions with `N = ⌈L/P⌉`.

Each patch is represented in **both temporal and spectral coordinates**. The raw patch is
the temporal view; for the spectral view the real FFT is split into real and imaginary
parts and concatenated:

```
f_{c,n} = [Re F(x_{c,n}) ; Im F(x_{c,n})] ∈ R^(2(P/2+1))
```

The two views are embedded by learnable projections `W_t` and `W_f` and fused by `W_fuse`:

```
h^t_{c,n} = W_t x_{c,n}      h^f_{c,n} = W_f f_{c,n}
h_{c,n}   = W_fuse [ h^t_{c,n} ; h^f_{c,n} ]
```

The fused representation is layer-normalized and given a **two-dimensional positional
encoding** that identifies both the channel and the temporal location of the patch — a
channel embedding `p^ch_c` plus a patch-position embedding `p^pos_n`:

```
h̃_{c,n} = LN(h_{c,n}) + p^ch_c + p^pos_n
```

A Transformer then maps the patch representation to the language-model width `d_model`,
producing `C·N` **soft time-series tokens** `z_{c,n} = g_φ(x_{c,n}, c, n)`.

Soft tokens are inserted into the language-model input in the original order of the
multimodal example, each series enclosed by learned `<ts>` and `</ts>` delimiters, with
any intervening text retained between them. The resulting sequence is processed by the
pretrained causal language model.

### Phase 1 — forecast-grounding pretraining

Phase 1 mixes forecasting, captioning and time-series question answering. The encoder and
projector train jointly with a **next-patch decoder**, while the language model is adapted
through LoRA.

- **Language objective** — standard autoregressive cross-entropy over answer positions,
  updating the encoder parameters `φ` and the LoRA parameters `η`.
- **Forecasting objective** — the next-patch decoder `d_ψ` predicts the normalized
  succeeding patch from the **final hidden states of the LLM** at the time-series token
  positions. It projects those hidden states back to the time-series representation space,
  applies a causally masked Transformer decoder, and emits the `P` values of the next
  patch, so prediction at position `n` conditions on `H_{u(c,n)}` and on preceding states
  of the same channel. Gradients propagate through the language-model computation and
  jointly update `φ`, `η` and `ψ`.

Forecasting and language batches are drawn separately, so each optimization step uses one
objective or the other. Because the corpora differ greatly in size, smaller sources are
upsampled and larger ones downsampled according to fixed mixture weights, making the
language–forecasting balance independent of corpus size.

### Phase 2 — downstream adaptation

The next-patch decoder is **removed**. One model is adapted jointly across the downstream
corpora with the language objective alone, fully fine-tuning the encoder and projector
while applying LoRA to the LLM. The corpora are balanced by sampling 40,000 examples from
each per epoch, so no dataset dominates the mixture.

---

## Results

Accuracy on the six downstream splits. `# Params` counts trainable parameters in
downstream adaptation; `Avg.` is the mean over the six splits.

| Backbone | # Params | HAR | Sleep | ECG | PAMAP2 | MCQ | T/F | **Avg.** |
|---|---|---|---|---|---|---|---|---|
| Gemma3-270M | 1.9M | 58.38 | 70.54 | 33.16 | 55.15 | 39.41 | 74.47 | **55.19** |
| Gemma3-1B | 11.1M | 62.67 | 77.10 | 37.66 | 52.56 | 44.98 | 78.42 | **58.90** |
| Qwen3-0.6B | 9.3M | 63.66 | 77.20 | 31.58 | 53.42 | 56.72 | 76.90 | **59.91** |

On Qwen3-0.6B, LOTUS reaches 59.91% — 4.58% above a fully fine-tuned TS-Reasoner-7B while
being 14× smaller in backbone size. On Gemma3-270M it comes within 0.14% of that same
7B system (55.19% vs. 55.33%) with an order of magnitude fewer trainable parameters.

---

## Installation

Python ≥ 3.12 is required.

```bash
git clone https://github.com/Davido111200/FoundationalTS.git
cd FoundationalTS

# uv (declared build backend)
uv sync

# or conda + pip
conda create -n lotus python=3.12 -y
conda activate lotus
pip install -e .
```

The GIFT-Eval zero-shot forecasting scripts additionally need:

```bash
pip install gluonts
```

Log in once for gated base models (Gemma backbones require accepting the licence):

```bash
huggingface-cli login
```

---

## Paths and environment

Every launcher resolves its paths through `bashfiles/_paths.sh`, so the same script runs on
any account without edits. Each value is overridable from the environment:

| Variable | Meaning |
|---|---|
| `FTS_REPO_ROOT` | Repository checkout. Auto-detected from `SLURM_SUBMIT_DIR`, the script location, or `$PWD`. |
| `FTS_PYTHON` / `FTS_TORCHRUN` | Interpreter and launcher. Defaults to `.env/` then the `opentslm` conda env. |
| `FTS_CONDA_ENV` | Conda environment name (default `opentslm`). |
| `FTS_SCRATCH` / `FTS_VAST` | Writable scratch bases (default `/scratch/$USER`, `/vast/$USER`). |
| `FTS_RESULTS_ROOT` | Checkpoints and results. |
| `FTS_PRETRAINED_DIR` | Phase-1 checkpoints that Phase 2 warm-starts from. |
| `FTS_HF_HUB` | HuggingFace hub cache (`$HF_HOME/hub`, else `~/.cache/huggingface/hub`). |
| `FTS_SYNTHETIC_DIR` | Synthetic multivariate corpus (`unified_*.arrow`). Colon-separated for multiple directories. |
| `FTS_ETT_DIR` | ETT csv directory. |
| `FTS_GIFT_DATA` | GIFT-Eval data root. |
| `CHRONOS_UV_BASE_PATH` | Chronos/Monash arrow datasets root. |
| `CHRONOS_PREPROCESSED_SPLITS_DIR` | Preprocessed split cache. Use a separate directory per run configuration. |

---

## Data

**Phase 1** draws from four corpora:

| Corpus | Role | Where the code looks |
|---|---|---|
| Monash half-hourly electricity demand (five Australian states) | forecasting | `CHRONOS_UV_BASE_PATH` |
| Synthetic multivariate corpus (844,275 series) | forecasting | `FTS_SYNTHETIC_DIR` |
| TSQA multiple-choice (trend, volatility, seasonality, outliers) | language | HuggingFace |
| M4 series with generated captions | language | HuggingFace |

The synthetic corpus combines four generator families, none contributing more than 37%:

| Generator family | Series | Share | Variates | Length |
|---|---|---|---|---|
| Chronos-2 multivariatizer | 310,000 | 36.7% | 3–10 | 512–8,192 |
| KernelSynth (univariate) | 299,999 | 35.5% | 1 | 1,024–4,096 |
| TimePFN LMC | 215,000 | 25.5% | 3–10 | 512–2,048 |
| SymTime S2Generator | 19,276 | 2.3% | 3–10 | 1,024–4,096 |
| **Total** | **844,275** | **100%** | 1–10 | 512–8,192 |

Raw Chronos arrow datasets can be fetched on a login node (compute nodes have no network):

```bash
python scripts/prefetch_chronos_uv.py --out_dir "$FTS_VAST/FoundationalTS/_chronos_uv_datasets"
export CHRONOS_UV_BASE_PATH="$FTS_VAST/FoundationalTS/_chronos_uv_datasets"
```

**Phase 2** adapts jointly on HAR, PAMAP2, Sleep-EDF, ECG-QA and TimeMQA, whose MCQ and
T/F splits are evaluated separately. The code reads them from
`$FTS_SCRATCH/FoundationalTS/data`.

---

## Reproducing Phase 1

### 1. Build the split cache (CPU only)

Dataset construction is the expensive part, so build it once on CPU and let the GPU job
start training immediately. A build that fails to produce a split now exits non-zero.

```bash
sbatch bashfiles/joint_phase1/build_cache/build_cache_joint_phase1_extbal_freq.sh
```

Equivalently, directly:

```bash
export CHRONOS_PREPROCESSED_SPLITS_DIR=_preprocessed_chronos_uv_synth_joint_extbal
python curriculum_learning.py \
    --model OpenTSLMSPFixPos2D \
    --stages joint_phase1_extend_balanced_lora_r8 \
    --llm_id google/gemma-3-270m \
    --num_epochs 1 --patch_size 32 --pe_type sincos \
    --enc_dec_type transformer --use_freq_embed \
    --synthetic_data_dir "$FTS_SYNTHETIC_DIR" \
    --build_cache_only
```

### 2. Train

One launcher per backbone in the results table, all running the stage
`joint_phase1_extend_balanced_lora_r8`:

| Backbone | Launcher |
|---|---|
| Gemma3-270M | `bashfiles/joint_phase1/trans_enc/train_trans_joint_phase1_extbal_lora_r8_nofreq_mvpos.sh` |
| Gemma3-1B | `bashfiles/joint_phase1/trans_enc/train_trans_joint_phase1_extbal_lora_r8_freq_mvpos_gemma3_1b.sh` |
| Qwen3-0.6B | `bashfiles/joint_phase1/trans_enc/train_trans_joint_phase1_extbal_lora_r8_freq_mvpos_qwen3_06b.sh` |

```bash
export FTS_SYNTHETIC_DIR=/path/to/synthetic/unified
TASK_WEIGHT_CHRONOS_UV=1 TASK_WEIGHT_SYNTHETIC=2 \
TASK_WEIGHT_TSQA=1       TASK_WEIGHT_M4=1 \
  sbatch bashfiles/joint_phase1/trans_enc/train_trans_joint_phase1_extbal_lora_r8_freq_mvpos_qwen3_06b.sh
```

The four `TASK_WEIGHT_*` variables set the mixture; the paper uses **1:2:1:1**, i.e. 60%
numeric and 40% linguistic optimization steps.

`--run_suffix <tag>` writes to a separate results directory, so two configurations of the
same stage can train in parallel without clobbering each other's checkpoints.

Checkpoints land under
`$FTS_RESULTS_ROOT/sincos_patch32/<llm>/OpenTSLMSPFixPos2D/<stage><suffix>/checkpoints/`.

---

## Reproducing Phase 2

Phase 2 warm-starts from a Phase-1 checkpoint and drops the next-patch decoder.

```bash
export WARM_START_PRETRAINED_CKPT="$FTS_PRETRAINED_DIR/joint_phase1_extend_balanced_freq/checkpoints/best_model.pt"
sbatch bashfiles/joint_phase2/joint5/train_joint5_ours.sh
```

The launcher runs the stage `joint_phase2_downstream` with `--warm_start_from pretrained`.
Its behaviour is controlled by:

| Variable | Meaning |
|---|---|
| `PHASE2_COT_TASKS` | Comma-separated task list, e.g. `har,sleep,ecg,pamap2,timemqa`. |
| `PHASE2_COT_MIX` | Mixture strategy; `balanced_all` for the balanced joint mixture. |
| `PHASE2_BALANCE_TARGET` | Examples per task per epoch (paper: `40000`). |
| `PHASE2_COT_ONLY` | Restrict to chain-of-thought corpora. |
| `PHASE2_INCLUDE_ECG`, `PHASE2_INCLUDE_PAMAP2` | Include those corpora. |
| `PHASE2_SELECT_BY_GEN`, `PHASE2_GEN_EM_N` | Select checkpoints by mean generative exact match over N validation items (paper: 192). |
| `PHASE2_EARLY_STOP_PAT` | Per-task early-stopping patience (paper: 8). |
| `SLEEP_UPSAMPLE` | Upsampling factor for the Sleep corpus. |
| `RESULTS_BASE_DIR` | Output root for the run. |

---

## Evaluation

All systems are evaluated under **greedy decoding**. Each response contains a rationale
followed by an explicit answer field; the parser extracts that field, applies case folding
and whitespace normalization, and maps dataset-specific variants onto the task's canonical
label. An output is counted incorrect when no label can be recovered. ECG-QA uses the
released answer normalization. The headline metric is generative accuracy, followed by an
unweighted mean across the non-overlapping datasets.

```bash
# chain-of-thought downstream tasks
sbatch bashfiles/evaluate/eval_phase2.sh

# TimeMQA multiple-choice only
sbatch bashfiles/evaluate/eval_phase2_mcq_only.sh

# direct invocation
python evaluation/cot/eval_cot_phase2.py --help
```

Predictions are written to `<stage_dir>/results/test_predictions.jsonl` and metrics to
`<stage_dir>/results/metrics.json`. Read accuracy from the generative-EM metrics.

### Smoke test

A bounded GPU run that checks a Phase-2 stage completes real optimizer steps with a finite
loss, intended as a fast regression check rather than a research run:

```bash
sbatch bashfiles/smoke/smoke_phase2_gpu.sh     # prints SMOKE PASS / SMOKE FAIL
```

---

## Configuration

Settings shared by both phases first.

| | Phase 1 | Phase 2 |
|---|---|---|
| Patch size | `P = 32`, non-overlapping | ← |
| Encoder | Transformer decoder, width 64, FF 256, output 768 (95,680 params) | ← |
| Positional encoding | fixed 2-D sin-cos over (patch index, channel index); not learned | ← |
| Next-patch decoder | linear projection to encoder width, causally masked Transformer decoder, linear head to `P` outputs | removed |
| Optimiser | AdamW, weight decay `1e-2` | ← |
| LR (encoder / projector) | `1e-4` | ← |
| LR schedule | linear warmup over the first 3% of steps, then linear decay to 0 | ← |
| Gradient clipping | global norm `1.0` | ← |
| Precision | AMP fp16 with `GradScaler` | ← |
| Hardware | 8× NVIDIA V100 (2 nodes × 4), effective batch 128 | ← |
| Data | forecasting, QA, captioning | six CoT corpora |
| Mixture | weighted 1:2:1:1 (60% numeric / 40% linguistic) | balanced, 40,000 examples per task per epoch |
| Epoch size | task-weighted sampler | 5 × 40,000 = 200,000 |
| Objective | `L_CE + L_MSE` | `L_CE` only |
| LoRA | `r=8`, `α=16`, all attention and MLP projections | same, excluding decoder parameters |
| LR (LoRA) | `2e-4` | `2e-4` |
| Batch size | 16 | 1 |
| Gradient accumulation | 1 | 16 |
| Selection | validation loss | mean generative EM, 192 validation items per task |
| Early stopping | patience 5 | per task, patience 8 |

Each Phase-1 run trains for at most ten days under the same early-stopping protocol:
validation loss is evaluated at regular intervals and training stops when it does not
improve for five consecutive checks. The checkpoint with the lowest validation loss is
retained rather than the final one.

### Backbone configurations

Gemma3-270M uses the time-series encoder with LoRA adapters, training 3.17M parameters.
Gemma3-1B and Qwen3-0.6B use the transformer encoder with LoRA adapters, training 14.73M
and 12.65M parameters, of which 6.52M and 5.05M are adapter weights, and halve the
per-device batch to 8. Gemma3-1B uses twice the replicas of Qwen3-0.6B, restoring its
effective batch to 128. These totals include the next-patch decoder, which Phase 2
discards.

---

## Pretrained checkpoints

| Repository | Contents |
|---|---|
| [hiepnh137/LangTimeFoundationModel](https://huggingface.co/hiepnh137/LangTimeFoundationModel) | All Phase-1 runs, with optional LoRA adapters |
| [hiepnh137/LangTimeFoundationModel-nolora](https://huggingface.co/hiepnh137/LangTimeFoundationModel-nolora) | Best checkpoint — transformer encoder, frequency branch, mv-pos, no LoRA |

See `README_hiepnh.md` for the full Phase-1 run catalogue and the GIFT-Eval protocol.

---

## Repository layout

```
curriculum_learning.py          training entry point for every stage
src/opentslm/
  model/encoder/                patch encoders (transformer, MLP), 2-D positional encoding
  model/decoder/                next-patch decoders
  model/llm/                    model classes (OpenTSLMSPFixPos2D and variants)
  time_series_datasets/         Phase-1 / Phase-2 datasets, Chronos loaders, per-task corpora
bashfiles/
  _paths.sh                     shared path resolution; source it, do not execute it
  joint_phase1/                 Phase-1 launchers and cache builders
  joint_phase2/                 Phase-2 launchers
  evaluate/                     evaluation launchers
  smoke/                        bounded regression runs
evaluation/                     evaluation harnesses (CoT, forecasting, GIFT-Eval)
scripts/                        data prefetch, plotting, comparison utilities
```

### Key arguments

| Argument | Description |
|---|---|
| `--model` | Model class, e.g. `OpenTSLMSPFixPos2D`. |
| `--stages` | Stage(s) to run, e.g. `joint_phase1_extend_balanced_lora_r8`, `joint_phase2_downstream`. |
| `--llm_id` | HuggingFace id or local snapshot of the base LLM. |
| `--num_epochs` | Number of epochs. |
| `--patch_size` | Patch length (default 32). |
| `--pe_type` | Positional encoding; `sincos` for the 2-D sin-cos encoding. |
| `--enc_dec_type` | `transformer` or `mlp` encoder/decoder. |
| `--use_freq_embed` | Enable the spectral branch of the patch representation. |
| `--synthetic_data_dir` | Synthetic corpus directory, or several separated by `:`. |
| `--task_weight_*` | Phase-1 mixture weights. |
| `--warm_start_from` | Load weights before training, e.g. `pretrained`. |
| `--run_suffix` | Suffix for the results directory, for parallel A/B runs. |
| `--llm_mv_position_ids` / `--no_llm_mv_position_ids` | Share one RoPE position across channels at the same timestep, or use the LLM's stock sequential positions. |
| `--build_cache_only` | Construct the datasets to populate the caches, then exit. |
| `--eval_only` | Skip training and run evaluation. |
| `--max_samples_per_source` | Cap each source, for quick checks. |

Full list: `python curriculum_learning.py --help`.

---

## License and acknowledgement

Released under the MIT License. This repository builds on
[OpenTSLM](https://github.com/StanfordBDHG/OpenTSLM) (Stanford University, ETH Zurich and
the project authors); the original copyright notices are retained in the source headers,
`LICENSE.md` and `CONTRIBUTORS.md`.
