<!--
SPDX-FileCopyrightText: 2026 Deakin University and the LOTUS authors

SPDX-License-Identifier: MIT
-->

# Baselines

Every row of the paper's main table (Table 1) and what in this repository produces it.
All systems are evaluated under greedy decoding with the parser described in the README's
Evaluation section. The backbone-matched rows share the Phase-2 data, mixture, balanced
sampling (40,000 examples per task per epoch) and selection metric (mean generative exact
match over 192 validation items per task, patience 8), so the only difference between them
is the time-series interface.

## Backbone-matched baselines (in this repository)

| Row | Interface | Train | Evaluate |
|---|---|---|---|
| Text-tokenized, FT | series written into the prompt as comma-separated decimals (`baselines/tokenized_llm/tokenized_sft.py --serializer ours`), LoRA `r=8, α=16` on the backbone | `LORA_R=8 sbatch bashfiles/joint_phase2/joint5/train_joint5_tokenized_gemma.sh` (270M), `LORA_R=8 sbatch train_joint5_tokenized_gemma_1b.sh` (1B), `LORA_R=8 sbatch train_joint5_tokenized.sh full qwen3` (Qwen3-0.6B) | `tokenized_sft.py --mode eval --ckpt <dir>/best --split test --out_dir <d>` |
| Text-tokenized, zero-shot | same serialisation, base model, no training | — | `tokenized_sft.py --mode eval --model_id <base LLM> --split test --out_dir <d>` |
| LLMTime-style serialisation | digits with LLMTime's encoding (`--serializer llmtime`), selected by validation loss | `train_joint5_llmtime.sh` | the launcher runs the evaluation after training |
| OpenTSLM-SP | OpenTSLM's soft-prompt encoder (`--model OpenTSLMSP`) | `train_joint5_official.sh` (270M), `train_joint5_official_1b.sh` (1B) | `bashfiles/evaluate/eval_baselines.sh sp <ckpt> <out> <tasks>` |
| OpenTSLM-Flamingo | OpenTSLM's Flamingo cross-attention interface (`--model OpenTSLMFlamingo`; `pip install -e '.[baselines]'`) | same launchers | `eval_baselines.sh fl <ckpt> <out> <tasks>` |
| LOTUS | this paper (`--model OpenTSLMSPFixPos2D --use_freq_embed`) | `train_joint5_ours{,_1b,_qwen3_06b}.sh` | `eval_baselines.sh joint …` or `bashfiles/evaluate/eval_phase2_cot.sh` |

Without `LORA_R` the tokenized launchers fully fine-tune the backbone at learning rate
`1e-5`; that arm is a deliberately generous control and is not a row in the paper. With
`LORA_R=8` they attach the same adapter (rank, alpha, target modules) the LOTUS runs use,
at the LoRA learning rate `2e-4`, which is the configuration behind the paper's
"Text-tokenized, FT" rows (1.9M / 6.5M / 5.0M trainable parameters).

The OpenTSLM-SP and OpenTSLM-Flamingo launchers call two small wrappers,
`train_official_sp.py` and `train_official_flamingo.py`, through
`${FTS_OFFICIAL_VARIANTS:-$FTS_REPO_ROOT/baselines/official_variants}`. Those wrappers
adapt OpenTSLM's classes to the joint Phase-2 protocol and are not yet in this repository;
the launcher fails with a clear message until they are added.

## Sequential† rows

The rows marked † follow OpenTSLM's original sequential fine-tuning recipe and dataset
order (HAR → Sleep-EDF → ECG-QA), with PAMAP2 and TimeMQA appended, each task trained to
convergence before the next starts. The first three stages are OpenTSLM's own curriculum
stages, still present in this repository (`stage3_cot`, `stage4_sleep_cot`,
`stage5_ecg_cot` in `curriculum_learning.py --stages`). The PAMAP2 and TimeMQA
continuations are single-task `joint_phase2_downstream` stages warm-started from the
previous stage's checkpoint (`PHASE2_COT_TASKS=pamap2`, then `timemqa`, with
`WARM_START_PRETRAINED_CKPT` pointing at the previous `best_model.pt`). There is no single
launcher for the chain.

## Reference systems (not in this repository)

The unmatched reference block of Table 1 (Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct and
Mistral-7B-Instruct-v0.3 zero-shot; Gemma3-4B-pt with the series rendered as an image,
zero-shot and fine-tuned; TS-Reasoner-7B zero-shot and fine-tuned) was produced with the
respective projects' own code and is reported for context. Those runs are not reproducible
from this repository. The only related launcher here is
`bashfiles/joint_phase2/joint5/train_joint5_timeomni_4b.sh`, which adapts TimeOmni-1 to the
Phase-2 mixture and needs `FTS_TIMEOMNI_PYTHON` pointing at an environment with its
dependencies.

## TSRBench

The zero-shot TSRBench comparison (paper Figure 2 and Table 6) uses the benchmark's own
prompts and released scorer. The LOTUS runner is `evaluation/tsrbench/run_tsrbench.py`
(see `evaluation/tsrbench/README.md`). The other systems' TSRBench numbers come from the
benchmark's own evaluation of those models.
