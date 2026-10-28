---
license: mit
language:
  - en
tags:
  - time-series
  - time-series-language-model
  - forecast-grounding
  - lora
base_model:
  - google/gemma-3-270m
  - google/gemma-3-1b-pt
  - Qwen/Qwen3-0.6B
pipeline_tag: time-series-forecasting
---

# LOTUS Phase-1 (forecast-grounding) checkpoints

Phase-1 checkpoints for **LOTUS: Learning to Understand Time-series by Learning to
Forecast** ([arXiv:2610.18877](https://arxiv.org/abs/2610.18877)). Each checkpoint is a
`torch.save` dictionary containing the time-series encoder, projector, next-patch decoder
and (where applicable) LoRA adapter weights trained under forecast grounding. They are the
warm-start point for Phase-2 downstream adaptation; they do not include the base LLM
weights, which are loaded from the HuggingFace ids above.

Code and usage: <https://github.com/Davido111200/FoundationalTS>.

## Runs used in the paper

| Directory | Backbone | Encoder | Frequency branch | LoRA |
|---|---|---|---|---|
| `joint_phase1_extend_balanced_freq/` | Gemma3-270M | MLP 64/256 → 768 | yes | none (Phase 2 adds r=4) |
| `gemma_3_1b_pt_joint_phase1_extend_balanced_lora_r8_freq_mvpos/` | Gemma3-1B | Transformer 192 × 6 | yes | r=8, α=16 |
| `qwen3_0_6b_joint_phase1_extend_balanced_lora_r8_freq_mvpos/` | Qwen3-0.6B | Transformer 192 × 6 | yes | r=8, α=16 |

## Ablation runs

| Directory | Purpose |
|---|---|
| `gemma_3_270m_joint_phase1_extend_balanced_lora_r8_nofreq_mvpos/` | Transformer encoder, no frequency branch, LoRA r=8 |
| `gemma_3_270m_joint_phase1_tsqa_m4_langonly_lora_r8_freq_mvpos/` | Language-only Phase 1 (no forecasting objective) |
| `joint_phase1_extend_balanced_lora_nofreq/` | MLP encoder, no frequency branch, LoRA r=4 |
| `phase1_chronos_mse_*` | Forecasting-only Phase 1 variants |

Each directory holds `checkpoints/best_model.pt` (lowest validation loss) and
`checkpoints/loss_history.txt`.

## Usage

```bash
huggingface-cli download hiepnh137/LangTimeFoundationModel --local-dir "$FTS_PRETRAINED_DIR"
export WARM_START_PRETRAINED_CKPT="$FTS_PRETRAINED_DIR/joint_phase1_extend_balanced_freq/checkpoints/best_model.pt"
sbatch bashfiles/joint_phase2/joint5/train_joint5_ours.sh
```

See the repository README for the full Phase-2 recipe.

## Citation

```bibtex
@article{do2026lotus,
  title   = {Lotus: Learning to Understand Time-series by Learning to Forecast},
  author  = {Do, Van Dai and Nguyen, Huu Hiep and Nguyen, Minh Hoang and Nguyen, Manh and Le, Hung},
  journal = {arXiv preprint arXiv:2610.18877},
  year    = {2026}
}
```
