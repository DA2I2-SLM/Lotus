---
license: mit
language:
  - en
tags:
  - time-series
  - time-series-language-model
  - forecast-grounding
base_model:
  - google/gemma-3-270m
pipeline_tag: time-series-forecasting
---

# LOTUS Phase-1 forecasting-only checkpoint (no LoRA)

A forecasting-only Phase-1 checkpoint from **LOTUS: Learning to Understand Time-series by Learning to
Forecast** ([arXiv:2610.18877](https://arxiv.org/abs/2610.18877)): Gemma3-270M backbone, transformer patch
encoder (width 192, 6 layers) with the frequency branch and shared multivariate RoPE positions, trained
with the next-patch MSE objective only (stage `phase1_chronos_mse`) and **no LoRA** on the LLM. The
`torch.save` dictionary holds the encoder, projector and next-patch decoder; the base LLM is loaded from
the HuggingFace id above.

Code and usage: <https://github.com/DA2I2-SLM/Lotus>.

## Contents

`checkpoints/best_model.pt` (lowest validation loss) and `checkpoints/loss_history.txt`. This is the
checkpoint behind the GIFT-Eval zero-shot forecasting comparison in `docs/phase1_run_catalogue.md`; it
was trained with `bashfiles/joint_phase1/trans_enc/train_trans_phase1_chronos_mse_freq_1p5x_mvpos.sh`.
The Phase-1 checkpoints behind the paper's downstream results are in
[hiepnh137/LangTimeFoundationModel](https://huggingface.co/hiepnh137/LangTimeFoundationModel).

## Usage

```bash
huggingface-cli download hiepnh137/LangTimeFoundationModel-nolora --local-dir "$LOTUS_PRETRAINED_DIR/phase1_chronos_mse_freq_mvpos"
CKPT="$LOTUS_PRETRAINED_DIR/phase1_chronos_mse_freq_mvpos/checkpoints/best_model.pt" \
  sbatch bashfiles/evaluate/joint_phase1/eval_gift_eval.sh
```

See `docs/phase1_run_catalogue.md` in the repository for the forecasting evaluation protocol.

## Citation

```bibtex
@article{do2026lotus,
  title   = {Lotus: Learning to Understand Time-series by Learning to Forecast},
  author  = {Do, Van Dai and Nguyen, Huu Hiep and Nguyen, Minh Hoang and Nguyen, Manh and Le, Hung},
  journal = {arXiv preprint arXiv:2610.18877},
  year    = {2026}
}
```
