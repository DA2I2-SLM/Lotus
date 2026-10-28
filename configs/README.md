<!--
SPDX-FileCopyrightText: 2026 Deakin University and the LOTUS authors
SPDX-License-Identifier: MIT
-->

# configs/ — settings of the three released LOTUS runs

One file per backbone, collecting every hyper-parameter of the paper's released
runs (arXiv 2610.18877, Table 2 / Table 7) as plain shell variable assignments:

| File | Backbone | Encoder | Phase-1 launcher | Phase-2 launcher |
|---|---|---|---|---|
| `gemma3_270m.env` | Gemma3-270M | MLP 64/256 + frequency branch | `bashfiles/joint_phase1/mlp/train_mlp_joint_phase1_proc_synth_freq.sh` | `bashfiles/joint_phase2/joint5/train_joint5_ours.sh` |
| `gemma3_1b.env` | Gemma3-1B-pt | Transformer 192 x 6 + frequency branch | `bashfiles/joint_phase1/trans_enc/train_trans_joint_phase1_extbal_lora_r8_freq_mvpos_gemma3_1b.sh` | `bashfiles/joint_phase2/joint5/train_joint5_ours_1b.sh` |
| `qwen3_0_6b.env` | Qwen3-0.6B | Transformer 192 x 6 + frequency branch | `bashfiles/joint_phase1/trans_enc/train_trans_joint_phase1_extbal_lora_r8_freq_mvpos_qwen3_06b.sh` | `bashfiles/joint_phase2/joint5/train_joint5_ours_qwen3_06b.sh` |

These files are **documentation, not inputs**: the launchers do not source them.
Every value was read out of the launcher named above (and, for LoRA ranks, out of
`stages_with_lora` in `curriculum_learning.py`), so the launchers remain the
single source of truth. Where a launcher default differs from what the paper
reports, the line carries a `# paper Table 2 says ...` comment — currently only
the Phase-1 task weights of the two transformer-encoder runs (launcher default
`1:1:0.75:0.75`, paper `1:2:1:1`; export `TASK_WEIGHT_*` before `sbatch` to use
the paper's mixture, as the README's Phase-1 section shows).

Each file is valid bash (`bash -n configs/<file>.env` passes) so it can be
`source`d in a shell to inspect values, e.g.

```bash
source configs/gemma3_270m.env && echo "$PHASE2_BALANCE_TARGET $PHASE1_EFFECTIVE_BATCH"
```

Variables that reference `FTS_*` locations (`WARM_START_PRETRAINED_CKPT`,
`RESULTS_BASE_DIR`, `PHASE1_SYNTHETIC_DATA_DIR`) expand to the `bashfiles/_paths.sh`
convention; source `bashfiles/_paths.sh` first if you want them resolved.

Paper Table 4 ablations (270M) are launched with
`bashfiles/joint_phase2/joint5/train_ablation_270m.sh <arm>`; they share
`gemma3_270m.env`'s Phase-2 settings and differ only in warm start / frequency
branch (see that launcher's header).
