# Synthetic multivariate pretraining corpus: generation recipe

This directory holds the user-authored drivers that produced the 844k-series
synthetic corpus Phase 1 trains on (`FTS_SYNTHETIC_DIR`, read by
`src/opentslm/time_series_datasets/Phase1JointDataset.py`). The corpus is
built in two stages:

1. **Generate** raw series with four generator families, each a thin wrapper
   around an upstream open-source generator (cloned at a pinned commit).
2. **Unify** every raw output into `unified_<source>.arrow` files with one
   schema (`unify/collect_synthetic.py`).

Nothing here is upstream code. Each family section says which upstream repo
and commit to clone, where the files in this directory go inside that
checkout, which upstream files are patched (`upstream.patch`), and the exact
commands that were run. Sections marked **NOT RECOVERED** list what could not
be reconstructed from the surviving material; see the end of this file for a
consolidated list.

## Environment variables

| Variable | Meaning |
|---|---|
| `SYNTH_CHRONOS_REPO` | Checkout of `amazon-science/chronos-forecasting` (see §1, §2a). |
| `SYNTH_TIMEPFN_REPO` | Checkout of `egetaga/TimePFN` (see §2b, §3). |
| `SYNTH_SYMTIME_REPO` | Checkout of `wwhenxuan/SymTime` (see §4). |
| `FTS_SYNTHETIC_DIR` | Output root: where `unified_*.arrow` + `stats.{json,csv}` land, and what the trainer reads. |
| `SYNTH_ROOT` | Optional fallback parent for the three checkouts (`collect_synthetic.py` only). |
| `SYNTH_CONDA_ENV` | Conda env with the generator dependencies (default name `uni2st`, see below). |
| `SYNTH_SYMTIME_ENV` | SymTime's own Python 3.12 venv with `s2generator` (default `./.env` inside the SymTime checkout). |
| `SYNTH_UNIFY_DIR` | Directory holding `collect_synthetic.py` (only needed by `collect_synthetic.slurm`; defaults to the submit dir). |

Slurm headers keep the resource requests that were used; `--partition` /
`--qos` are site-specific and marked as such. `#SBATCH --output` paths are
relative (`slurm/logs/...` in the checkouts, `logs/` for the unify step), so
`mkdir -p slurm/logs` (or `logs`) before submitting from the directory the
comments name.

### Software versions that produced the corpus

Generator env (`uni2st`, Python 3.10.12): `gluonts 0.14.4`, `scikit-learn 1.7.2`,
`numpy 1.26.4`, `joblib 1.5.2`, `pyarrow 22.0.0`, `tqdm 4.67.1`, `scipy 1.11.4`,
`pandas 2.1.4`, `matplotlib 3.10.7`, `torch 2.9.1` (torch is only needed by the
unify step, to read SymTime `.pt` files).

SymTime venv (`./.env`, Python 3.12): `s2generator 0.0.3`, `numpy 2.3.5`,
`torch 2.11.0+cu126`, `sympy 1.14.0`, `scipy 1.17.1`, `pandas 2.3.3`, `tqdm 4.67.3`.
(SymTime's `requirements.txt` was un-pinned to install on Python 3.12; that
change is in `symtime/upstream.patch`.)

All generators are CPU-only (sklearn GP sampling dominates). A GPU does not help.

---

## 1. Chronos-2 multivariatizer (`chronos2_multivariatizer/`)

**Upstream**: `https://github.com/amazon-science/chronos-forecasting`, commit
`32111085d85de6cc17d0ea4dd08adba324f57590` (2026-04-21). No upstream file is
modified; everything is added. Amazon did not release the Chronos-2 §4.2
multivariatizer, so `multivariatize.py` is a re-implementation
(design choices and caveats are documented in `PIPELINE_NOTES.md`, which is
the README that accompanied the pipeline).

**Install into the checkout**

```bash
export SYNTH_CHRONOS_REPO=/path/to/chronos-forecasting
git clone https://github.com/amazon-science/chronos-forecasting "$SYNTH_CHRONOS_REPO"
git -C "$SYNTH_CHRONOS_REPO" checkout 32111085d85de6cc17d0ea4dd08adba324f57590
cp chronos2_multivariatizer/{multivariatize.py,run_multivariatize.sh,visualize_samples.py,run_visualize.sh} "$SYNTH_CHRONOS_REPO/scripts/"
cp chronos2_multivariatizer/*.slurm "$SYNTH_CHRONOS_REPO/slurm/"
mkdir -p "$SYNTH_CHRONOS_REPO/slurm/logs" "$SYNTH_CHRONOS_REPO/data"
```

`multivariatize.py` imports `KERNEL_BANK`, `random_binary_map` and
`sample_from_gp_prior` from the upstream `scripts/kernel-synth.py`, which is
why it has to live next to it.

**Commands that were run** (four jobs, 32 CPU / 64 GB each)

```bash
cd "$SYNTH_CHRONOS_REPO"
# (a) original single-config run: 10,000 items, L in [1024,4096], D in [3,10], K in [3,10],
#     MAX_LAG=32, COVARIATE_PROB=0, SEED=0  -> data/multivariate-kernelsynth.arrow   (8m50s)
sbatch slurm/multivariatize.slurm
# (b) diversity sweep: 3 array tasks x 100,000 items -> data/mv_{short,medium,long}.arrow  (~1h each)
sbatch slurm/multivariatize_sweep.slurm
```

Sweep settings (fixed inside `multivariatize_sweep.slurm`):

| task | file | length | variates D | sources K | max_kernels | MAX_LAG | seed |
|---|---|---|---|---|---|---|---|
| 1 short | `mv_short.arrow` | U[512, 1024] | U[3, 5] | U[2, 5] | 3 | 16 | 1 |
| 2 medium | `mv_medium.arrow` | U[1024, 3096] | U[4, 8] | U[3, 8] | 5 | 64 | 2 |
| 3 long | `mv_long.arrow` | U[4096, 8192] | U[5, 10] | U[4, 10] | 7 | 128 | 3 |

Per item one of four base generators (KernelSynth / AR(p) / TSI / ETS) is drawn
K times, standardised, and mixed by either a cotemporaneous (`f(A@Y)`, A ~
N(0,1/K), f in {id, tanh, sin, signed-sqrt}) or a sequential lead-lag
multivariatizer (p = 0.5 each). Output items are `{start, target}` with
`target` of shape `(D, L)`, written with GluonTS `ArrowWriter` (LZ4).

**Expected output**: 4 GluonTS arrow files, 10,000 + 3 x 100,000 = **310,000
multivariate series**, D in 3..10, L in 512..8192. "SVD did not converge"
lines in the logs are expected: the KernelSynth base retries on
`LinAlgError`.

`visualize.slurm` / `visualize_samples.py` only render sample PNGs
(`figures/<dataset>/`); they are not part of the data path.

---

## 2. KernelSynth, univariate (`kernelsynth/`)

The paper's "KernelSynth (univariate)" family is the union of two sweeps that
use the two upstream copies of KernelSynth. Both wrappers import the kernel
bank from the upstream `kernel-synth.py` next to them and only add a
configurable length and seed; upstream `kernel-synth.py` is unchanged in both
repos.

| flavor | upstream file | GP sampler | output dir |
|---|---|---|---|
| `chronos_flavor/` | `chronos-forecasting/scripts/kernel-synth.py` | `sample_from_gp_prior` (sklearn) | `$SYNTH_CHRONOS_REPO/data/ks_{short,medium,long}.arrow` |
| `timepfn_flavor/` | `TimePFN/synthetic_data_generation/kernel-synth.py` | `sample_from_gp_prior_efficient` (eigh) | `$SYNTH_TIMEPFN_REPO/datasets/synthetic/ks_{short,medium,long}.arrow` |

**Install**

```bash
cp kernelsynth/chronos_flavor/{kernel_synth_sweep.py,run_kernel_synth.sh} "$SYNTH_CHRONOS_REPO/scripts/"
cp kernelsynth/chronos_flavor/*.slurm                                    "$SYNTH_CHRONOS_REPO/slurm/"
cp kernelsynth/timepfn_flavor/kernel_synth_sweep.py "$SYNTH_TIMEPFN_REPO/synthetic_data_generation/"
cp kernelsynth/timepfn_flavor/gen_kernelsynth.sh    "$SYNTH_TIMEPFN_REPO/scripts/"
cp kernelsynth/timepfn_flavor/*.slurm               "$SYNTH_TIMEPFN_REPO/slurm/"
```

**Commands that were run** (32 CPU / 64 GB; 15-35 min per task)

```bash
cd "$SYNTH_CHRONOS_REPO" && sbatch slurm/kernel_synth_sweep.slurm      # job name ks_sweep, array 1-3
cd "$SYNTH_TIMEPFN_REPO"  && sbatch slurm/gen_kernelsynth_sweep.slurm  # job name tpfn_ks_sweep, array 1-3
```

Both sweeps use the same grid, `NUM_SERIES=50000` per task:

| task | length | max_kernels | seed |
|---|---|---|---|
| 1 short | 1024 | 3 | 1 |
| 2 medium | 2048 | 5 | 2 |
| 3 long | 4096 | 7 | 3 |

**Expected output**: 6 files x 50,000 univariate series (1-D `target` of
length L). After unification one row of `ks_medium.arrow` (chronos flavor)
is dropped for containing NaN, hence **299,999** series in the corpus
(149,999 + 150,000).

`kernel_synth.slurm` / `run_kernel_synth.sh` (chronos) and
`gen_kernelsynth.slurm` / `gen_kernelsynth.sh` (TimePFN) are the earlier
fixed-length (1024) single runs; their outputs
(`kernelsynth-data.arrow`, `ind-kernelsynth-largechannel.arrow`) are **not**
in the released corpus. They are kept because the sweep wrappers were
derived from them.

---

## 3. TimePFN LMC-Synth (`timepfn_lmc/`)

**Upstream**: `https://github.com/egetaga/TimePFN`, commit
`e11d5c2982b0e49b87873cbddcb7712641861268` (2025-07-29).

**Upstream patch** (`timepfn_lmc/upstream.patch`, modifies
`synthetic_data_generation/LMC_Synth.py`): `generate_time_series` takes a
per-call `length`; the CLI replaces `-C/--num-channels` and `-L/--length`
with `--min-channels/--max-channels/--min-length/--max-length/--seed`, and
each item draws its own (C, L) uniformly from those ranges via a per-item
`SeedSequence`-derived seed. Everything else (Weibull latent count,
Dirichlet channel mixing, kernel bank) is upstream.

**Install**

```bash
export SYNTH_TIMEPFN_REPO=/path/to/TimePFN
git clone https://github.com/egetaga/TimePFN "$SYNTH_TIMEPFN_REPO"
git -C "$SYNTH_TIMEPFN_REPO" checkout e11d5c2982b0e49b87873cbddcb7712641861268
git -C "$SYNTH_TIMEPFN_REPO" apply "$PWD/timepfn_lmc/upstream.patch"
mkdir -p "$SYNTH_TIMEPFN_REPO/scripts" "$SYNTH_TIMEPFN_REPO/slurm/logs" "$SYNTH_TIMEPFN_REPO/datasets/synthetic"
cp timepfn_lmc/gen_coreg_15k.sh "$SYNTH_TIMEPFN_REPO/scripts/"
cp timepfn_lmc/*.slurm          "$SYNTH_TIMEPFN_REPO/slurm/"
```

**Commands that were run** (32 CPU / 128 GB)

```bash
cd "$SYNTH_TIMEPFN_REPO"
# (a) base run: 15,000 items, C in [3,10], L in [1024,4096], max_kernels 5,
#     Dirichlet alpha in [0.1,2.0], Weibull (2.0,10.0), seed 0        (~2h)
#     -> datasets/synthetic/lmc_synth_C3-10_L1024-4096.arrow
sbatch slurm/gen_coreg_15k.slurm
# (b) diversity sweep: NUM_SERIES=200000 per array task             (short: 1h23m)
#     -> datasets/synthetic/lmc_{short,medium,long}.arrow
sbatch slurm/gen_lmc_sweep.slurm
```

Sweep settings (`gen_lmc_sweep.slurm`):

| task | file | length | channels | max_kernels | Dirichlet alpha | Weibull (shape, scale) | seed |
|---|---|---|---|---|---|---|---|
| 1 short | `lmc_short.arrow` | U[512, 1024] | U[3, 5] | 3 | [0.05, 0.5] | (2.0, 5.0) | 1 |
| 2 medium | `lmc_medium.arrow` | U[1024, 3096] | U[4, 8] | 5 | [0.1, 2.0] | (2.0, 10.0) | 2 |
| 3 long | `lmc_long.arrow` | U[4096, 8192] | U[5, 10] | 7 | [0.5, 5.0] | (2.0, 15.0) | 3 |

**What actually made it into the corpus**: only the base run (15,000) and
the *short* sweep task (200,000) = **215,000 series**, C in 3..10,
L in 512..2048. Sweep tasks 2 and 3 (`lmc_medium`, `lmc_long`) hit the 5-day
wall-clock limit and never wrote a file; `collect_synthetic.py` lists them
but skips missing sources. The paper's "TimePFN LMC, 512-2048" row reflects
exactly this. To reproduce the corpus you can submit `--array=1` only.

The base run's collected file spans L = 1024-2048 (mean 1534, see
`reference_stats.json`), although the recovered `gen_coreg_15k.sh` defaults
to `--max-length 4096` and the file name says `L1024-4096`. The job's stdout
does not echo its arguments, so the run was most likely submitted with
`MAX_LENGTH=2048` overridden at sbatch time (or with an earlier revision of
the script). To match the released file, run it with `MAX_LENGTH=2048`
(see NOT RECOVERED, item 7).

---

## 4. SymTime S2Generator (`symtime/`)

**Upstream**: `https://github.com/wwhenxuan/SymTime`, commit
`24a938649e140d92e8bf642d37a11fd9624e7698` (2025-11-13). The generator
itself is the `s2generator` PyPI package (0.0.3), driven by the upstream
`data_provider/s2generator_main.py`.

**Upstream patch** (`symtime/upstream.patch`):

* `data_provider/s2generator_main.py`: adds `--min_input_dim`,
  `--min_output_dim`, optional `--length` (if unset, sampled per (seed, ID,
  OD) in `[--min_length, --max_length]`), iterates ID/OD over the configured
  ranges instead of `1..max`, removes the 3 s per-seed cooldown, fixes the
  progress-bar total.
* `requirements.txt`: version pins removed (Python 3.12 venv).

**Install**

```bash
export SYNTH_SYMTIME_REPO=/path/to/SymTime
git clone https://github.com/wwhenxuan/SymTime "$SYNTH_SYMTIME_REPO"
git -C "$SYNTH_SYMTIME_REPO" checkout 24a938649e140d92e8bf642d37a11fd9624e7698
git -C "$SYNTH_SYMTIME_REPO" apply "$PWD/symtime/upstream.patch"
cd "$SYNTH_SYMTIME_REPO" && python3.12 -m venv .env && . .env/bin/activate && pip install -r requirements.txt   # provides s2generator 0.0.3
mkdir -p slurm/logs && cp "$OLDPWD"/symtime/{s2generate.sh,s2generate_sweep.slurm,s2generate_test.sh,submit_all.sh} slurm/
```

Each (seed, input_dim, output_dim) triple writes
`data/datasets/<cohort>/<seed>/ID=<i>_OD=<o>.pt` holding
`{"symbol", "excitation", "response"}`; `response` has shape
`(length, output_dim)` and is the multivariate target.

**Commands that were run** (32 CPU / 64 GB)

```bash
cd "$SYNTH_SYMTIME_REPO"
# (a) base cohort: 10 array tasks x 10 seeds, ID in [1,6], OD in [3,10], L=1024
#     -> data/datasets/pretrain_data/           (job s2gen, 16h50m)
sbatch slurm/s2generate.sh
# (b) three cohorts in one 24-task array (job s2gen_sweep, 5-day limit)
#     -> data/datasets/pretrain_data_{short,medium,long}/
sbatch slurm/s2generate_sweep.slurm
```

Sweep layout (`s2generate_sweep.slurm`):

| array ids | cohort | length | input_dim | output_dim | seeds per chunk |
|---|---|---|---|---|---|
| 0-3 | short | 1024 | [1, 4] | [3, 5] | 5000 |
| 4-11 | medium | 2048 | [2, 6] | [4, 8] | 1250 |
| 12-23 | long | 4096 | [3, 8] | [5, 10] | 600 |

S2Generator is slow (tens of seconds per file) and ~45 % of draws fail, so
the sweep was sized for ~100k samples per cohort but the short and medium
tasks (ids 0-11) hit the 5-day wall clock; only the long tasks completed.
The cohorts are therefore whatever had been written when the jobs stopped:

| cohort | `.pt` files on disk | dropped (NaN/inf) | kept |
|---|---|---|---|
| `pretrain_data` | 2,162 | 218 | 1,944 |
| `pretrain_data_short` | 14,558 | 905 | 13,653 |
| `pretrain_data_medium` | 3,420 | 330 | 3,090 |
| `pretrain_data_long` | 803 | 75 | 728 |

Any individual `(seed, ID, OD)` file is reproducible (the generator is
seeded by `seed`), but the *count* obtained inside a wall-clock budget is
not. See **NOT RECOVERED** below for the 139-series difference between the
numbers above and the paper.

---

## 5. Unify step (`unify/`)

`collect_synthetic.py` streams every source into
`$FTS_SYNTHETIC_DIR/unified_<source>.arrow` (pyarrow IPC *file* format,
32 rows per record batch) with the schema the trainer expects:

```
source      string      # e.g. "chronos_short"
sample_id   int32       # 0-based within the file
n_variates  int32       # C
length      int32       # L
target      list<list<float32>>   # [C][L]  (univariate sources are promoted to C=1)
```

`Phase1JointDataset._build_one_synthetic_part` reads `target` row by row as a
`[C, L]` nested list and transposes it to `[L, C]`. Readers for the two raw
formats: GluonTS `ArrowFile` (`kind="arrow"`), and the SymTime `.pt` tree
(`kind="symtime_pt"`, with a stub unpickler so `s2generator` need not be
installed in the unify env). Rows with any non-finite value, or outside the
`--min/max-variates` / `--min/max-length` envelope, are dropped and counted
in `stats.json` / `stats.csv` (written after every source).

**Run** (8 CPU / 32 GB, 1-2 h for the full corpus)

```bash
export SYNTH_CHRONOS_REPO=... SYNTH_TIMEPFN_REPO=... SYNTH_SYMTIME_REPO=... FTS_SYNTHETIC_DIR=...
cd unify && mkdir -p logs && sbatch collect_synthetic.slurm     # or: bash run_collect.sh
```

`run_collect.sh` passes the envelope that produced the released corpus
(`--min-variates 1 --max-variates 10000 --min-length 1 --max-length 100000000`,
recorded under `"envelope"` in `reference_stats.json`). The script's own
defaults (3..10 variates, 1024..4096 length) would drop the univariate
KernelSynth files and the 512- and 8192-length sweeps. `ONLY="name1 name2"`
re-collects a subset while preserving the other sources' stats;
`SOURCES_JSON` overrides the source list. The source list in
`collect_synthetic.py` includes `chronos_multivariate` (the original 10k
run) because it is part of the corpus; the recovered copy had it commented
out after that file was collected in an earlier pass.

`reference_stats.json` / `reference_stats.csv` are the `stats.{json,csv}`
produced by the collection run that the surviving arrow files came from
(paths genericised), useful for checking a re-run.

---

## 6. Expected totals

Verified by counting rows in the surviving `unified_*.arrow` files
(17 files, ~31 GB):

| file | rows | C | L |
|---|---|---|---|
| `unified_chronos_multivariate.arrow` | 10,000 | 3-10 | 1024-4096 |
| `unified_chronos_short.arrow` | 100,000 | 3-5 | 512-1024 |
| `unified_chronos_medium.arrow` | 100,000 | 4-8 | 1024-3096 |
| `unified_chronos_long.arrow` | 100,000 | 5-10 | 4096-8192 |
| `unified_kernel_synth_short.arrow` | 50,000 | 1 | 1024 |
| `unified_kernel_synth_medium.arrow` | 49,999 | 1 | 2048 |
| `unified_kernel_synth_long.arrow` | 50,000 | 1 | 4096 |
| `unified_timepfn_kernelsynth_short.arrow` | 50,000 | 1 | 1024 |
| `unified_timepfn_kernelsynth_medium.arrow` | 50,000 | 1 | 2048 |
| `unified_timepfn_kernelsynth_long.arrow` | 50,000 | 1 | 4096 |
| `unified_timepfn_lmc.arrow` | 15,000 | 3-10 | 1024-2048 |
| `unified_timepfn_lmc_short.arrow` | 200,000 | 3-5 | 512-1024 |
| `unified_symtime_s2generator.arrow` | 1,944 | 3-10 | 1024 |
| `unified_symtime_s2generator_short.arrow` | 13,653 | 3-5 | 1024 |
| `unified_symtime_s2generator_medium.arrow` | 3,090 (paper: 2,955) | 4-8 | 2048 |
| `unified_symtime_s2generator_long.arrow` | 728 (paper: 724) | 5-7 | 4096 |

| family | files | series | paper |
|---|---|---|---|
| Chronos-2 multivariatizer | `unified_chronos_*` | 310,000 | 310,000 |
| KernelSynth (univariate) | `unified_kernel_synth_*` + `unified_timepfn_kernelsynth_*` | 299,999 | 299,999 |
| TimePFN LMC | `unified_timepfn_lmc*` | 215,000 | 215,000 |
| SymTime S2Generator | `unified_symtime_*` | 19,415 | 19,276 |
| **Total** | 17 files | **844,414** | **844,275** |

The `--synthetic_data_dir` loader also applies a per-channel std floor and
value clip at window-sampling time (`norm_std_floor`, `norm_clip` in
`Phase1JointDataset`) because 15-25 % of SymTime rows have a near-constant
channel; that is a training-time safeguard, not a corpus filter.

---

## NOT RECOVERED

The working copy that produced the corpus was purged (only empty directory
skeletons and the `.git` of the three clones remained). Everything above
was reconstructed from a second copy of the project tree that was synced to
a shared group location on 2026-05-23, from Slurm accounting (job names,
resources, work dirs, run times), from the surviving per-job stdout logs, and
from the FoundationalTS code that consumes the corpus. Slurm's accounting
database did not store the batch scripts. The following could not be
recovered:

1. **The exact SymTime arrow files the paper counted (19,276).** The paper
   and `README.md` total 844,275, which equals the surviving files with
   `symtime_s2generator_medium = 2,955` and `symtime_s2generator_long = 724`
   (an earlier collection snapshot, noted in project notes dated 2026-06-29),
   instead of the 3,090 / 728 in the surviving files. The `.pt` trees were
   evidently re-collected after the sweep jobs wrote a few more files; the
   earlier pair of arrow files was overwritten. The raw `.pt` cohorts
   (2,162 / 14,558 / 3,420 / 803 files) match the surviving arrow files
   exactly, so the 139 extra series are additional valid samples, not a
   different generator setting. A re-run of the unify step on the raw data
   reproduces 844,414, not 844,275.
2. **The command line of the final collection run.** `run_collect.sh` as
   recovered did not pass envelope flags, yet `stats.json` records
   `min_variates=1, max_variates=10000, min_length=1, max_length=1e8`, so
   the flags were given by hand (or through an updated wrapper that was not
   synced). The copy here passes them explicitly. The surviving
   `collect_synthetic.out` shows the last run re-collected only three
   sources (`ONLY=...`), consistent with incremental re-collection of the
   SymTime cohorts.
3. **A July 2026 "xlong" regeneration** (Slurm jobs `mv_xlong_{short,medium,long,vlong}`,
   `lmc_xlong_*`, 128 CPU / 256 GB, 2026-07-06/07). Their raw outputs exist
   (`mv_xlong_*.arrow` ~100 GB, `lmc_xlong_{short,medium}.arrow` ~97 GB) but the
   driver scripts were never synced and no `unified_*xlong*.arrow` was ever
   produced; the family totals above are met without them, so they are not
   part of the paper corpus. Not needed for reproduction.
4. **A later revision of `collect_synthetic.py`** that added
   `n_covariates` / `feat_dynamic_real` columns (mentioned in the same
   2026-06-29 notes). The released corpus and the trainer use the 5-column
   schema produced by the version here, so this revision is not needed.
5. **Exact seeds-to-files bookkeeping for SymTime.** Because the short and
   medium sweep tasks were cut by the wall clock, the set of
   `(seed, ID, OD)` files present is the prefix each worker reached; the
   per-file list was not saved. Re-running `s2generate_sweep.slurm` with the
   same time limit on different hardware will yield a different count.
6. **A frozen `uni2st` environment file.** Only the installed package
   versions listed at the top were recoverable; there is no `environment.yml`.
7. **The sbatch-time overrides of the TimePFN base run.** The collected
   `lmc_synth_C3-10_L1024-4096.arrow` spans L = 1024-2048 while the recovered
   `gen_coreg_15k.sh` defaults to `MAX_LENGTH=4096`; the job's stdout does not
   echo its arguments (Slurm accounting shows the job ran three times on
   2026-05-12/14, the last one completing in 1h57m). Reproduce with
   `MAX_LENGTH=2048`.

Everything else (all generator wrappers, Slurm drivers, the two upstream
patches, the unify script, and the collection statistics) is in this
directory.
