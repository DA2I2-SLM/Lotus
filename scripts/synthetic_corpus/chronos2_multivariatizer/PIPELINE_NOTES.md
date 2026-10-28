# Synthetic Time Series Generation (KernelSynth + Multivariatizer)

This document describes the synthetic data generation pipeline available in
this repository and the wrappers added under [scripts/](scripts/) and
[slurm/](slurm/) for batch generation.

It covers:

1. The **univariate** KernelSynth generator from the Chronos v1 paper
   (Ansari et al., 2024), which is the only generator open-sourced upstream.
2. A re-implementation of the **multivariate / covariate-informed synthetic
   data pipeline** described in the Chronos-2 technical report (Ansari
   et al., 2025, §4.2). Amazon did not release this pipeline; the code in
   `scripts/multivariatize.py` is a faithful re-implementation following the
   paper's design.

---

## 1. What's in the upstream code

### 1.1 Univariate KernelSynth — [scripts/kernel-synth.py](scripts/kernel-synth.py)

KernelSynth generates a univariate time series by:

1. Sampling 1..`max_kernels` base kernels (with replacement) from a fixed
   `KERNEL_BANK` of 33 sklearn GP kernels: many `ExpSineSquared` variants
   covering hourly/sub-hourly/daily/weekly/monthly/quarterly/yearly
   periodicities, plus `DotProduct` (linear trend), `RBF` (smooth),
   `RationalQuadratic`, `WhiteKernel` (noise), `ConstantKernel` (bias).
2. Combining the sampled kernels into one composite kernel by reducing them
   with a random binary operator picked uniformly from `{+, *}` at each
   step (`random_binary_map`).
3. Drawing a single sample from the resulting GP prior over a fixed time
   grid of length `LENGTH = 1024` via `GaussianProcessRegressor.sample_y`.

Each item is `{"start": np.datetime64(...), "target": ts}` with `ts` of
shape `(1024,)`. All items go into a single GluonTS-compatible Arrow file
via `gluonts.dataset.arrow.ArrowWriter` (LZ4-compressed). Paper defaults:
`num_series = 1,000,000`, `max_kernels = 5`. Per-series generation runs in
parallel via `joblib.Parallel(n_jobs=-1)`.

### 1.2 Chronos-2 multivariate generation (paper §4.2, not in source)

The Chronos-2 report describes that multivariate / covariate-informed
training relies entirely on synthetic data, produced by feeding samples
from **base univariate generators** (KernelSynth, AR, ETS, TSI, TCM) into a
**multivariatizer** that imposes multivariate structure:

- **Cotemporaneous multivariatizer**: linear or nonlinear transformation
  applied at the same time step across several univariate series ➔
  instantaneous correlations.
- **Sequential multivariatizer**: induces dependencies across time ➔
  lead-lag effects, cointegration.

The multivariatizer output is then used to build either **multivariate
tasks** (all variates predicted) or **covariate-informed tasks** (a random
subset of variates is designated as known covariates).

**The multivariatizer code is not released by Amazon.** This repo ships a
faithful re-implementation in [scripts/multivariatize.py](scripts/multivariatize.py)
covering: KernelSynth + AR + TSI + ETS base generators, cotemporaneous +
sequential lead-lag multivariatizers, and covariate-informed task
construction (see §3 below). Caveats are listed in §6.

---

## 2. Univariate generation: how to run

### 2.1 Layout

| File | Purpose |
|---|---|
| [scripts/kernel-synth.py](scripts/kernel-synth.py) | Upstream KernelSynth generator (unchanged) |
| [scripts/run_kernel_synth.sh](scripts/run_kernel_synth.sh) | Pure-bash wrapper with env-var configuration |
| [slurm/kernel_synth.slurm](slurm/kernel_synth.slurm) | Slurm submission script |

The wrapper exists because `kernel-synth.py` hard-codes the output path to
`<script_dir>/kernelsynth-data.arrow`. The wrapper runs it in a temporary
working directory and moves the resulting Arrow file to `${OUT}`.

### 2.2 Local run (no slurm)

The generator needs `gluonts`, `joblib`, `scikit-learn`, `tqdm`, `numpy`.
On this cluster the `uni2st` conda env has all of these.

```bash
conda activate uni2st

# Defaults: NUM_SERIES=10000, MAX_KERNELS=5,
#           OUT=${SYNTH_CHRONOS_REPO}/data/kernelsynth-data.arrow
bash ${SYNTH_CHRONOS_REPO}/scripts/run_kernel_synth.sh

# Paper-scale run
NUM_SERIES=1000000 MAX_KERNELS=5 \
OUT=${SYNTH_CHRONOS_REPO}/data/kernelsynth-1M.arrow \
    bash ${SYNTH_CHRONOS_REPO}/scripts/run_kernel_synth.sh
```

### 2.3 Slurm

```bash
# Defaults (10k series)
sbatch ${SYNTH_CHRONOS_REPO}/slurm/kernel_synth.slurm

# Paper defaults (1M series)
sbatch --export=ALL,NUM_SERIES=1000000,MAX_KERNELS=5 \
       ${SYNTH_CHRONOS_REPO}/slurm/kernel_synth.slurm

# Custom output path
sbatch --export=ALL,NUM_SERIES=500000,OUT=/path/to/my.arrow \
       ${SYNTH_CHRONOS_REPO}/slurm/kernel_synth.slurm
```

Resource request: `--partition=cpu`, `--cpus-per-task=32`, `--mem=64gb`,
`--time 1-00:00:00`, `--qos=batch-short`. KernelSynth is CPU-bound (sklearn
GP); a GPU does not help. `joblib` saturates all allocated cores.

Logs land in [slurm/logs/](slurm/logs/).

### 2.4 Output schema

```python
from gluonts.dataset.arrow import ArrowFile
af = ArrowFile("/path/to/kernelsynth-data.arrow")
for item in af:
    print(item["start"], item["target"].shape)  # ("2000-01-01T00:00:00", (1024,))
    break
```

Standard GluonTS univariate file dataset, directly consumable by the
Chronos training configs in [scripts/training/configs/](scripts/training/configs/).

---

## 3. Multivariate + covariate-informed generation

### 3.1 Layout

| File | Purpose |
|---|---|
| [scripts/multivariatize.py](scripts/multivariatize.py) | Base generators (KernelSynth, AR, TSI, ETS) + multivariatizers (cotemporaneous, sequential) + task construction |
| [scripts/run_multivariatize.sh](scripts/run_multivariatize.sh) | Pure-bash wrapper |
| [slurm/multivariatize.slurm](slurm/multivariatize.slurm) | Slurm submission script |

### 3.2 Base univariate generators

Per source draw, one of the four is chosen uniformly at random:

| Name | Description |
|---|---|
| `kernelsynth` | Upstream Chronos KernelSynth (random GP-prior composition) |
| `ar` | AR(p) with `p ∈ {1..5}`. Stationarity enforced by drawing partial autocorrelations in `(-0.95, 0.95)` and converting to AR coefficients via the Levinson recursion |
| `tsi` | Trend-Seasonality-Irregularity (Bahrpeyma et al., 2021): random polynomial trend + 1..3 sinusoidal seasonalities + Gaussian noise |
| `ets` | Additive Holt-Winters state-space simulator with random level/trend/seasonal smoothing constants and seasonal period |

Each base series is **standardized** (zero mean, unit std) before mixing.

### 3.3 Multivariatizers

Per item, one class is chosen with probability 0.5:

- **Cotemporaneous**: `Z = f(A @ Y)`, where `A ∈ R^{D×K}` has iid
  `N(0, 1/K)` entries. `f` is either identity (purely linear, 50%) or one
  of `{tanh, sin, signed-sqrt}` (purely nonlinear, 50%). Each call is
  *either* linear *or* nonlinear, matching the paper's "linear OR nonlinear"
  wording.
- **Sequential lead-lag**: `Z[d, t] = Σ_k a_{d,k} · Y_k[t − τ_{d,k}]` with
  `τ_{d,k} ~ U{0..MAX_LAG}`. Per-`(d, k)` lags introduce lead-lag effects
  between output variates. The transient `t < τ` is zero-padded.

Both produce `Z ∈ R^{D × L}`. **Cointegration** arises naturally when the
cotemporaneous branch mixes non-stationary bases (e.g. ETS with random
walks in the level state) — two outputs sharing the same `I(1)` bases are
cointegrated by construction.

### 3.4 Task construction (covariate-informed)

After multivariatization, with probability `COVARIATE_PROB` (default 0.3) a
random subset of variates is split off as **known covariates** (size in
`[1, floor(D · MAX_COVARIATE_FRAC)]`). Items are routed into one of two
homogeneous-schema files:

- **`OUT`** (multivariate task): `{start, target}` with `target` shape `(D, L)`
- **`OUT.cov.arrow`** (covariate-informed task): `{start, target, feat_dynamic_real}`
  with `target` shape `(D_target, L)` and `feat_dynamic_real` shape `(D_cov, L)`

Two files are written because GluonTS's `ArrowWriter` takes its schema from
item 0 and either silently drops extra columns on later items or raises
`KeyError`, so heterogeneous schemas are broken in either direction. The
split exactly matches the paper's task taxonomy.

### 3.5 Reproducibility

Each item derives its RNG state from `SeedSequence(item_seed)`, and
`gen_kernelsynth` seeds NumPy's legacy global state (which the upstream
`kernel-synth.py` uses) from that per-item stream. Two runs with the same
`SEED` produce byte-identical output:

```text
NUM_SERIES=100 SEED=42  -> 71 multi + 29 cov, reproducible=True (run A vs run B)
```

### 3.6 Local run

```bash
conda activate uni2st

# Defaults:
#   NUM_SERIES=10000, MAX_KERNELS=5, LENGTH=1024,
#   MIN_VARIATES=2, MAX_VARIATES=8, MIN_SOURCES=2, MAX_SOURCES=8,
#   MAX_LAG=32, COVARIATE_PROB=0.3, MAX_COVARIATE_FRAC=0.5, SEED=0
#   OUT=${SYNTH_CHRONOS_REPO}/data/multivariate-kernelsynth.arrow
bash ${SYNTH_CHRONOS_REPO}/scripts/run_multivariatize.sh

# More variates, more sources, no covariate split
NUM_SERIES=100000 MAX_VARIATES=16 MAX_SOURCES=16 COVARIATE_PROB=0.0 \
OUT=${SYNTH_CHRONOS_REPO}/data/mv-100k.arrow \
    bash ${SYNTH_CHRONOS_REPO}/scripts/run_multivariatize.sh

# Heavy covariate split
NUM_SERIES=100000 COVARIATE_PROB=0.7 MAX_COVARIATE_FRAC=0.75 \
OUT=${SYNTH_CHRONOS_REPO}/data/mv-100k.arrow \
    bash ${SYNTH_CHRONOS_REPO}/scripts/run_multivariatize.sh
# -> writes mv-100k.arrow + mv-100k.cov.arrow
```

### 3.7 Slurm

```bash
# Defaults (10k series)
sbatch ${SYNTH_CHRONOS_REPO}/slurm/multivariatize.slurm

# Scale up
sbatch --export=ALL,NUM_SERIES=200000,MIN_VARIATES=2,MAX_VARIATES=16 \
       ${SYNTH_CHRONOS_REPO}/slurm/multivariatize.slurm

# Custom output + heavy covariates
sbatch --export=ALL,NUM_SERIES=50000,COVARIATE_PROB=0.5,OUT=/path/to/my-mv.arrow \
       ${SYNTH_CHRONOS_REPO}/slurm/multivariatize.slurm
```

Resource request mirrors the univariate job (cpu partition, 32 cpus, 64 GB,
1 day, qos=batch-short). Logs in [slurm/logs/](slurm/logs/).

Measured throughput on this cluster: ~7 s wall for 50 items at 32 cores
(includes the sklearn GP draws for KernelSynth, which dominate).

### 3.8 Reading the output

```python
import numpy as np
from gluonts.dataset.arrow import ArrowFile

# Multivariate task items
for item in ArrowFile("/path/to/multivariate-kernelsynth.arrow"):
    t = np.asarray(item["target"])  # shape (D, 1024)
    break

# Covariate-informed task items
for item in ArrowFile("/path/to/multivariate-kernelsynth.cov.arrow"):
    target = np.asarray(item["target"])               # shape (D_target, 1024)
    covs   = np.asarray(item["feat_dynamic_real"])    # shape (D_cov,    1024)
    break
```

---

## 4. Configurable parameters (summary)

All scripts are configured via env vars; slurm forwards them with
`sbatch --export=ALL,VAR=value,...`.

### Univariate ([run_kernel_synth.sh](scripts/run_kernel_synth.sh))

| Variable | Default | Meaning |
|---|---|---|
| `NUM_SERIES` | `10000` | Total number of univariate series to generate |
| `MAX_KERNELS` | `5` | Maximum number of base kernels combined per series |
| `OUT` | `<repo>/data/kernelsynth-data.arrow` | Output Arrow file path |

### Multivariate ([run_multivariatize.sh](scripts/run_multivariatize.sh))

| Variable | Default | Meaning |
|---|---|---|
| `NUM_SERIES` | `10000` | Total number of multivariate series to generate |
| `MAX_KERNELS` | `5` | Forwarded to KernelSynth when it's the chosen base |
| `LENGTH` | `1024` | Time-series length |
| `MIN_VARIATES` / `MAX_VARIATES` | `2` / `8` | Range for the number of output variates `D` per series |
| `MIN_SOURCES` / `MAX_SOURCES` | `2` / `8` | Range for the number of base univariate sources `K` per series |
| `MAX_LAG` | `32` | Upper bound on `τ` in the sequential multivariatizer |
| `COVARIATE_PROB` | `0.3` | Probability an item is converted to a covariate-informed task |
| `MAX_COVARIATE_FRAC` | `0.5` | Upper bound on the fraction of variates designated as known covariates |
| `SEED` | `0` | RNG seed; runs with the same seed are byte-identical |
| `OUT` | `<repo>/data/multivariate-kernelsynth.arrow` | Multivariate output path; covariate-informed items go to `<stem>.cov<suffix>` |

---

## 5. Environment

Scripts assume the `uni2st` conda env (`gluonts`, `joblib`, `scikit-learn`,
`tqdm`, `numpy`). The slurm scripts activate it via:

```bash
module load Anaconda3
source activate
conda activate uni2st
```

The default `time-series` env on this cluster does **not** ship `gluonts`,
which is why `uni2st` is used. Any other env with the dependencies above
also works.

---

## 6. Caveats and references

- The **univariate** generator is the upstream Amazon code, unchanged.
  Matches Chronos v1 paper §3.1.
- The **multivariate / covariate-informed** generator is a re-implementation
  of Chronos-2 §4.2. It is **not** Amazon's official code, which has not
  been open-sourced. The paper's description of multivariatizers is brief
  (no equations, no distributions, no ranges); the specific choices in
  this repo — mixing-matrix distribution (`N(0, 1/K)`), nonlinearity set
  (`{tanh, sin, signed-sqrt}`), regime split (purely linear vs purely
  nonlinear, 50/50), lag distribution (`U{0..MAX_LAG}`), `D` and `K`
  ranges — are reasonable but uncited design choices.
- The **TCM (Temporal Causal Model)** base generator mentioned in the
  paper is **not** implemented here — it would require sampling random
  causal graphs from Runge et al. 2023 and an autoregression step over
  them. Four bases (KernelSynth, AR, TSI, ETS) are implemented.
- The sequential multivariatizer here covers **lead-lag** explicitly. The
  paper also lists **cointegration** as a property; this arises naturally
  when the cotemporaneous branch mixes non-stationary bases (e.g. ETS with
  a random-walk level), so there is no separate cointegration generator.
- Papers:
  - Ansari et al., 2024. *Chronos: Learning the Language of Time Series.*
    See [chronos.pdf](chronos.pdf).
  - Ansari et al., 2025. *Chronos-2: From Univariate to Universal
    Forecasting.* See [chronos-2.pdf](chronos-2.pdf).
  - Bahrpeyma et al., 2021. *A methodology for validating diversity in
    synthetic time series generation* (TSI generator reference).
