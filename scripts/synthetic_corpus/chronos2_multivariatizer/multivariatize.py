# Re-implementation of the synthetic multivariate / covariate-informed
# data pipeline described in Chronos-2 (Ansari et al., 2025) §4.2.
#
# The paper describes the design but does not release the code. This module
# implements the full pipeline at the level of detail given in the paper:
#
#   - Base univariate generators: KernelSynth (upstream, reused),
#     AR, TSI (Bahrpeyma et al., 2021), ETS. (TCM omitted; see Caveats below.)
#   - Multivariatizers, applied per item with random class:
#       * Cotemporaneous: Z = f(A @ Y), with f either identity (linear)
#         or one of {tanh, sin, signed-sqrt} (nonlinear). Each item is
#         purely one or the other, matching "linear OR nonlinear" wording.
#       * Sequential lead-lag: Z[d, t] = sum_k a_{d,k} * Y_k[t - tau_{d,k}].
#   - Covariate-informed task construction: with probability `covariate_prob`,
#     a random subset of variates is split off as known covariates (stored
#     as `feat_dynamic_real` in the Arrow file). The remaining channels are
#     the target.
#   - Per-item reproducibility: each worker derives independent RNG streams
#     for its own choices AND seeds NumPy's legacy global state for the
#     upstream KernelSynth call.
#
# Output: TWO GluonTS-compatible Arrow files with homogeneous schemas:
#   - <OUT>          (multivariate-only items): {start, target},
#                     target shape (D_target, L), float32.
#   - <OUT>.cov.arrow (covariate-informed items): {start, target, feat_dynamic_real},
#                     target shape (D_target, L), feat_dynamic_real shape (D_cov, L).
# Splitting into two files is required because GluonTS's ArrowWriter takes its
# schema from the first item and either silently drops extra fields on later
# items (if item 0 has fewer columns) or raises KeyError (if item 0 has more).

import argparse
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from gluonts.dataset.arrow import ArrowWriter
from joblib import Parallel, delayed
from tqdm.auto import tqdm

# Reuse the official KernelSynth generator.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
_ks = import_module("kernel-synth")
_KS_generate_time_series = _ks.generate_time_series
LENGTH = _ks.LENGTH


# ============================ base univariate generators ============================

def gen_kernelsynth(rng: np.random.Generator, length: int, max_kernels: int) -> np.ndarray:
    """KernelSynth wrapper. Seed NumPy's legacy global state from `rng` so the
    upstream `generate_time_series` (which uses `np.random.*`) is reproducible
    per call.
    """
    np.random.seed(int(rng.integers(0, 2**31 - 1)))
    out = _KS_generate_time_series(max_kernels=max_kernels)["target"].astype(np.float64)
    if out.shape[0] >= length:
        return out[:length]
    return np.pad(out, (0, length - out.shape[0]))


def gen_ar(rng: np.random.Generator, length: int) -> np.ndarray:
    """AR(p) with stable random coefficients. Stationarity is enforced by
    drawing partial autocorrelations in (-0.95, 0.95) and converting to AR
    coefficients via the Levinson recursion.
    """
    p = int(rng.integers(1, 6))
    pacs = rng.uniform(-0.95, 0.95, size=p)
    phi = np.zeros(p)
    for k in range(p):
        new = np.zeros(k + 1)
        new[k] = pacs[k]
        for j in range(k):
            new[j] = phi[j] - pacs[k] * phi[k - 1 - j]
        phi[: k + 1] = new
    sigma = float(rng.uniform(0.5, 2.0))
    burn = 100
    n = length + burn
    eps = rng.standard_normal(n) * sigma
    y = np.zeros(n)
    for t in range(p, n):
        y[t] = phi @ y[t - p : t][::-1] + eps[t]
    return y[burn:].astype(np.float64)


def gen_tsi(rng: np.random.Generator, length: int) -> np.ndarray:
    """Trend-Seasonality-Irregularity (Bahrpeyma et al., 2021).
    Random polynomial trend + 1..3 sinusoidal seasonalities + Gaussian noise.
    """
    t = np.arange(length, dtype=np.float64) / length

    deg = int(rng.integers(0, 4))
    coefs = rng.normal(0.0, 1.0, size=deg + 1)
    trend = sum(c * t**k for k, c in enumerate(coefs))

    n_season = int(rng.integers(1, 4))
    seasonality = np.zeros(length)
    for _ in range(n_season):
        period = float(rng.uniform(0.02, 0.5))
        amp = float(rng.uniform(0.3, 2.0))
        phase = float(rng.uniform(0.0, 2 * np.pi))
        seasonality += amp * np.sin(2 * np.pi * t / period + phase)

    sigma = float(rng.uniform(0.05, 0.5))
    noise = rng.standard_normal(length) * sigma
    return (trend + seasonality + noise).astype(np.float64)


def gen_ets(rng: np.random.Generator, length: int) -> np.ndarray:
    """Additive Holt-Winters (ETS) state-space simulator.
    y_t = l_{t-1} + b_{t-1} + s_{t-m} + eps_t, with random level/trend/seasonal
    smoothing constants and seasonal period m.
    """
    alpha = float(rng.uniform(0.05, 0.5))
    beta = float(rng.uniform(0.001, 0.05))  # small beta to keep trend bounded
    gamma = float(rng.uniform(0.05, 0.5))
    m = int(rng.choice([12, 24, 48, 96, 168]))
    sigma = float(rng.uniform(0.1, 1.0))

    burn = max(2 * m, 100)
    n = length + burn
    eps = rng.standard_normal(n) * sigma
    l = float(rng.normal(0.0, 1.0))
    b = float(rng.normal(0.0, 0.05))
    s = list(rng.normal(0.0, 0.5, size=m))
    y = np.zeros(n)
    for t in range(n):
        s_prev = s[t % m]
        y[t] = l + b + s_prev + eps[t]
        l_new = l + b + alpha * eps[t]
        b_new = b + beta * eps[t]
        s[t % m] = s_prev + gamma * eps[t]
        l, b = l_new, b_new
    return y[burn:].astype(np.float64)


BASE_GENERATORS: dict[str, Callable] = {
    "kernelsynth": gen_kernelsynth,
    "ar": gen_ar,
    "tsi": gen_tsi,
    "ets": gen_ets,
}
BASE_NAMES = list(BASE_GENERATORS.keys())


def _draw_base(rng: np.random.Generator, length: int, max_kernels: int) -> np.ndarray:
    name = BASE_NAMES[int(rng.integers(0, len(BASE_NAMES)))]
    if name == "kernelsynth":
        return gen_kernelsynth(rng, length=length, max_kernels=max_kernels)
    return BASE_GENERATORS[name](rng, length=length)


def _standardize(y: np.ndarray) -> np.ndarray:
    mu = y.mean()
    sd = y.std()
    if sd < 1e-8:
        return y - mu
    return (y - mu) / sd


# ============================ multivariatizers ============================

def cotemporaneous_multivariatize(
    Y: np.ndarray, rng: np.random.Generator, D: int
) -> np.ndarray:
    """Z = f(A @ Y). f is identity (purely linear, 50%) or one of
    {tanh, sin, signed-sqrt} (purely nonlinear, 50%). Each call picks one
    regime for all D output channels, matching "linear OR nonlinear" wording.
    """
    K, L = Y.shape
    A = rng.standard_normal((D, K)) / np.sqrt(K)
    Z = A @ Y
    if rng.random() < 0.5:
        nls = [np.tanh, np.sin, lambda x: np.sign(x) * np.sqrt(np.abs(x))]
        f = nls[int(rng.integers(0, len(nls)))]
        Z = f(Z)
    return Z


def sequential_multivariatize(
    Y: np.ndarray, rng: np.random.Generator, D: int, max_lag: int = 32
) -> np.ndarray:
    """Sequential lead-lag mixing:
        Z[d, t] = sum_k a_{d,k} * Y_k[t - tau_{d,k}],   tau_{d,k} ~ U{0,..,max_lag}.

    With per-(d,k) lags, output variates lead/lag each other through their
    shared bases. Leading edge t < tau is zero-padded (transient).
    """
    K, L = Y.shape
    A = rng.standard_normal((D, K)) / np.sqrt(K)
    lags = rng.integers(0, max_lag + 1, size=(D, K))
    Z = np.zeros((D, L), dtype=np.float64)
    for d in range(D):
        for k in range(K):
            tau = int(lags[d, k])
            if tau == 0:
                Z[d] += A[d, k] * Y[k]
            else:
                Z[d, tau:] += A[d, k] * Y[k, :-tau]
    return Z


# ============================ task construction ============================

def generate_multivariate_item(
    seed: int,
    length: int,
    max_kernels: int,
    min_variates: int,
    max_variates: int,
    min_sources: int,
    max_sources: int,
    covariate_prob: float,
    max_covariate_frac: float,
    max_lag: int,
    min_length: int | None = None,
    max_length: int | None = None,
):
    rng = np.random.default_rng(np.random.SeedSequence(seed))

    D = int(rng.integers(min_variates, max_variates + 1))
    K = int(rng.integers(min_sources, max_sources + 1))

    # Per-item length sampling (uniform integer over [min_length, max_length]).
    if min_length is not None and max_length is not None:
        length = int(rng.integers(min_length, max_length + 1))

    # Sample K base univariate sources from heterogeneous generators.
    bases = [
        _standardize(_draw_base(rng, length=length, max_kernels=max_kernels))
        for _ in range(K)
    ]
    Y = np.stack(bases, axis=0)  # (K, L)

    # Pick a multivariatizer class with equal probability.
    if rng.random() < 0.5:
        Z = cotemporaneous_multivariatize(Y, rng, D=D)
    else:
        Z = sequential_multivariatize(Y, rng, D=D, max_lag=max_lag)

    # Optionally designate a random subset of variates as known covariates.
    feat_dynamic_real: Optional[np.ndarray] = None
    if D >= 2 and rng.random() < covariate_prob:
        max_cov = max(1, int(np.floor(D * max_covariate_frac)))
        n_cov = int(rng.integers(1, max_cov + 1))
        cov_idx = rng.choice(D, size=n_cov, replace=False)
        cov_mask = np.zeros(D, dtype=bool)
        cov_mask[cov_idx] = True
        feat_dynamic_real = Z[cov_mask].astype(np.float32)
        target = Z[~cov_mask].astype(np.float32)
    else:
        target = Z.astype(np.float32)

    item = {
        "start": np.datetime64("2000-01-01 00:00", "s"),
        "target": target,  # always 2D (D_target, L) for homogeneous schema
    }
    if feat_dynamic_real is not None:
        item["feat_dynamic_real"] = feat_dynamic_real
    return item


# ============================ CLI ============================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", "--num-series", type=int, default=10000)
    parser.add_argument("-J", "--max-kernels", type=int, default=5)
    parser.add_argument("--length", type=int, default=LENGTH,
                        help="Fixed length when --min-length/--max-length are not set.")
    parser.add_argument("--min-length", type=int, default=None,
                        help="If set together with --max-length, length is sampled "
                             "uniformly per item in [min_length, max_length].")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--min-variates", type=int, default=2)
    parser.add_argument("--max-variates", type=int, default=8)
    parser.add_argument("--min-sources", type=int, default=2)
    parser.add_argument("--max-sources", type=int, default=8)
    parser.add_argument("--max-lag", type=int, default=32,
                        help="Maximum lag for the sequential multivariatizer.")
    parser.add_argument("--covariate-prob", type=float, default=0.3,
                        help="Probability an item is converted to a covariate-informed task.")
    parser.add_argument("--max-covariate-frac", type=float, default=0.5,
                        help="Upper bound on the fraction of variates designated as known covariates.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-o", "--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng_master = np.random.default_rng(args.seed)
    item_seeds = rng_master.integers(0, 2**31 - 1, size=args.num_series).tolist()

    dataset = Parallel(n_jobs=-1)(
        delayed(generate_multivariate_item)(
            seed=int(s),
            length=args.length,
            max_kernels=args.max_kernels,
            min_variates=args.min_variates,
            max_variates=args.max_variates,
            min_sources=args.min_sources,
            max_sources=args.max_sources,
            covariate_prob=args.covariate_prob,
            max_covariate_frac=args.max_covariate_frac,
            max_lag=args.max_lag,
            min_length=args.min_length,
            max_length=args.max_length,
        )
        for s in tqdm(item_seeds)
    )

    multi_items = [it for it in dataset if "feat_dynamic_real" not in it]
    cov_items = [it for it in dataset if "feat_dynamic_real" in it]
    multi_out = args.out
    cov_out = args.out.with_name(args.out.stem + ".cov" + args.out.suffix)

    if multi_items:
        ArrowWriter(compression="lz4").write_to_file(multi_items, path=multi_out)
        print(f"[multivariatize] wrote {len(multi_items)} multivariate items to {multi_out}")
    if cov_items:
        ArrowWriter(compression="lz4").write_to_file(cov_items, path=cov_out)
        print(f"[multivariatize] wrote {len(cov_items)} covariate-informed items to {cov_out}")


if __name__ == "__main__":
    main()
