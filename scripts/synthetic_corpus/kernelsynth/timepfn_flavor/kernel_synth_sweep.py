"""Univariate KernelSynth generator (TimePFN flavor) with a configurable length.

TimePFN's `kernel-synth.py` hard-codes `LENGTH = 1024` as a module constant and
its `generate_time_series` takes no length argument. This wrapper imports the
upstream `KERNEL_BANK`, `random_binary_map`, and `sample_from_gp_prior_efficient`
and runs the same composition logic on a user-specified time grid, so length and
max_kernels can both be swept without modifying upstream code.

Uses `sample_from_gp_prior_efficient` (multivariate_normal with method='eigh'),
which is the faster sampler available in TimePFN's module, in contrast to the
sklearn-based `sample_from_gp_prior` used by upstream `generate_time_series`.

Output: GluonTS-compatible Arrow file with 1D `target` of shape `(length,)`.
"""
from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path

import numpy as np
from gluonts.dataset.arrow import ArrowWriter
from joblib import Parallel, delayed
from tqdm.auto import tqdm

# Pull the kernel bank + faster sampler from TimePFN's kernel-synth.py module.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module
_ks = import_module("kernel-synth")
KERNEL_BANK = _ks.KERNEL_BANK
random_binary_map = _ks.random_binary_map
sample_from_gp_prior_efficient = _ks.sample_from_gp_prior_efficient


def generate_one(length: int, max_kernels: int) -> dict:
    """Same logic as upstream `generate_time_series`, but at a configurable length
    and via the faster `sample_from_gp_prior_efficient` sampler.
    """
    while True:
        X = np.linspace(0, 1, length)
        selected = np.random.choice(
            KERNEL_BANK, np.random.randint(1, max_kernels + 1), replace=True
        )
        kernel = functools.reduce(random_binary_map, selected)
        try:
            ts = sample_from_gp_prior_efficient(kernel=kernel, X=X)
        except np.linalg.LinAlgError as err:
            print("Error caught:", err)
            continue
        return {"start": np.datetime64("2000-01-01 00:00", "s"), "target": np.asarray(ts).squeeze()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-N", "--num-series", type=int, default=20000)
    p.add_argument("-J", "--max-kernels", type=int, default=5)
    p.add_argument("-L", "--length", type=int, default=1024)
    p.add_argument("-o", "--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Seed numpy's legacy global state so upstream `np.random.*` calls are
    # reproducible across runs with the same --seed (worker processes inherit
    # the seed at fork time).
    np.random.seed(args.seed)

    dataset = Parallel(n_jobs=-1)(
        delayed(generate_one)(length=args.length, max_kernels=args.max_kernels)
        for _ in tqdm(range(args.num_series))
    )

    ArrowWriter(compression="lz4").write_to_file(dataset, path=args.out)
    print(f"[kernel_synth_sweep] wrote {args.num_series} series (L={args.length}, "
          f"max_kernels={args.max_kernels}) -> {args.out}")


if __name__ == "__main__":
    main()
