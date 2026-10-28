"""Collect synthetic MTS datasets into a unified schema and emit statistics.

Sources of two kinds:
  - kind="arrow": gluonts ArrowFile (TimePFN LMC, chronos multivariatize)
  - kind="symtime_pt": tree of `<seed>/ID=*_OD=*.pt` torch tensors

Each `target` is cast to float32 with shape `(n_variates, length)` and streamed
into `<out_dir>/unified_<source>.arrow` via pyarrow's RecordBatchFileWriter (so
huge inputs do not need to fit in RAM).

Samples violating the constraint envelope (`n_variates` outside [MIN_C, MAX_C]
or `length` outside [MIN_L, MAX_L]) are dropped during collection. Per-source
counts of dropped samples are reported in the stats file.

Per-source statistics computed in the same pass:
  - n_samples kept / dropped
  - length: min / max / mean / std (over kept samples)
  - n_variates: min / max / mean / std (over kept samples)
  - per-variate value mean: mean / std over all (sample, variate) pairs
  - per-variate value std:  mean / std over all (sample, variate) pairs
  - global value mean / std: exact, computed with running sums
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import pickle as _pickle
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
from gluonts.dataset.arrow import ArrowFile
from tqdm import tqdm

# Upstream checkouts and output root come from the environment so the script
# runs on any machine. SYNTH_ROOT is the fallback parent dir for all three.
_SYNTH_ROOT = os.environ.get("SYNTH_ROOT", os.path.join(os.getcwd(), "synthetic_data"))
_CHRONOS = os.environ.get("SYNTH_CHRONOS_REPO", os.path.join(_SYNTH_ROOT, "chronos-forecasting"))
_TIMEPFN = os.environ.get("SYNTH_TIMEPFN_REPO", os.path.join(_SYNTH_ROOT, "TimePFN"))
_SYMTIME = os.environ.get("SYNTH_SYMTIME_REPO", os.path.join(_SYNTH_ROOT, "SymTime"))
_OUT_DIR = os.environ.get("FTS_SYNTHETIC_DIR", os.path.join(_SYNTH_ROOT, "unified", "data"))


class _S2GeneratorStubUnpickler(_pickle.Unpickler):
    """Unpickler that stubs out `S2Generator.*` classes.

    SymTime .pt files store {"symbol": <S2Generator object>, "excitation": tensor,
    "response": tensor}. We only consume `response`; `symbol` references the
    `S2Generator` package, which is not installed in this env (and need not be).
    Resolving those classes to empty stubs lets `torch.load` succeed without the
    dependency — the stubbed `symbol` object is simply discarded.
    """

    def find_class(self, module, name):
        if module == "S2Generator" or module.startswith("S2Generator."):
            return type(f"_S2GenStub_{name}", (), {})
        return super().find_class(module, name)


class _s2_stub_pickle_module:
    """Drop-in `pickle_module` for `torch.load` using the stub unpickler above."""

    Unpickler = _S2GeneratorStubUnpickler
    load = staticmethod(_pickle.load)
    loads = staticmethod(_pickle.loads)


# (name, kind, path) tuples. kind selects the reader.
DEFAULT_SOURCES: list[dict] = [
    {
        "name": "timepfn_lmc",
        "kind": "arrow",
        "path": f"{_TIMEPFN}/datasets/synthetic/lmc_synth_C3-10_L1024-4096.arrow",
    },
    # TimePFN LMC-Synth diversity sweep — produced by slurm/gen_lmc_sweep.slurm
    # (three length/channel/LMC-structure regimes).
    {
        "name": "timepfn_lmc_short",
        "kind": "arrow",
        "path": f"{_TIMEPFN}/datasets/synthetic/lmc_short.arrow",
    },
    {
        "name": "timepfn_lmc_medium",
        "kind": "arrow",
        "path": f"{_TIMEPFN}/datasets/synthetic/lmc_medium.arrow",
    },
    {
        "name": "timepfn_lmc_long",
        "kind": "arrow",
        "path": f"{_TIMEPFN}/datasets/synthetic/lmc_long.arrow",
    },
    # Chronos diversity sweep — produced by slurm/multivariatize_sweep.slurm
    # (three length/complexity regimes that together form the chronos
    # contribution to the pretraining corpus).
    {
        "name": "chronos_short",
        "kind": "arrow",
        "path": f"{_CHRONOS}/data/mv_short.arrow",
    },
    {
        "name": "chronos_medium",
        "kind": "arrow",
        "path": f"{_CHRONOS}/data/mv_medium.arrow",
    },
    {
        "name": "chronos_long",
        "kind": "arrow",
        "path": f"{_CHRONOS}/data/mv_long.arrow",
    },
    # Univariate KernelSynth diversity sweeps. Each item has 1D target shape
    # (length,); the collector promotes to (1, length). The default envelope
    # rejects n_variates<3 — pass `--min-variates 1` to include these.
    # Chronos flavor — slurm/kernel_synth_sweep.slurm
    {
        "name": "kernel_synth_short",
        "kind": "arrow",
        "path": f"{_CHRONOS}/data/ks_short.arrow",
    },
    {
        "name": "kernel_synth_medium",
        "kind": "arrow",
        "path": f"{_CHRONOS}/data/ks_medium.arrow",
    },
    {
        "name": "kernel_synth_long",
        "kind": "arrow",
        "path": f"{_CHRONOS}/data/ks_long.arrow",
    },
    # TimePFN flavor — slurm/gen_kernelsynth_sweep.slurm. Uses TimePFN's
    # kernel bank + the faster `sample_from_gp_prior_efficient` sampler.
    {
        "name": "timepfn_kernelsynth_short",
        "kind": "arrow",
        "path": f"{_TIMEPFN}/datasets/synthetic/ks_short.arrow",
    },
    {
        "name": "timepfn_kernelsynth_medium",
        "kind": "arrow",
        "path": f"{_TIMEPFN}/datasets/synthetic/ks_medium.arrow",
    },
    {
        "name": "timepfn_kernelsynth_long",
        "kind": "arrow",
        "path": f"{_TIMEPFN}/datasets/synthetic/ks_long.arrow",
    },
    # Original single-config chronos run (slurm/multivariatize.slurm defaults,
    # 10k items). It IS part of the released 844k corpus
    # (unified_chronos_multivariate.arrow), so it stays enabled here. Sources
    # that don't exist on disk are skipped automatically by `main()`.
    {
        "name": "chronos_multivariate",
        "kind": "arrow",
        "path": f"{_CHRONOS}/data/multivariate-kernelsynth.arrow",
    },
    {
        "name": "symtime_s2generator",
        "kind": "symtime_pt",
        "path": f"{_SYMTIME}/data/datasets/pretrain_data",
    },
    # SymTime / S2Generator diversity sweep — produced by
    # slurm/s2generate_sweep.slurm. Each tag writes to its own pretrain_data_<tag>
    # directory so the collector can read each cohort as a distinct source.
    {
        "name": "symtime_s2generator_short",
        "kind": "symtime_pt",
        "path": f"{_SYMTIME}/data/datasets/pretrain_data_short",
    },
    {
        "name": "symtime_s2generator_medium",
        "kind": "symtime_pt",
        "path": f"{_SYMTIME}/data/datasets/pretrain_data_medium",
    },
    {
        "name": "symtime_s2generator_long",
        "kind": "symtime_pt",
        "path": f"{_SYMTIME}/data/datasets/pretrain_data_long",
    },
]


# Constraint envelope; samples outside this are dropped.
DEFAULT_MIN_VARIATES = 3
DEFAULT_MAX_VARIATES = 10
DEFAULT_MIN_LENGTH = 1024
DEFAULT_MAX_LENGTH = 4096


UNIFIED_SCHEMA = pa.schema([
    pa.field("source", pa.string()),
    pa.field("sample_id", pa.int32()),
    pa.field("n_variates", pa.int32()),
    pa.field("length", pa.int32()),
    pa.field("target", pa.list_(pa.list_(pa.float32()))),  # [n_variates][length]
])


def _normalize_target(arr) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 1:
        a = a[None, :]
    elif a.ndim != 2:
        raise ValueError(f"unexpected target ndim={a.ndim} shape={a.shape}")
    return a.astype(np.float32, copy=False)


def _flush_batch(writer, batch_rows):
    if not batch_rows:
        return
    table = pa.Table.from_pylist(batch_rows, schema=UNIFIED_SCHEMA)
    for rb in table.to_batches():
        writer.write_batch(rb)


def _iter_arrow(src_path: str):
    """Yield (target_ndarray,) for each row of a gluonts arrow file."""
    af = ArrowFile(src_path)
    try:
        total = len(af)
    except Exception:
        total = None
    for row in tqdm(af, total=total, desc=Path(src_path).name):
        yield _normalize_target(row["target"])


def _iter_symtime_pt(root: str):
    """Yield target ndarrays for each .pt file under <root>/<seed>/ID=*_OD=*.pt.

    The torch payload is `{"symbol", "excitation", "response"}`. `response` is
    stored as (length, output_dim); we transpose it to (n_variates, length) to
    match the unified schema. `symbol` is unpickled through a stub (see
    `_S2GeneratorStubUnpickler`) so the `S2Generator` package is not required.
    """
    import torch  # lazy import — only needed for SymTime
    paths = sorted(glob.glob(os.path.join(root, "*", "ID=*_OD=*.pt")))
    n_unreadable = 0
    for p in tqdm(paths, desc=Path(root).name):
        try:
            d = torch.load(
                p, weights_only=False, map_location="cpu",
                pickle_module=_s2_stub_pickle_module,
            )
            resp = d["response"]
        except Exception:
            # A .pt file may be truncated/partial if SymTime generation is
            # still writing it concurrently — skip rather than abort the source.
            n_unreadable += 1
            continue
        if hasattr(resp, "detach"):
            resp = resp.detach()
        if hasattr(resp, "numpy"):
            resp = resp.numpy()
        resp = np.asarray(resp)
        if resp.ndim == 2:
            resp = resp.T  # (length, output_dim) -> (output_dim, length)
        yield _normalize_target(resp)
    if n_unreadable:
        print(f"[symtime] skipped {n_unreadable} unreadable/partial .pt files under {root}",
              file=sys.stderr)


def _make_iter(src: dict):
    if src["kind"] == "arrow":
        return _iter_arrow(src["path"])
    if src["kind"] == "symtime_pt":
        return _iter_symtime_pt(src["path"])
    raise ValueError(f"unknown kind: {src['kind']}")


def collect_one(src: dict, out_path: str, *, min_c, max_c, min_l, max_l,
                batch_size: int = 32) -> dict:
    name = src["name"]
    lengths: list[int] = []
    n_variates: list[int] = []
    pv_means: list[float] = []
    pv_stds: list[float] = []
    g_count = 0
    g_sum = 0.0
    g_sumsq = 0.0
    n_dropped_variates = 0
    n_dropped_length = 0
    n_dropped_nan = 0
    sample_id = 0

    batch: list[dict] = []
    with pa.OSFile(out_path, "wb") as sink:
        with pa.ipc.new_file(sink, UNIFIED_SCHEMA) as writer:
            for target in _make_iter(src):
                C, L = int(target.shape[0]), int(target.shape[1])
                if not (min_c <= C <= max_c):
                    n_dropped_variates += 1
                    continue
                if not (min_l <= L <= max_l):
                    n_dropped_length += 1
                    continue
                if not np.isfinite(target).all():
                    # Corrupt sample (NaN/inf) — would poison downstream training.
                    n_dropped_nan += 1
                    continue

                lengths.append(L)
                n_variates.append(C)
                pv_means.extend(target.mean(axis=1).astype(np.float64).tolist())
                pv_stds.extend(target.std(axis=1).astype(np.float64).tolist())
                td = target.astype(np.float64, copy=False)
                g_count += td.size
                g_sum += float(td.sum())
                g_sumsq += float(np.square(td).sum())

                batch.append({
                    "source": name,
                    "sample_id": sample_id,
                    "n_variates": C,
                    "length": L,
                    "target": [v.tolist() for v in target],
                })
                sample_id += 1
                if len(batch) >= batch_size:
                    _flush_batch(writer, batch)
                    batch.clear()
            _flush_batch(writer, batch)
            batch.clear()

    lengths_a = np.asarray(lengths, dtype=np.int64)
    nv_a = np.asarray(n_variates, dtype=np.int64)
    pv_means_a = np.asarray(pv_means, dtype=np.float64)
    pv_stds_a = np.asarray(pv_stds, dtype=np.float64)

    g_mean = g_sum / g_count if g_count else float("nan")
    g_var = (g_sumsq / g_count - g_mean * g_mean) if g_count else float("nan")
    g_std = float(np.sqrt(max(g_var, 0.0))) if g_count else float("nan")

    def _stats(a):
        if a.size == 0:
            return {"min": None, "max": None, "mean": None, "std": None}
        return {"min": int(a.min()) if a.dtype.kind in "iu" else float(a.min()),
                "max": int(a.max()) if a.dtype.kind in "iu" else float(a.max()),
                "mean": float(a.mean()),
                "std": float(a.std())}

    return {
        "source": name,
        "kind": src["kind"],
        "src_path": src["path"],
        "out_path": out_path,
        "n_samples_kept": int(lengths_a.size),
        "n_dropped_variates": int(n_dropped_variates),
        "n_dropped_length": int(n_dropped_length),
        "n_dropped_nan": int(n_dropped_nan),
        "length": _stats(lengths_a),
        "n_variates": _stats(nv_a),
        "per_variate_value_mean": {
            "mean": float(pv_means_a.mean()) if pv_means_a.size else None,
            "std": float(pv_means_a.std()) if pv_means_a.size else None,
        },
        "per_variate_value_std": {
            "mean": float(pv_stds_a.mean()) if pv_stds_a.size else None,
            "std": float(pv_stds_a.std()) if pv_stds_a.size else None,
        },
        "global_value": {
            "mean": g_mean,
            "std": g_std,
            "count": int(g_count),
        },
    }


def write_stats(stats: list[dict], out_dir: Path, envelope: dict) -> None:
    payload = {"envelope": envelope, "datasets": stats}
    with open(out_dir / "stats.json", "w") as f:
        json.dump(payload, f, indent=2)

    columns = [
        "source", "kind", "n_samples_kept", "n_dropped_variates", "n_dropped_length",
        "n_dropped_nan",
        "length_min", "length_max", "length_mean", "length_std",
        "n_variates_min", "n_variates_max", "n_variates_mean", "n_variates_std",
        "per_var_mean_mean", "per_var_mean_std",
        "per_var_std_mean", "per_var_std_std",
        "global_value_mean", "global_value_std", "global_value_count",
        "src_path", "out_path",
    ]
    with open(out_dir / "stats.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for s in stats:
            w.writerow([
                s["source"], s["kind"], s["n_samples_kept"],
                s["n_dropped_variates"], s["n_dropped_length"], s.get("n_dropped_nan", ""),
                s["length"]["min"], s["length"]["max"], s["length"]["mean"], s["length"]["std"],
                s["n_variates"]["min"], s["n_variates"]["max"], s["n_variates"]["mean"], s["n_variates"]["std"],
                s["per_variate_value_mean"]["mean"], s["per_variate_value_mean"]["std"],
                s["per_variate_value_std"]["mean"], s["per_variate_value_std"]["std"],
                s["global_value"]["mean"], s["global_value"]["std"], s["global_value"]["count"],
                s["src_path"], s["out_path"],
            ])
    print(f"[stats] wrote {out_dir / 'stats.json'}")
    print(f"[stats] wrote {out_dir / 'stats.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=_OUT_DIR)
    ap.add_argument("--sources-json", default=None,
                    help="optional JSON file with a list of {name, kind, path} entries")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--min-variates", type=int, default=DEFAULT_MIN_VARIATES)
    ap.add_argument("--max-variates", type=int, default=DEFAULT_MAX_VARIATES)
    ap.add_argument("--min-length",   type=int, default=DEFAULT_MIN_LENGTH)
    ap.add_argument("--max-length",   type=int, default=DEFAULT_MAX_LENGTH)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = DEFAULT_SOURCES
    if args.sources_json:
        with open(args.sources_json) as f:
            sources = json.load(f)
    if args.only:
        sources = [s for s in sources if s["name"] in set(args.only)]

    envelope = {
        "min_variates": args.min_variates,
        "max_variates": args.max_variates,
        "min_length": args.min_length,
        "max_length": args.max_length,
    }

    # Preserve stats for sources not processed in this run (e.g. when --only is
    # used) so the stats files remain a complete picture rather than being
    # clobbered down to just the subset.
    processed_names = {s["name"] for s in sources}
    all_stats: list[dict] = []
    stats_json_path = out_dir / "stats.json"
    if stats_json_path.exists():
        try:
            with open(stats_json_path) as f:
                prev = json.load(f)
            all_stats = [
                s for s in prev.get("datasets", [])
                if s.get("source") not in processed_names
            ]
        except (json.JSONDecodeError, OSError):
            all_stats = []

    for src in sources:
        if not os.path.exists(src["path"]):
            print(f"[skip] {src['name']}: source missing -> {src['path']}", file=sys.stderr)
            continue
        out_arrow = str(out_dir / f"unified_{src['name']}.arrow")
        t0 = time.time()
        s = collect_one(
            src, out_arrow,
            min_c=args.min_variates, max_c=args.max_variates,
            min_l=args.min_length,   max_l=args.max_length,
            batch_size=args.batch_size,
        )
        s["elapsed_s"] = round(time.time() - t0, 2)
        all_stats.append(s)
        write_stats(all_stats, out_dir, envelope)

    if not all_stats:
        print("no datasets were collected", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
