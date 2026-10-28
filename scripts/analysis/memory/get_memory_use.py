# SPDX-FileCopyrightText: 2026 Deakin University and the LOTUS authors
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT
"""Measure peak GPU memory of ONE training step for a model x dataset pair.

Backs the paper's "Computational Cost" appendix / memory-scaling figure
(Fig. 7): run it over the SimulationQADataset grid (series length L x number
of series N) with scripts/analysis/memory/run_all_memory.sh and plot the CSV
with plot_memory_scaling.py.

Restored from OpenTSLM's evaluation/memory/get_memory_use.py with three
changes: repo-relative imports (no personal paths), the LOTUS model
(OpenTSLMSPFixPos2D, --model OpenTSLMSPFixPos2D with the paper's encoder
flags) next to the OpenTSLM-SP / Flamingo baselines, and an output path that
follows the FTS_* convention (default $FTS_RESULTS_ROOT/memory/memory_use.csv).

Usage:
    python scripts/analysis/memory/get_memory_use.py \
        --llm_id google/gemma-3-270m --model OpenTSLMSPFixPos2D \
        --dataset SimulationQADataset --length 1000 --num_series 3 --device cuda
"""

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# repo-relative import of the opentslm package (no personal paths)
REPO_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_DIR / "src"))

try:  # optional: per-process GPU memory as seen by the driver
    import pynvml  # type: ignore

    _NVML_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    pynvml = None
    _NVML_AVAILABLE = False

try:  # optional: CPU RSS fallback when no GPU is present
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None

from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)

MODEL_CHOICES = ["OpenTSLMSPFixPos2D", "OpenTSLMSP", "OpenTSLMFlamingo"]
DATASET_CHOICES = [
    "TSQADataset",
    "HARCoTQADataset",
    "SleepEDFCoTQADataset",
    "ECGQACoTQADataset",
    "SimulationQADataset",
]


def default_results_csv() -> str:
    root = os.environ.get("FTS_RESULTS_ROOT")
    if root:
        return os.path.join(root, "memory", "memory_use.csv")
    return str(REPO_DIR / "memory_use.csv")


def get_device(device_arg: str | None) -> str:
    if device_arg:
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def measure_peak_cpu_bytes() -> int:
    """Resident set size of this process in bytes (-1 if psutil is missing)."""
    if psutil is None:
        return -1
    try:
        return int(psutil.Process().memory_info().rss)
    except Exception:
        return -1


def nvml_current_process_bytes() -> int:
    if not _NVML_AVAILABLE or not torch.cuda.is_available():
        return -1
    try:
        pynvml.nvmlInit()
        pid = os.getpid()
        total_bytes = 0
        found = False
        for device_index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses_v3(handle)
            except Exception:
                procs = []
            try:
                procs_gfx = pynvml.nvmlDeviceGetGraphicsRunningProcesses_v3(handle)
            except Exception:
                procs_gfx = []
            for p in list(procs) + list(procs_gfx):
                if (
                    int(p.pid) == pid
                    and p.usedGpuMemory is not None
                    and p.usedGpuMemory >= 0
                ):
                    total_bytes += int(p.usedGpuMemory)
                    found = True
        return total_bytes if found else -1
    except Exception:
        return -1


def build_optimizer(model, model_type: str, base_lr: float = 2e-4):
    """AdamW over the parameters each model type trains.

    OpenTSLMSPFixPos2D: every trainable parameter (encoder, projector, decoder,
    decoder projector and, when enabled, the LoRA adapter; the LLM base weights
    are frozen by the model). OpenTSLMSP: encoder + projector (as upstream).
    Flamingo: upstream's gated-cross-attention grouping.
    """
    if model_type == "OpenTSLMSPFixPos2D":
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            return None
        return torch.optim.AdamW([{"params": params, "weight_decay": 0.1}], lr=base_lr)
    if model_type == "OpenTSLMSP":
        enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
        proj_params = [p for p in model.projector.parameters() if p.requires_grad]
        param_groups = []
        if enc_params:
            param_groups.append({"params": enc_params, "weight_decay": 0.1})
        if proj_params:
            param_groups.append({"params": proj_params, "weight_decay": 0.1})
        return torch.optim.AdamW(param_groups, lr=base_lr) if param_groups else None
    # Flamingo-like
    trainable = [
        (n, p)
        for n, p in model.named_parameters()
        if p.requires_grad and not getattr(p, "exclude_from_optimizer", False)
    ]
    params_with_wd = [p for n, p in trainable if "gated_cross_attn" in n]
    params_without_wd = [p for n, p in trainable if "gated_cross_attn" not in n]
    if not trainable:
        return None
    return torch.optim.AdamW(
        [
            {"params": params_with_wd, "weight_decay": 0.1},
            {"params": params_without_wd, "weight_decay": 0.0},
        ],
        lr=base_lr,
    )


def train_for_steps(model, model_type: str, dataset, steps: int) -> Tuple[float, int, int, int]:
    model.train()
    optimizer = build_optimizer(model, model_type)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    cpu_memory_baseline = measure_peak_cpu_bytes()
    last_loss = 0.0
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda b: extend_time_series_to_match_patch_size_and_aggregate(b),
        drop_last=False,
    )

    def _to_gb(val: int) -> float:
        return float(val) / (1024.0**3) if isinstance(val, (int, float)) and val >= 0 else 0.0

    pbar = tqdm(total=steps, desc="Training", leave=False)
    max_peak_bytes = -1
    max_reserved_bytes = -1
    max_nvml_bytes = -1
    max_cpu_bytes = cpu_memory_baseline
    step = 0
    for batch in loader:
        if optimizer:
            optimizer.zero_grad(set_to_none=True)
        loss = model.compute_loss(batch)
        if optimizer and loss.requires_grad:
            loss.backward()
            optimizer.step()
        last_loss = float(loss.detach().item())

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            max_peak_bytes = max(max_peak_bytes, int(torch.cuda.max_memory_allocated()))
            max_reserved_bytes = max(max_reserved_bytes, int(torch.cuda.max_memory_reserved()))
            max_nvml_bytes = max(max_nvml_bytes, nvml_current_process_bytes())
        max_cpu_bytes = max(max_cpu_bytes, measure_peak_cpu_bytes())

        pbar.set_postfix(
            {
                "alloc_gb": f"{_to_gb(max_peak_bytes):.2f}",
                "res_gb": f"{_to_gb(max_reserved_bytes):.2f}",
                "nvml_gb": f"{_to_gb(max_nvml_bytes):.2f}",
                "cpu_gb": f"{_to_gb(max_cpu_bytes):.2f}",
            }
        )
        step += 1
        pbar.update(1)
        if step >= steps:
            break
    pbar.close()

    if torch.cuda.is_available():
        return last_loss, max_peak_bytes, max_reserved_bytes, max_nvml_bytes
    # CPU-only: report RSS in every column so the CSV stays well-formed.
    return last_loss, max_cpu_bytes, max_cpu_bytes, max_cpu_bytes


def ensure_csv(path: str, header: List[str]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)


def append_row(path: str, row: List[Any]):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


def run_for_dataset(model_name: str, model, dataset_name: str, dataset_obj, steps: int) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "model": model_name,
        "dataset": dataset_name,
        "loss": None,
        "peak_cuda_bytes": None,
        "peak_cuda_reserved_bytes": -1,
        "nvml_peak_bytes": -1,
        "status": "ok",
        "error": "",
    }
    try:
        steps = max(1, min(len(dataset_obj), steps))
        loss, peak, peak_reserved, nvml_peak = train_for_steps(model, model_name, dataset_obj, steps)
        result["loss"] = loss
        result["peak_cuda_bytes"] = peak
        result["peak_cuda_reserved_bytes"] = peak_reserved
        result["nvml_peak_bytes"] = nvml_peak
    except Exception as e:  # OOM lands here: recorded as status=error, not a crash
        result["status"] = "error"
        result["error"] = str(e)
    return result


def build_lotus_model(args, device: str):
    """OpenTSLMSPFixPos2D with the paper's encoder geometry (defaults = 270M run)."""
    from opentslm.model.llm.OpenTSLMSPFixPos2D import OpenTSLMSPFixPos2D

    ns = argparse.Namespace(
        use_only_time_pe=False,
        use_only_channel_pe=False,
        is_pretraining=False,
        gradient_checkpointing=bool(args.gradient_checkpointing),
        use_freq_embed=bool(args.use_freq_embed),
        pe_type=args.pe_type,
        pe_base_t=args.pe_base_t,
        pe_base_c=args.pe_base_c,
        patch_size=args.patch_size,
        freq_mode=args.freq_mode,
        enc_dec_type=args.enc_dec_type,
        mlp_enc_hidden_dim=args.mlp_enc_hidden_dim,
        mlp_enc_ff_dim=args.mlp_enc_ff_dim,
        mlp_dec_hidden_dim=args.mlp_dec_hidden_dim,
        mlp_dec_ff_dim=args.mlp_dec_ff_dim,
        trans_enc_dim=args.trans_enc_dim,
        trans_enc_num_layers=args.trans_enc_num_layers,
        trans_enc_num_heads=args.trans_enc_num_heads,
        trans_enc_ff_dim=args.trans_enc_ff_dim,
        trans_enc_output_dim=None,
        trans_dec_dim=args.trans_dec_dim,
        trans_dec_num_layers=args.trans_dec_num_layers,
        trans_dec_num_heads=args.trans_dec_num_heads,
        trans_dec_ff_dim=args.trans_dec_ff_dim,
        llm_dtype=args.llm_dtype,
        llm_mv_position_ids=True,
    )
    model = OpenTSLMSPFixPos2D(args=ns, llm_id=args.llm_id, device=device)
    if args.lora_r > 0:
        # Phase-2 configuration trains a LoRA adapter on the LLM as well.
        model.enable_lora(lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0)
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Measure peak memory of a few training steps for a model x dataset pair."
    )
    parser.add_argument("--llm_id", "-llm_id", required=True, help="HuggingFace id of the backbone LLM")
    parser.add_argument("--model", required=True, choices=MODEL_CHOICES,
                        help="OpenTSLMSPFixPos2D = LOTUS; OpenTSLMSP / OpenTSLMFlamingo = OpenTSLM baselines")
    parser.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    parser.add_argument("--device", default=None, help="cuda, cuda:0 or cpu (default: cuda if available)")
    parser.add_argument("--length", type=int, default=100,
                        help="Series length L for SimulationQADataset (default 100)")
    parser.add_argument("--num_series", type=int, default=1,
                        help="Number of series N for SimulationQADataset (default 1)")
    parser.add_argument("--steps", type=int, default=100,
                        help="Training steps to run (peak is taken over all of them; default 100)")
    parser.add_argument("--results_csv", default=default_results_csv(),
                        help="CSV to append to (default $FTS_RESULTS_ROOT/memory/memory_use.csv)")
    # LOTUS encoder geometry (defaults = released 270M run; pass the transformer
    # flags for the 1B / Qwen3 configuration, see configs/*.env)
    g = parser.add_argument_group("OpenTSLMSPFixPos2D architecture")
    g.add_argument("--enc_dec_type", default="mlp", choices=["mlp", "transformer"])
    g.add_argument("--use_freq_embed", action="store_true", default=True)
    g.add_argument("--no_freq_embed", dest="use_freq_embed", action="store_false")
    g.add_argument("--patch_size", type=int, default=32)
    g.add_argument("--pe_type", default="sincos")
    g.add_argument("--pe_base_t", type=float, default=None)
    g.add_argument("--pe_base_c", type=float, default=10.0)
    g.add_argument("--freq_mode", default="linear", choices=["linear", "spectral"])
    g.add_argument("--mlp_enc_hidden_dim", type=int, default=64)
    g.add_argument("--mlp_enc_ff_dim", type=int, default=256)
    g.add_argument("--mlp_dec_hidden_dim", type=int, default=64)
    g.add_argument("--mlp_dec_ff_dim", type=int, default=256)
    g.add_argument("--trans_enc_dim", type=int, default=192)
    g.add_argument("--trans_enc_num_layers", type=int, default=6)
    g.add_argument("--trans_enc_num_heads", type=int, default=8)
    g.add_argument("--trans_enc_ff_dim", type=int, default=896)
    g.add_argument("--trans_dec_dim", type=int, default=192)
    g.add_argument("--trans_dec_num_layers", type=int, default=4)
    g.add_argument("--trans_dec_num_heads", type=int, default=8)
    g.add_argument("--trans_dec_ff_dim", type=int, default=896)
    g.add_argument("--lora_r", type=int, default=4, help="Phase-2 LoRA rank (0 = frozen LLM)")
    g.add_argument("--lora_alpha", type=int, default=8)
    g.add_argument("--llm_dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    g.add_argument("--gradient_checkpointing", action="store_true")
    args = parser.parse_args()

    device = get_device(args.device)

    header = [
        "timestamp", "llm_id", "device", "model", "dataset", "loss",
        "peak_cuda_bytes", "peak_cuda_gb",
        "peak_cuda_reserved_bytes", "peak_cuda_reserved_gb",
        "nvml_peak_bytes", "nvml_peak_gb",
        "status", "error",
    ]
    ensure_csv(args.results_csv, header)

    # Models are imported lazily: Flamingo needs the open_flamingo package,
    # which the LOTUS environment does not require.
    if args.model == "OpenTSLMSPFixPos2D":
        model = build_lotus_model(args, device)
    elif args.model == "OpenTSLMFlamingo":
        from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo

        model = OpenTSLMFlamingo(
            device=device, llm_id=args.llm_id, cross_attn_every_n_layers=1,
            gradient_checkpointing=True,
        )
    elif args.model == "OpenTSLMSP":
        from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP

        model = OpenTSLMSP(llm_id=args.llm_id, device=device)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    model.to(device)
    eos = model.get_eos_token()

    if args.dataset == "TSQADataset":
        from opentslm.time_series_datasets.TSQADataset import TSQADataset

        dataset = TSQADataset(split="train", EOS_TOKEN=eos)
        dataset_name = "TSQA"
    elif args.dataset == "HARCoTQADataset":
        from opentslm.time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset

        dataset = HARCoTQADataset(split="train", EOS_TOKEN=eos)
        dataset_name = "HAR-CoT"
    elif args.dataset == "SleepEDFCoTQADataset":
        from opentslm.time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset

        dataset = SleepEDFCoTQADataset(split="train", EOS_TOKEN=eos)
        dataset_name = "SleepEDF-CoT"
    elif args.dataset == "ECGQACoTQADataset":
        from opentslm.time_series_datasets.ecg_qa.ECGQACoTQADataset import ECGQACoTQADataset

        dataset = ECGQACoTQADataset(split="train", EOS_TOKEN=eos, max_samples=1,
                                    preload_processed_data=False)
        dataset_name = "ECG-QA-CoT"
    elif args.dataset == "SimulationQADataset":
        from opentslm.time_series_datasets.simulation.SimulationQADataset import SimulationQADataset

        dataset = SimulationQADataset(split="train", EOS_TOKEN=eos,
                                      length=args.length, num_series=args.num_series)
        dataset_name = f"Simulation-L{args.length}-N{args.num_series}"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    res = run_for_dataset(args.model, model, dataset_name, dataset, args.steps)

    def _gb(b) -> str:
        return f"{float(b) / (1024.0 ** 3):.4f}" if isinstance(b, (int, float)) and b >= 0 else "-1"

    append_row(
        args.results_csv,
        [
            datetime.utcnow().isoformat(), args.llm_id, device,
            res["model"], res["dataset"], res["loss"],
            res["peak_cuda_bytes"], _gb(res["peak_cuda_bytes"]),
            res["peak_cuda_reserved_bytes"], _gb(res["peak_cuda_reserved_bytes"]),
            res["nvml_peak_bytes"], _gb(res["nvml_peak_bytes"]),
            res["status"], res["error"],
        ],
    )
    print(f"[{res['status']}] {args.model} / {dataset_name}: "
          f"peak_reserved={_gb(res['peak_cuda_reserved_bytes'])} GB "
          f"-> {args.results_csv}")


if __name__ == "__main__":
    main()
