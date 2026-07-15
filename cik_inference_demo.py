#!/usr/bin/env python
"""
CiK inference demo for an OpenTSLM checkpoint.

Loads an OpenTSLMSPFixPos2D checkpoint, picks one or more Context-is-Key (CiK)
tasks, runs the SAME forecaster used by the benchmark (`OpenTSLMForecaster`), and
for each example prints:
  * the task's textual context (background / scenario / constraints),
  * the numeric history summary,
  * the model's probabilistic forecast (median + 10/90% band) vs the ground truth,
  * the RCRPS score (via task.evaluate), and whether persistence-fallback was used.
Optionally saves a forecast plot per example.

This is turnkey: it injects the CiK deps (pylibs) and the repo onto sys.path and
routes the big caches off /home, so you can just run it. Runs on CPU (fp32 upcast)
or GPU (bf16) automatically.

--------------------------------------------------------------------------------
RUN (CPU, from anywhere):
    conda activate opentslm
    python /home/s223540177/FoundationalTS/cik_inference_demo.py --n-tasks 3 --context on

  # a specific task, context off, more samples:
    python cik_inference_demo.py --task ElectricityIncreaseInPredictionTask \
        --context off --n-samples 50

  # list available task names:
    python cik_inference_demo.py --list

Plots default to  /home/s223540177/FoundationalTS/cik_demo_plots/<Task>/forecast.png
Checkpoint defaults to the phase-1 model (best CiK forecaster: RCRPS 0.300 no-ctx).
--------------------------------------------------------------------------------
"""
import os
import sys
import argparse

# ── make the CiK repo + its deps + FoundationalTS importable, big caches off /home ──
_U = os.environ.get("USER", "s223540177")
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = "/scratch/%s/cik_work/context-is-key-forecasting" % _U
for _p in ("/scratch/%s/cik_work/pylibs" % _U, _REPO):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("FTS_ROOT", _HERE)
os.environ.setdefault("HF_HOME", "/tmp/%s_hf_cikdemo" % _U)
os.environ.setdefault("CIK_RESULT_CACHE", "/scratch/%s/cik_work/demo_cache" % _U)
os.environ.setdefault("CIK_METRIC_SCALING_CACHE", "/scratch/%s/cik_work/metric_scaling_cache" % _U)
os.environ.setdefault("CIK_DATA_STORE", "/scratch/%s/cik_work/data" % _U)
os.environ.setdefault("CIK_MODEL_STORE", "/scratch/%s/cik_work/models" % _U)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

# Default checkpoints (phase-1 = best CiK forecaster; phase-2 = CoT-only).
_P1 = "/scratch/%s/FoundationalTS/pretrained_weights/joint_phase1_extend_balanced_lora_nofreq/checkpoints/best_model.pt" % _U
_P2 = "/scratch/%s/FoundationalTS/results_harsleep_balanced/sincos_patch32_mlp/gemma_3_270m/OpenTSLMSPFixPos2D/joint_phase2_downstream_from_pretrained_nofreq/checkpoints/best_model.pt" % _U
# Local gemma snapshot avoids any network model fetch (offline-safe).
_SNAP = "/home/%s/.cache/huggingface/hub/models--google--gemma-3-270m/snapshots/9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1" % _U
# Plots live in the project tree by default.
_PLOTS = os.path.join(_HERE, "cik_demo_plots")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=_P1, help="OpenTSLM best_model.pt (default: phase-1 extbal)")
    ap.add_argument("--phase2", action="store_true", help="shortcut: use the phase-2 balanced checkpoint")
    ap.add_argument("--llm-id", default=_SNAP, help="gemma id/path (default: local snapshot)")
    ap.add_argument("--context", choices=["on", "off"], default="on", help="inject CiK context text")
    ap.add_argument("--task", default=None, help="a specific CiK task class name (see --list)")
    ap.add_argument("--n-tasks", type=int, default=3, help="if --task unset, run the first N tasks")
    ap.add_argument("--seed", type=int, default=1, help="task instance seed")
    ap.add_argument("--n-samples", type=int, default=25, help="probabilistic samples to draw")
    ap.add_argument("--show-steps", type=int, default=8, help="how many forecast timesteps to print")
    ap.add_argument("--plot-dir", default=_PLOTS,
                    help="where to save a forecast plot per example (default: <repo>/cik_demo_plots); pass '' to disable")
    ap.add_argument("--list", action="store_true", help="list all CiK task names and exit")
    args = ap.parse_args()
    if args.phase2:
        args.ckpt = _P2

    from cik_benchmark import ALL_TASKS
    by_name = {t.__name__: t for t in ALL_TASKS}

    if args.list:
        print("Available CiK tasks (%d):" % len(ALL_TASKS))
        for n in sorted(by_name):
            print("  " + n)
        return

    if args.task:
        if args.task not in by_name:
            sys.exit("Unknown task %r. Use --list to see the %d available names." % (args.task, len(ALL_TASKS)))
        task_classes = [by_name[args.task]]
    else:
        task_classes = list(ALL_TASKS)[: max(1, args.n_tasks)]

    from cik_benchmark.baselines.opentslm import OpenTSLMForecaster
    forecaster = OpenTSLMForecaster(
        ckpt_path=args.ckpt,
        llm_id=args.llm_id,
        use_context=(args.context == "on"),
        label="demo",
    )
    print("=" * 78)
    print("checkpoint : %s" % args.ckpt)
    print("context    : %s   |   samples: %d   |   device: %s"
          % (args.context.upper(), args.n_samples, forecaster.device))
    print("=" * 78)

    if args.plot_dir:
        os.makedirs(args.plot_dir, exist_ok=True)
        from cik_benchmark.evaluation import plot_forecast_univariate

    scores = []
    for tcls in task_classes:
        task = tcls(seed=args.seed)
        hist = task.past_time.iloc[:, -1].values.astype(float)
        gt = task.future_time.iloc[:, -1].values.astype(float)

        # ---- inference (identical to the benchmark path) ----
        samples, info = forecaster(task_instance=task, n_samples=args.n_samples)
        pts = samples[:, :, 0]                                  # [n_samples, pred_len]
        med = np.median(pts, axis=0)
        q10, q90 = np.quantile(pts, 0.10, axis=0), np.quantile(pts, 0.90, axis=0)
        ev = task.evaluate(samples)
        rcrps = ev["metric"] if isinstance(ev, dict) else ev
        scores.append(float(rcrps))

        # ---- report ----
        print("\n" + "-" * 78)
        print("TASK: %s   (seed=%d)" % (tcls.__name__, args.seed))
        print("-" * 78)
        for attr in ("background", "scenario", "constraints"):
            v = getattr(task, attr, None)
            if v:
                print("  %-11s: %s" % (attr, str(v).strip().replace("\n", " ")[:300]))
        print("  history    : %d steps, last 5 = %s" % (len(hist), np.round(hist[-5:], 3)))
        print("  horizon    : %d steps" % len(gt))
        k = min(args.show_steps, len(gt))
        print("  %-6s %-12s %-12s %-12s %-12s" % ("t", "median", "[q10", "q90]", "ground_truth"))
        for i in range(k):
            print("  %-6d %-12.4f %-12.4f %-12.4f %-12.4f" % (i, med[i], q10[i], q90[i], gt[i]))
        mae = float(np.mean(np.abs(med[: len(gt)] - gt)))
        print("  --> RCRPS = %.4f   (median MAE = %.4f, fallback=%s)"
              % (rcrps, mae, info.get("used_fallback")))

        if args.plot_dir:
            from pathlib import Path
            pdir = Path(args.plot_dir) / tcls.__name__
            pdir.mkdir(parents=True, exist_ok=True)
            try:
                plot_forecast_univariate(task=task, samples=samples, path=pdir)
                print("  plot       : %s/forecast.png" % pdir)
            except Exception as e:
                print("  plot       : skipped (%s)" % e)

    print("\n" + "=" * 78)
    print("mean RCRPS over %d task(s): %.4f" % (len(scores), sum(scores) / max(1, len(scores))))
    print("=" * 78)


if __name__ == "__main__":
    main()
