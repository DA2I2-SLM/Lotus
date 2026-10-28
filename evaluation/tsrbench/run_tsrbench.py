#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Deakin University and the LOTUS authors
# SPDX-License-Identifier: MIT
"""Zero-shot TSRBench evaluation for a LOTUS Phase-2 checkpoint.

TSRBench (Yu et al., arXiv:2601.18744; data: https://huggingface.co/datasets/
umd-zhou-lab/TSRBench; code: https://github.com/tianyi-lab/TSRBench) ships 12
jsonl files = 15 multiple-choice tasks. This adapter

  1. loads the benchmark jsonl files from --data_dir,
  2. builds the benchmark's OWN prompts (the ones its time-series-embedding
     pipeline uses: inference/embed_opensource/inference.py in the TSRBench
     repo -- standard template, the temporal-relation "Time Stamps" filter and
     the abductive basketball template),
  3. routes every series through our encoder at the <ts>...</ts> placeholders,
     one z-scored, captioned soft-token block per channel exactly as in
     Phase 2 ("'<name>', it has mean X and std Y:"),
  4. generates greedily, and
  5. scores with a letter extractor that mirrors the benchmark's released
     scorer (inference/embed_opensource/evaluate.py), and optionally exports
     the generations in the layout that scorer reads, so the "unmodified
     scorer" protocol can be followed verbatim.

Nothing here is trained; the checkpoint is used as-is.

Usage (see evaluation/tsrbench/README.md):
    python evaluation/tsrbench/run_tsrbench.py \
        --ckpt .../joint_phase2_downstream_from_pretrained_freq/checkpoints/best_model.pt \
        --llm_id google/gemma-3-270m --use_freq_embed \
        --data_dir /path/to/TSRBench \
        --out_dir results/tsrbench --export_for_official_scorer

    # Build and inspect the prompts without a model / GPU:
    python evaluation/tsrbench/run_tsrbench.py --dry_run --data_dir ... --out_dir ... --limit 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "evaluation" / "cot"))  # load_model / setup_device

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark layout
# ─────────────────────────────────────────────────────────────────────────────
# file stem -> (category dir in the HF snapshot, name the released scorer uses)
# The scorer (inference/embed_opensource/evaluate.py) hard-codes three names
# that differ from the file names on the Hub: math_reasoning, event_forecast,
# pattern_decision. --export_for_official_scorer writes under the scorer names.
SUBSETS: Dict[str, Tuple[str, str]] = {
    "perception":                  ("perception", "perception"),
    "etiological_reasoning":       ("reasoning",  "etiological_reasoning"),
    "causal_reasoning":            ("reasoning",  "causal_reasoning"),
    "abductive_reasoning":         ("reasoning",  "abductive_reasoning"),
    "temporal_relation_reasoning": ("reasoning",  "temporal_relation_reasoning"),
    "numerical_reasoning":         ("reasoning",  "math_reasoning"),
    "deductive_reasoning":         ("reasoning",  "deductive_reasoning"),
    "inductive_reasoning":         ("reasoning",  "inductive_reasoning"),
    "time_series_forecasting":     ("prediction", "time_series_forecasting"),
    "event_prediction":            ("prediction", "event_forecast"),
    "qualitative_decision":        ("decision",   "pattern_decision"),
    "quantitative_decision":       ("decision",   "quantitative_decision"),
}
_ALIASES = {alias: stem for stem, (_, alias) in SUBSETS.items()}

# Perception is one file with a `category` field holding four sub-tasks.
PERCEPTION_CATEGORIES = {
    "PR": "Pattern Recognition",
    "NU": "Noise Understanding",
    "AD": "Anolmaly Detection",   # sic -- the released data spells it this way
    "CA": "Similarity Analysis",
}

# Paper column -> (file stem, perception category or None). The paper's table
# has 14 columns; deductive_reasoning (DR in the TSRBench paper) is not among
# them, so it is reported separately and excluded from `overall_paper14`.
PAPER_COLUMNS: Dict[str, Tuple[str, Optional[str]]] = {
    "PR":      ("perception", "Pattern Recognition"),
    "NU":      ("perception", "Noise Understanding"),
    "AD":      ("perception", "Anolmaly Detection"),
    "CA":      ("perception", "Similarity Analysis"),
    "ER":      ("etiological_reasoning", None),
    "CD":      ("causal_reasoning", None),
    "AR":      ("abductive_reasoning", None),
    "TR":      ("temporal_relation_reasoning", None),
    "NR":      ("numerical_reasoning", None),
    "IR":      ("inductive_reasoning", None),
    "TSF":     ("time_series_forecasting", None),
    "EP":      ("event_prediction", None),
    "QualDM":  ("qualitative_decision", None),
    "QuantDM": ("quantitative_decision", None),
}
EXTRA_COLUMNS = {"DR": ("deductive_reasoning", None)}

OPTION_LABELS = ["A", "B", "C", "D", "E", "F", "G"]
TS_PLACEHOLDER = "<ts><ts/>"   # the benchmark's own placeholder string

# Verbatim from TSRBench inference/embed_opensource/inference.py::_answer_instruction
ANSWER_FORMAT = (
    "Output your reasoning process in <think> tags and final answer "
    "as a single letter in <answer> tags. Format:\n"
    "<think>Your reasoning here (less than 2048 tokens)</think>\n"
    "<answer>A</answer>"
)
ABDUCTIVE_PREAMBLE = (
    "You are an expert in basketball game analysis. Your task is to perform abductive reasoning.\n"
    "Given a sequence of past events, future events, and corresponding time series data from a game, "
    "determine the most plausible event that occurred in between to link them.\n\n"
)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def resolve_subsets(spec: str) -> List[str]:
    if spec.strip().lower() in ("", "all"):
        return list(SUBSETS)
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        stem = _ALIASES.get(tok, tok)
        if stem not in SUBSETS:
            raise SystemExit(f"unknown subset {tok!r}; known: {', '.join(SUBSETS)} "
                             f"(aliases: {', '.join(_ALIASES)})")
        if stem not in out:
            out.append(stem)
    return out


def find_subset_file(data_dir: str, stem: str) -> str:
    """Accept the HF snapshot layout (<category>/<stem>.jsonl), a flat layout
    (<stem>.jsonl, which is what the released scorer reads) and the nested
    layout used by the repo's text launcher (<category>/<stem>/<stem>.jsonl)."""
    cat, alias = SUBSETS[stem]
    cands = []
    for name in (stem, alias):
        cands += [
            os.path.join(data_dir, cat, f"{name}.jsonl"),
            os.path.join(data_dir, f"{name}.jsonl"),
            os.path.join(data_dir, cat, name, f"{name}.jsonl"),
        ]
    for c in cands:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        f"no jsonl for subset {stem!r} under {data_dir}; tried:\n  " + "\n  ".join(cands))


def read_jsonl(path: str) -> List[dict]:
    """Same enumeration as the released scorer: blank lines skipped, every other
    line must parse, and `idx` is the position in the resulting list."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: malformed JSON ({e})") from e
    return rows


def maybe_download(data_dir: str, repo_id: str) -> None:
    """Mirror of the repo's download_data.py but pointed at the public Hub id."""
    from huggingface_hub import snapshot_download
    print(f"[tsrbench] snapshot_download({repo_id!r}) -> {data_dir}")
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=data_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction (mirrors TSRBench inference/embed_opensource/inference.py)
# ─────────────────────────────────────────────────────────────────────────────
def format_choices(choices: Any) -> str:
    """Verbatim semantics of the benchmark's format_choices(): dict -> 'K. v'
    in sorted key order; list -> A..G in stored order; anything else -> str()."""
    if isinstance(choices, dict):
        return "\n".join(f"{k}. {v}" for k, v in sorted(choices.items()))
    if isinstance(choices, list):
        if len(choices) > len(OPTION_LABELS):
            raise ValueError(f"{len(choices)} choices but only {len(OPTION_LABELS)} labels")
        return "\n".join(f"{OPTION_LABELS[i]}. {v}" for i, v in enumerate(choices))
    return str(choices)


def answer_instruction(choice_text: str) -> str:
    # benchmark: " Select from the options below:\n" + choice_text + "\n" + ANSWER_FORMAT
    return "Select from the options below:\n" + choice_text + "\n" + ANSWER_FORMAT


def _names_for(row: dict, n_series: int) -> List[str]:
    names = list(row.get("name_of_series") or [])
    # benchmark default when the field is missing: "Series {i+1}". We also use
    # it when the list is shorter than `timeseries` (time_series_forecasting
    # ships one name for four series).
    names = [str(n) for n in names[:n_series]]
    names += [f"Series {i + 1}" for i in range(len(names), n_series)]
    return names


def _is_numeric_series(s: Sequence[Any]) -> bool:
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in s)


class Built:
    """One prompt: pre text, [(name, raw series)], post text, gold letter."""
    __slots__ = ("pre", "blocks", "post", "gold", "choice_text", "meta", "caption_style")

    def __init__(self, pre, blocks, post, gold, choice_text, meta=None, caption_style="named"):
        self.pre, self.blocks, self.post, self.gold = pre, blocks, post, gold
        self.choice_text, self.meta = choice_text, (meta or {})
        self.caption_style = caption_style


def build_standard(row: dict) -> Built:
    ts = row["timeseries"]
    names = _names_for(row, len(ts))
    choice_text = format_choices(row["choices"])
    pre = row["question"] + " Here are the time series"
    post = ". " + answer_instruction(choice_text)
    return Built(pre, list(zip(names, ts)), post, str(row["answer"]), choice_text)


def build_temporal(row: dict) -> Built:
    """temporal_relation_reasoning: drop the 'Time Stamps' channel (it holds
    ISO date strings, not numbers) before emitting placeholders -- exactly what
    the benchmark's build_prompt_temporal does."""
    ts = row.get("timeseries", [])
    names = _names_for(row, len(ts))
    kept = [(n, s) for n, s in zip(names, ts)
            if not (isinstance(n, str) and n.strip().lower() == "time stamps")]
    dropped_nonnum = [n for n, s in kept if not _is_numeric_series(s)]
    kept = [(n, s) for n, s in kept if _is_numeric_series(s)]
    if not kept:
        raise ValueError("temporal_relation_reasoning row has no numeric channel left")
    choice_text = format_choices(row["choices"])
    pre = row["question"] + " Here are the time series"
    post = ". " + answer_instruction(choice_text)
    meta = {"dropped_channels": ["Time Stamps"] + dropped_nonnum}
    return Built(pre, kept, post, str(row["answer"]), choice_text, meta)


def build_abductive(row: dict) -> Built:
    """abductive_reasoning: basketball play-by-play; verbatim port of the
    benchmark's build_prompt_abductive (window of 10 steps either side of the
    critical moment, win-probability table in the text, two series)."""
    ctx = row["context"]
    h_ev, h_t = ctx["history_events"], ctx["history_times"]
    f_ev, f_t = ctx["future_events"], ctx["future_times"]
    tsd = row["numerical_time_series"]
    all_t = h_t + f_t
    a = tsd["wp_Team A"]["history"] + tsd["wp_Team A"]["future"]
    b = tsd["wp_Team B"]["history"] + tsd["wp_Team B"]["future"]
    n = min(len(all_t), len(a), len(b))
    all_t, a, b = all_t[:n], a[:n], b[:n]

    window = 10
    crit = len(h_t)
    lo, hi = max(0, crit - window), min(n, crit + window)
    all_t, a, b = all_t[lo:hi], a[lo:hi], b[lo:hi]

    h_ev = h_ev[max(0, len(h_ev) - window):]
    h_t = h_t[max(0, len(h_t) - window):]
    f_ev, f_t = f_ev[:window], f_t[:window]

    past = "Past Events (History):\n" + "".join(f"- {t}: {e}\n" for t, e in zip(h_t, h_ev))
    fut = "Future Events:\n" + "".join(f"- {t}: {e}\n" for t, e in zip(f_t, f_ev))
    table = ("\nTime Series Data (Win Probability):\n"
             "Time | Team A Win Prob | Team B Win Prob\n" + "-" * 60 + "\n"
             + "".join(f"{t} | {a[i]:.3f} | {b[i]:.3f}\n" for i, t in enumerate(all_t)))

    mcq = row["multiple_choice_question"]
    choice_text = format_choices(mcq["choices"])
    pre = (ABDUCTIVE_PREAMBLE
           + "--- CONTEXT ---\n"
           + f"{past.strip()}\n"
           + "\n... [A CRITICAL EVENT HAPPENED HERE] ...\n\n"
           + f"{fut.strip()}\n\n"
           + f"{table.strip()}\n\n"
           + "--- TASK ---\n"
           + f"{mcq['question']}"
           + " Here are the time series")
    post = ". " + answer_instruction(choice_text)
    blocks = [("Team A Win Probability", a), ("Team B Win Probability", b)]
    return Built(pre, blocks, post, str(mcq["answer"]), choice_text,
                 {"window": [lo, hi]})


_INLINE_OPT_RE = re.compile(r"^\s*([A-G])[.)]\s*(.*?)\s*$", re.M)


def build_perception(row: dict, mode: str) -> Built:
    """perception.jsonl differs from every other file: no `choices`, no
    `name_of_series`; the question already contains one `<ts><ts/>` per series
    and the options inline ("A. ...") plus "You must start response from A) or
    B) ...". The benchmark's embedding-pipeline builder indexes row["choices"]
    and would raise on it, so there is no released reference for this file in
    the embedding modality (see README "Caveats").

    mode="inplace": inject the series at the question's own placeholders and
        append the shared <think>/<answer> format instruction.
    mode="append":  remove the placeholder markers, re-list the inline options
        and use the standard template with names "Series {i+1}".
    """
    ts = row["timeseries"]
    q = row["question"]
    gold = str(row["answer"])
    inline = _INLINE_OPT_RE.findall(q)
    choice_text = "\n".join(f"{k}. {v}" for k, v in inline)
    n_ph = q.count(TS_PLACEHOLDER)
    if mode == "inplace" and n_ph == len(ts) and n_ph > 0:
        parts = q.split(TS_PLACEHOLDER)
        pre = parts[0]
        # text that precedes block i (for i > 0) is carried by the block itself
        blocks = [(parts[i] if i > 0 else "", ts[i]) for i in range(len(ts))]
        post = parts[-1] + "\n" + ANSWER_FORMAT
        return Built(pre, blocks, post, gold, choice_text,
                     {"perception_mode": "inplace", "category": row.get("category")},
                     caption_style="inplace")
    # append mode (also the fallback when placeholder count != series count)
    q_clean = re.sub(r"\s*" + re.escape(TS_PLACEHOLDER) + r"\s*", " ", q).strip()
    names = [f"Series {i + 1}" for i in range(len(ts))]
    pre = q_clean + " Here are the time series"
    post = ". " + answer_instruction(choice_text)
    return Built(pre, list(zip(names, ts)), post, gold, choice_text,
                 {"perception_mode": "append", "category": row.get("category"),
                  "placeholders_in_question": n_ph})


def build_prompt(stem: str, row: dict, perception_mode: str) -> Built:
    if stem == "abductive_reasoning":
        return build_abductive(row)
    if stem == "temporal_relation_reasoning":
        return build_temporal(row)
    if stem == "perception":
        return build_perception(row, perception_mode)
    return build_standard(row)


# ─────────────────────────────────────────────────────────────────────────────
# Series -> our interface (one captioned, z-scored block per channel)
# ─────────────────────────────────────────────────────────────────────────────
def zscore(series: Sequence[float]) -> Tuple[List[float], float, float, int]:
    """Per-channel z-score with the Phase-2 conventions: torch-style unbiased
    std, std floored at 1e-8 (OpenTSLMSPFixPos2D._safe_std), non-finite -> 0."""
    x = np.asarray(series, dtype=np.float64).reshape(-1)
    n_bad = int((~np.isfinite(x)).sum())
    if n_bad:
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mean = float(x.mean()) if x.size else 0.0
    std = float(x.std(ddof=1)) if x.size > 1 else 0.0
    if not math.isfinite(std):
        std = 0.0
    std = max(std, 1e-8)
    return ((x - mean) / std).astype(np.float32).tolist(), mean, std, n_bad


def to_sample(b: Built) -> Tuple[dict, str, dict]:
    """Return (sample dict for model.generate, flat prompt text, stats)."""
    texts, series, stats, n_bad = [], [], [], 0
    for i, (name, raw) in enumerate(b.blocks):
        z, m, s, bad = zscore(raw)
        n_bad += bad
        if b.caption_style == "inplace":
            # `name` carries the question text that precedes this placeholder
            # ("" for the first block, whose preceding text is pre_prompt)
            lead = name if i > 0 else b.pre
            sep = "" if (not lead or lead.endswith((" ", "\n"))) else " "
            cap = f"{name}{sep}it has mean {m:.4f} and std {s:.4f}:"
        else:
            # renders as  … 'name_1', it has …:<ts>…</ts>. 'name_2', it has …:<ts>…</ts>. Select …
            cap = ("" if i == 0 else ".") + f" '{name}', it has mean {m:.4f} and std {s:.4f}:"
        texts.append(cap)
        series.append(z)
        stats.append((m, s))
    sample = {
        "type": "tsrbench",
        "pre_prompt": b.pre,
        "time_series_text": texts,
        "time_series": series,
        "post_prompt": b.post,
        "answer": "",
        "n_channels": len(series),
        "_n_channels": len(series),
        "ts_already_normalized": True,
        "time_series_norm_stats": stats,
    }
    # Human-readable rendering of what the LLM sees. <ts>/</ts> are inserted by
    # pad_and_apply_batch around each encoded block.
    flat = b.pre + "".join(f"{t}<ts>[{len(z)} values]</ts>" for t, z in zip(texts, series)) + b.post
    return sample, flat, {"n_series": len(series), "lengths": [len(z) for z in series],
                          "n_nonfinite_values": n_bad}


# ─────────────────────────────────────────────────────────────────────────────
# Scoring (mirror of TSRBench inference/embed_opensource/evaluate.py)
# ─────────────────────────────────────────────────────────────────────────────
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def extract_answer(response: str) -> Optional[str]:
    """Released scorer: `<answer>...</answer>` first; else eval() the response as
    a dict with an "answer" key. We use json.loads for that fallback instead of
    eval() -- same accepted outputs, no code execution."""
    if not response:
        return None
    m = _ANSWER_TAG_RE.search(response)
    if m:
        return m.group(1).strip()
    try:
        obj = json.loads(response)
        if isinstance(obj, dict) and "answer" in obj:
            return str(obj["answer"]).strip()
    except Exception:
        pass
    return None


def normalise_answer(ans: str) -> str:
    for ch in (",", "$", " ", "*", "(", ")"):
        ans = ans.replace(ch, "")
    return ans.strip()


def score_mcq(label: str, response: str) -> Tuple[Optional[float], Optional[str]]:
    """(score, letter): score None = unparseable (released scorer drops it),
    0.0/1.0 otherwise; letter = first char of the normalised answer."""
    ans = extract_answer(response)
    if ans is None:
        return None, None
    ans = normalise_answer(ans)
    if not ans:
        return 0.0, ""
    return (1.0 if ans[0].lower() == str(label).lower() else 0.0), ans[0].upper()


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
ARCH_KEYS = ("pe_type", "pe_base_t", "pe_base_c", "patch_size", "freq_mode", "enc_dec_type",
             "mlp_enc_hidden_dim", "mlp_enc_ff_dim", "mlp_dec_hidden_dim", "mlp_dec_ff_dim",
             "trans_enc_dim", "trans_enc_num_layers", "trans_enc_num_heads", "trans_enc_ff_dim",
             "trans_enc_output_dim", "trans_dec_dim", "trans_dec_num_layers",
             "trans_dec_num_heads", "trans_dec_ff_dim")


def add_model_args(ap: argparse.ArgumentParser) -> None:
    # Same flags/defaults as evaluation/cot/eval_cot_phase2.py so EVAL_EXTRA
    # strings used for the CoT eval work unchanged here.
    ap.add_argument("--lora_r", type=int, default=4)
    ap.add_argument("--lora_alpha", type=int, default=8)
    ap.add_argument("--use_freq_embed", action="store_true")
    ap.add_argument("--model_class", default="fixpos2d",
                    choices=["fixpos2d", "official_sp", "official_fl"])
    ap.add_argument("--enc_dec_type", default="mlp", choices=["mlp", "transformer"])
    ap.add_argument("--patch_size", type=int, default=32)
    ap.add_argument("--pe_type", default="sincos")
    ap.add_argument("--pe_base_t", type=float, default=None)
    ap.add_argument("--pe_base_c", type=float, default=None)
    ap.add_argument("--freq_mode", default="linear", choices=["linear", "spectral"])
    ap.add_argument("--mlp_enc_hidden_dim", type=int, default=64)
    ap.add_argument("--mlp_enc_ff_dim", type=int, default=256)
    ap.add_argument("--mlp_dec_hidden_dim", type=int, default=64)
    ap.add_argument("--mlp_dec_ff_dim", type=int, default=256)
    ap.add_argument("--trans_enc_dim", type=int, default=None)
    ap.add_argument("--trans_enc_num_layers", type=int, default=None)
    ap.add_argument("--trans_enc_num_heads", type=int, default=None)
    ap.add_argument("--trans_enc_ff_dim", type=int, default=None)
    ap.add_argument("--trans_enc_output_dim", type=int, default=None)
    ap.add_argument("--trans_dec_dim", type=int, default=None)
    ap.add_argument("--trans_dec_num_layers", type=int, default=None)
    ap.add_argument("--trans_dec_num_heads", type=int, default=None)
    ap.add_argument("--trans_dec_ff_dim", type=int, default=None)


def build_model(args: argparse.Namespace):
    import torch
    from eval_cot_phase2 import load_model, setup_device  # evaluation/cot
    device = args.device or setup_device()
    print(f"Device: {device}")
    model = load_model(args.ckpt, args.llm_id, device,
                       lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                       use_freq=args.use_freq_embed, model_class=args.model_class,
                       arch={k: getattr(args, k) for k in ARCH_KEYS})
    model.eval()
    return model, torch


def generate_batch(model, torch, rows: List[dict], max_new_tokens: int) -> List[str]:
    try:
        outs, _ = model.generate(rows, max_new_tokens=max_new_tokens)
        return [o if o is not None else "" for o in outs]
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        outs = []
        for r in rows:
            try:
                o, _ = model.generate([r], max_new_tokens=max_new_tokens)
                outs.append(o[0] if o else "")
            except Exception as e:  # noqa: BLE001
                torch.cuda.empty_cache()
                outs.append(f"<error: {type(e).__name__}>")
        return outs


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
def _acc(scores: List[Optional[float]]) -> dict:
    parsed = [s for s in scores if s is not None]
    n, n_parsed, correct = len(scores), len(parsed), int(sum(parsed))
    return {
        "n": n,
        "n_parsed": n_parsed,
        "n_unparsed": n - n_parsed,
        "correct": correct,
        # released scorer: unparseable responses are dropped from the denominator
        "accuracy_scorer": (correct / n_parsed) if n_parsed else None,
        # strict: unparseable counted as wrong
        "accuracy_strict": (correct / n) if n else None,
    }


def run_subset(stem: str, rows: List[dict], args, model, torch, out_path: str) -> List[dict]:
    done: Dict[int, dict] = {}
    if args.resume and os.path.isfile(out_path):
        for rec in read_jsonl(out_path):
            done[int(rec["idx"])] = rec
        if done:
            print(f"  resume: {len(done)} records already in {out_path}")

    todo = [i for i in range(len(rows)) if i not in done]
    if args.limit > 0:
        todo = [i for i in todo if i < args.limit]
        done = {i: r for i, r in done.items() if i < args.limit}

    records = dict(done)
    mode = "a" if done else "w"
    t0 = time.time()
    with open(out_path, mode, encoding="utf-8") as fout:
        for start in range(0, len(todo), args.batch_size):
            idxs = todo[start:start + args.batch_size]
            built, samples, flats, stats, errors = [], [], [], [], {}
            for i in idxs:
                try:
                    b = build_prompt(stem, rows[i], args.perception_mode)
                    s, flat, st = to_sample(b)
                except Exception as e:  # noqa: BLE001
                    errors[i] = f"{type(e).__name__}: {e}"
                    b, s, flat, st = None, None, "", {}
                built.append(b); samples.append(s); flats.append(flat); stats.append(st)

            ok = [j for j, s in enumerate(samples) if s is not None]
            if args.dry_run or not ok:
                preds = [""] * len(idxs)
            else:
                outs = generate_batch(model, torch, [samples[j] for j in ok], args.max_new_tokens)
                preds = [""] * len(idxs)
                for j, o in zip(ok, outs):
                    preds[j] = o

            for j, i in enumerate(idxs):
                b = built[j]
                gold = b.gold if b is not None else None
                score, letter = (score_mcq(gold, preds[j]) if (b is not None and not args.dry_run)
                                 else (None, None))
                rec = {
                    "id": f"{stem}:{i}",
                    "idx": i,
                    "subset": stem,
                    "category": rows[i].get("category"),
                    "prompt_text": flats[j],
                    "prediction": preds[j],
                    "gold": gold,
                    "pred_letter": letter,
                    "score": score,
                    "n_series": stats[j].get("n_series"),
                    "series_lengths": stats[j].get("lengths"),
                    "series_names": ([n for n, _ in b.blocks] if b.caption_style != "inplace"
                                     else [f"question placeholder {k + 1}" for k in range(len(b.blocks))])
                                    if b is not None else None,
                    "meta": (b.meta if b is not None else {}),
                }
                if stats[j].get("n_nonfinite_values"):
                    rec["meta"]["n_nonfinite_values"] = stats[j]["n_nonfinite_values"]
                if i in errors:
                    rec["error"] = errors[i]
                records[i] = rec
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_done = start + len(idxs)
            if n_done % max(50, args.batch_size * 10) == 0 or n_done == len(todo):
                el = time.time() - t0
                print(f"  [{stem}] {n_done}/{len(todo)}  ({el / max(n_done, 1):.1f}s/sample)", flush=True)
    return [records[i] for i in sorted(records)]


def summarise(all_records: Dict[str, List[dict]], args) -> dict:
    per_file, per_column = {}, {}
    for stem, recs in all_records.items():
        per_file[stem] = _acc([r["score"] for r in recs])
        per_file[stem]["scorer_name"] = SUBSETS[stem][1]
        if stem == "perception":
            per_file[stem]["by_category"] = {
                cat: _acc([r["score"] for r in recs if r.get("category") == cat])
                for cat in sorted({r.get("category") for r in recs if r.get("category")})
            }
    for col, (stem, cat) in {**PAPER_COLUMNS, **EXTRA_COLUMNS}.items():
        if stem not in all_records:
            continue
        recs = all_records[stem]
        if cat is not None:
            recs = [r for r in recs if r.get("category") == cat]
        per_column[col] = _acc([r["score"] for r in recs])
        per_column[col]["source"] = f"{stem}" + (f" / category={cat!r}" if cat else "")
        per_column[col]["in_paper_table"] = col in PAPER_COLUMNS

    def micro(stems):
        scores = [r["score"] for s in stems if s in all_records for r in all_records[s]]
        return _acc(scores)

    return {
        "checkpoint": args.ckpt,
        "llm_id": args.llm_id,
        "max_new_tokens": args.max_new_tokens,
        "perception_mode": args.perception_mode,
        "limit": args.limit,
        "dry_run": args.dry_run,
        "scoring": {
            "accuracy_scorer": "correct / parsed  (released scorer drops responses with no "
                               "recoverable letter; micro-average over subsets)",
            "accuracy_strict": "correct / all     (unparseable counted as incorrect)",
        },
        "paper_columns": per_column,
        "per_file": per_file,
        # micro over the 14 paper subsets = all files except deductive_reasoning
        "overall_paper14": micro([s for s in SUBSETS if s != "deductive_reasoning"]),
        # micro over everything evaluated (what the released scorer prints as Overall)
        "overall_all": micro(list(SUBSETS)),
    }


def export_for_official_scorer(all_records: Dict[str, List[dict]], data_files: Dict[str, str],
                               out_dir: str, model_name: str) -> str:
    """Write <out_dir>/official_scorer/{results,dataset} in the layout that
    TSRBench's inference/embed_opensource/evaluate.py reads:
        results/<scorer_name>_<model>/generated_answer_1_0.json  [{idx, response, ...}]
        dataset/<scorer_name>.jsonl                               (link to the data)
    """
    root = os.path.join(out_dir, "official_scorer")
    res_root, ds_root = os.path.join(root, "results"), os.path.join(root, "dataset")
    os.makedirs(res_root, exist_ok=True); os.makedirs(ds_root, exist_ok=True)
    for stem, recs in all_records.items():
        alias = SUBSETS[stem][1]
        d = os.path.join(res_root, f"{alias}_{model_name}")
        os.makedirs(d, exist_ok=True)
        payload = []
        for r in recs:
            item = {"idx": r["idx"], "question_text": r["prompt_text"],
                    "response": r["prediction"], "num_tokens": None}
            if r.get("pred_letter"):
                item["answer"] = r["pred_letter"]
            payload.append(item)
        with open(os.path.join(d, "generated_answer_1_0.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        link = os.path.join(ds_root, f"{alias}.jsonl")
        src = os.path.abspath(data_files[stem])
        if os.path.lexists(link):
            os.remove(link)
        try:
            os.symlink(src, link)
        except OSError:
            shutil.copyfile(src, link)
    return root


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None, help="Phase-2 best_model.pt (required unless --dry_run)")
    ap.add_argument("--llm_id", default="google/gemma-3-270m")
    ap.add_argument("--data_dir", required=True,
                    help="TSRBench root: <category>/<subset>.jsonl as on the Hub, or flat <subset>.jsonl")
    ap.add_argument("--download", action="store_true",
                    help="snapshot_download --hf_repo into --data_dir first (needs huggingface_hub)")
    ap.add_argument("--hf_repo", default="umd-zhou-lab/TSRBench")
    ap.add_argument("--subsets", default="all",
                    help="comma list of file stems (or scorer aliases); default: all 12 files")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=2048,
                    help="the benchmark prompt asks for < 2048 reasoning tokens")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="first N rows per subset (debug); 0 = all")
    ap.add_argument("--device", default=None, help="cuda | cpu | mps (default: auto)")
    ap.add_argument("--perception_mode", default="inplace", choices=["inplace", "append"],
                    help="how perception.jsonl (own <ts><ts/> placeholders, no `choices`) is "
                         "prompted; see README")
    ap.add_argument("--resume", action="store_true", help="skip rows already in <out_dir>/<subset>.jsonl")
    ap.add_argument("--dry_run", action="store_true",
                    help="build prompts and write records with empty predictions; no model loaded")
    ap.add_argument("--export_for_official_scorer", action="store_true")
    ap.add_argument("--model_name", default="lotus", help="model tag used in the exported result dirs")
    add_model_args(ap)
    args = ap.parse_args()

    if args.download:
        maybe_download(args.data_dir, args.hf_repo)
    if not args.dry_run and not args.ckpt:
        raise SystemExit("--ckpt is required unless --dry_run")

    stems = resolve_subsets(args.subsets)
    data_files = {s: find_subset_file(args.data_dir, s) for s in stems}
    os.makedirs(args.out_dir, exist_ok=True)

    model = torch = None
    if not args.dry_run:
        model, torch = build_model(args)

    all_records: Dict[str, List[dict]] = {}
    for stem in stems:
        rows = read_jsonl(data_files[stem])
        print(f"\n===== {stem}: {len(rows)} rows ({data_files[stem]}) =====", flush=True)
        out_path = os.path.join(args.out_dir, f"{stem}.jsonl")
        recs = run_subset(stem, rows, args, model, torch, out_path)
        all_records[stem] = recs
        a = _acc([r["score"] for r in recs])
        n_err = sum(1 for r in recs if r.get("error"))
        print(f"  {stem}: acc(scorer)={a['accuracy_scorer']}  acc(strict)={a['accuracy_strict']}  "
              f"parsed={a['n_parsed']}/{a['n']}  build_errors={n_err}")

    summary = summarise(all_records, args)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\nPaper columns (accuracy_scorer / accuracy_strict, n_parsed/n):")
    for col, m in summary["paper_columns"].items():
        fmt = lambda v: "   -  " if v is None else f"{100 * v:6.2f}"
        print(f"  {col:8s} {fmt(m['accuracy_scorer'])} / {fmt(m['accuracy_strict'])}   "
              f"{m['n_parsed']}/{m['n']}" + ("" if m["in_paper_table"] else "   (not in paper table)"))
    o = summary["overall_paper14"]
    print(f"  overall(14) {100 * o['accuracy_scorer'] if o['accuracy_scorer'] is not None else float('nan'):.2f}"
          f" / {100 * o['accuracy_strict'] if o['accuracy_strict'] is not None else float('nan'):.2f}")
    print(f"summary -> {os.path.join(args.out_dir, 'summary.json')}")

    if args.export_for_official_scorer:
        root = export_for_official_scorer(all_records, data_files, args.out_dir, args.model_name)
        print(f"official-scorer export -> {root}\n"
              f"  python <TSRBench>/inference/embed_opensource/evaluate.py "
              f"--results_dir {root}/results --dataset_dir {root}/dataset --model_name {args.model_name}")


if __name__ == "__main__":
    main()
