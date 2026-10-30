"""TimeMQA (Time-MQA/TSQA) Open-Ended-QA subset as an OpenTSLM QADataset.

Multi-domain (Healthcare/Nature/Web/Energy/Traffic) free-text time-series QA
(~37.6k). Each CSV row has application_domain, task_type, question_format, QA_list,
where QA_list is a JSON-ish string:  "question": "... [ts list] ...", "answer": "free text".

We parse question+answer, extract the embedded numeric time series into a separate
`time_series` input (z-normalized), replace it in the prompt with a placeholder, and
keep the free-text answer as the generation target. Eval is FREE-TEXT (ROUGE-L /
exact-match), NOT label accuracy. Single CSV → deterministic 90/5/5 split.
"""
import ast
import json
import re
from typing import List, Literal, Tuple

import torch

from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.time_series_datasets.QADataset import QADataset

_TS_RE = re.compile(r"\[([0-9eE:.,\s+\-]+)\]")


def _parse_qa(qa: str) -> Tuple[str, str]:
    """Return (question, answer) from the QA_list JSON-ish string."""
    try:
        d = json.loads("{" + qa + "}")
        return str(d.get("question", "")), str(d.get("answer", ""))
    except Exception:
        mq = re.search(r'"question"\s*:\s*"(.*?)"\s*,\s*"answer"', qa, re.S)
        ma = re.search(r'"answer"\s*:\s*"(.*)"\s*$', qa.strip(), re.S)
        q = mq.group(1) if mq else qa
        a = ma.group(1) if ma else ""
        return q, a


_MCQ_LEAD = re.compile(r"^\s*\(?([A-Da-d])[)\].:]\s*")
# (pattern, mode). Ordered most-specific-first. Each locates an option letter
# that IS present but is not in leading position.
#   drop  -- the whole match is boilerplate ("Correct answer: B)"); discard it
#   keep1 -- group(2) is real content ("Yes" in "Yes (a).");  keep it
_MCQ_RECOVER = [
    (re.compile(p, re.I), mode) for p, mode in (
        (r"^\s*(?:the\s+)?correct\s+answer\s+is[:\s]*\(?([a-d])[)\].:]\s*", "drop"),
        (r"^\s*(?:the\s+)?answer\s+is[:\s]*\(?([a-d])[)\].:]\s*", "drop"),
        (r"^\s*(?:correct\s+)?answer[:\s]*\(?([a-d])[)\].:]\s*", "drop"),
        (r"^\s*option\s+\(?([a-d])\)?[\s.,:]*", "drop"),
        (r"^\s*(yes|no|true|false)[,\s]*\(([a-d])\)[\s.,]*", "keep1"),
        (r"^\s*\(([a-d])\)\s*", "drop"),
        (r"^([^.]{0,40}?)\(([a-d])\)[\s.,]*", "keep1"),
    )
]


def normalize_mcq_answer(answer: str) -> str:
    """Rewrite a multiple-choice answer so it LEADS with its option letter.

    83.7% of TimeMQA MCQ answers already lead with "A) ...". The rest bury the
    letter -- "Correct answer: B) 675...", "Yes (a). There is...", "Answer: B)
    90.0..." -- so a model that imitates the targets faithfully learns to omit
    the letter on ~1 item in 6, and any letter-based scorer marks those wrong
    however good the reasoning was. Measured on our own model: 15.8% of MCQ
    predictions carried no extractable letter, against 0.6-2.1% for every other
    model evaluated, and in 22.9% of those the correct option's text was present
    verbatim in the prediction.

    This moves the letter to the front and leaves the rest of the answer intact.
    The 4.6% of answers that never reference an option at all are returned
    unchanged -- assigning them a letter would mean guessing which option the
    free text corresponds to.
    """
    a = (answer or "").strip()
    if not a or _MCQ_LEAD.match(a):
        return a
    for pat, mode in _MCQ_RECOVER:
        m = pat.match(a)
        if not m:
            continue
        if mode == "keep1":
            letter, kept = m.group(2), (m.group(1) or "").strip()
            rest = a[m.end():].strip()
            body = f"{kept}. {rest}".strip() if kept else rest
        else:
            letter, body = m.group(1), a[m.end():].strip()
        body = body.lstrip(" .,:;-")
        if not body:            # the letter was the entire answer; nothing to keep
            return f"{letter.upper()})"
        return f"{letter.upper()}) {body}"
    return a


def _extract_ts(question: str) -> Tuple[List[float], str]:
    """Pull the first numeric [...] list out of the question; return (ts, prompt_wo_ts)."""
    m = _TS_RE.search(question)
    if not m:
        return [], question
    try:
        vals = [float(x) for x in m.group(1).split(",") if x.strip() != ""]
    except Exception:
        return [], question
    prompt = question[: m.start()] + "<the provided time series>" + question[m.end():]
    return vals, prompt


class TimeMQAOpenQADataset(QADataset):
    def __init__(self, split: Literal["train", "test", "validation"], EOS_TOKEN: str,
                 format_sample_str: bool = False, time_series_format_function=None):
        super().__init__(split, EOS_TOKEN, format_sample_str, time_series_format_function)

    def _load_splits(self) -> Tuple[list, list, list]:
        import os
        import pandas as pd
        # OFF by default and deliberately so: it changes both training targets
        # and eval golds, so enabling it mid-flight would score models trained on
        # raw targets against normalised golds. Turn it on for a *fresh* training
        # round (train and eval must agree), not for runs already underway.
        norm_mcq = os.environ.get("TIMEMQA_NORMALIZE_MCQ", "0") == "1"
        # Optional single-format restriction, used ONLY by the Qwen3 curriculum
        # baseline, which trains MCQ and true/false as two separate sequential
        # stages. Unset (the default) keeps every format, so every existing run
        # and every joint row is byte-identical to before.
        #   TIMEMQA_FORMAT=mcq -> question_format == "multiple_choice"  (11283 rows)
        #   TIMEMQA_FORMAT=tf  -> question_format in {"true/false","True/False"} (6918)
        _fmt = os.environ.get("TIMEMQA_FORMAT", "").strip().lower()
        _KEEP = {"mcq": {"multiple_choice"}, "tf": {"true/false"}}
        if _fmt and _fmt not in _KEEP:
            raise ValueError(f"TIMEMQA_FORMAT={_fmt!r} not in {sorted(_KEEP)}")
        # Time-MQA is not auto-downloaded. Read the CSV path from
        # TIMEMQA_OPEN_ENDED_CSV, else look under the shared raw-data root.
        csv_path = os.environ.get("TIMEMQA_OPEN_ENDED_CSV", "").strip()
        if not csv_path:
            from opentslm.time_series_datasets.constants import RAW_DATA
            csv_path = os.path.join(RAW_DATA, "timemqa", "open_ended_QA.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"Time-MQA open-ended QA csv not found at {csv_path!r}. Download "
                "Open_Ended_QA/open_ended_QA.csv from the Time-MQA release "
                "(HF dataset Time-MQA/TSQA) and either place it at "
                "$LOTUS_DATA_DIR/timemqa/open_ended_QA.csv or export "
                "TIMEMQA_OPEN_ENDED_CSV=/path/to/open_ended_QA.csv"
            )
        df = pd.read_csv(csv_path)
        if _fmt:
            _n0 = len(df)
            df = df[df["question_format"].astype(str).str.strip().str.lower()
                    .isin(_KEEP[_fmt])]
            print(f"[TimeMQA] TIMEMQA_FORMAT={_fmt}: kept {len(df)}/{_n0} rows", flush=True)
        rows = []
        n_norm = 0
        for _, r in df.iterrows():
            q, a = _parse_qa(str(r["QA_list"]))
            ts, prompt = _extract_ts(q)
            if not ts or not a.strip():
                continue
            a = a.strip()
            if norm_mcq and str(r.get("question_format", "")).strip().lower() == "multiple_choice":
                a2 = normalize_mcq_answer(a)
                n_norm += (a2 != a)
                a = a2
            rows.append({"domain": r.get("application_domain", ""),
                         "prompt": prompt, "answer_text": a, "ts": ts})
        if norm_mcq:
            print(f"[TimeMQA] TIMEMQA_NORMALIZE_MCQ=1: rewrote {n_norm} MCQ answers "
                  f"to lead with their option letter", flush=True)
        import random
        random.Random(1234).shuffle(rows)
        n = len(rows)
        return rows[: int(0.90 * n)], rows[int(0.90 * n): int(0.95 * n)], rows[int(0.95 * n):]

    def _get_answer(self, row) -> str:
        return row["answer_text"]

    def _get_pre_prompt(self, row) -> str:
        return ("You are given a time series from the " + str(row["domain"]) +
                " domain. Answer the following question.\n" + row["prompt"])

    def _get_post_prompt(self, _row) -> str:
        return "\nAnswer:"

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        series = torch.tensor(row["ts"], dtype=torch.float32).flatten()
        mean = series.mean()
        std = series.std()
        # A length-1 series makes std() return nan (Bessel correction, dof=0) and
        # an empty one makes mean() nan. `nan < 1e-8` is False, so a bare
        # magnitude check lets nan through: the series normalises to nan, the
        # sample yields nan loss, and nan gradients then wipe out the whole
        # model (all ranks, since gradients are averaged). Test finiteness first.
        if not torch.isfinite(mean):
            mean = torch.tensor(0.0)
        if not torch.isfinite(std) or std < 1e-8:
            std = torch.tensor(1.0)
        series_norm = (series - mean) / std
        text = f"This is the time series, mean {mean.item():.4f} std {std.item():.4f}."
        return [TextTimeSeriesPrompt(text, series_norm.tolist())]

    @staticmethod
    def get_labels() -> List[str]:
        return []  # open-ended QA — no fixed label set (eval via ROUGE-L)
