<!--
SPDX-FileCopyrightText: 2026 Deakin University and the LOTUS authors
SPDX-License-Identifier: MIT
-->
# TSRBench zero-shot evaluation

Thin adapter that runs a LOTUS Phase-2 checkpoint on
[TSRBench](https://tsrbench.github.io/) (Yu et al., *TSRBench: A Comprehensive
Multi-task Multi-modal Time Series Reasoning Benchmark for Generalist Models*,
[arXiv:2601.18744](https://arxiv.org/abs/2601.18744)) without any training, using
the benchmark's own prompts and its released scorer.

Files:

| path | role |
|---|---|
| `evaluation/tsrbench/run_tsrbench.py` | loads the jsonl files, builds the benchmark prompts, encodes the series through our encoder, generates greedily, scores, optionally exports for the official scorer |
| `bashfiles/evaluate/eval_tsrbench.sh` | SLURM launcher (same conventions as `eval_phase2_cot.sh`) |

Everything below about the benchmark was read from these sources on 2026-10-28:

* code: <https://github.com/tianyi-lab/TSRBench> (files `download_data.py`,
  `evaluation/evaluate.py`, `inference/embed_opensource/{README.md,inference.py,evaluate.py,run.sh}`,
  `inference/text_opensource/*`)
* data: <https://huggingface.co/datasets/umd-zhou-lab/TSRBench> (file tree via
  `https://huggingface.co/api/datasets/umd-zhou-lab/TSRBench/tree/main?recursive=true`,
  and the jsonl files themselves)
* project page: <https://tsrbench.github.io/>

## 1. Getting the data

The benchmark is a 127 MB Hugging Face dataset repo with 12 jsonl files in four
category directories (no `data/` parquet shards; `load_dataset(...)` as printed on
the dataset card does **not** match the file tree, use a snapshot instead):

```bash
# needs: pip install huggingface_hub
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="umd-zhou-lab/TSRBench", repo_type="dataset",
                  local_dir="/path/to/TSRBench")
EOF
# or: python evaluation/tsrbench/run_tsrbench.py --download --data_dir /path/to/TSRBench ...
```

Resulting layout (sizes from the Hub file listing):

```
TSRBench/
├── perception/perception.jsonl                    2.7 MB   700 rows, field `category` ∈ {Pattern Recognition, Noise Understanding, Anolmaly Detection [sic], Similarity Analysis}
├── reasoning/etiological_reasoning.jsonl          1.3 MB   350 rows
├── reasoning/causal_reasoning.jsonl              19.5 MB
├── reasoning/abductive_reasoning.jsonl            5.4 MB   150 rows
├── reasoning/temporal_relation_reasoning.jsonl    2.1 MB   160 rows
├── reasoning/numerical_reasoning.jsonl            0.6 MB   400 rows
├── reasoning/deductive_reasoning.jsonl            2.6 MB   250 rows
├── reasoning/inductive_reasoning.jsonl            0.2 MB   100 rows
├── prediction/time_series_forecasting.jsonl       7.8 MB
├── prediction/event_prediction.jsonl              0.5 MB   360 rows
├── decision/qualitative_decision.jsonl           83.2 MB
└── decision/quantitative_decision.jsonl           0.9 MB   300 rows
```

(Row counts were taken by reading the files; the three largest were only
partially downloaded for inspection, so their counts are omitted here. The
TSRBench repo's `inference/embed_opensource/README.md` lists 300 / 720 / 335 rows
for causal / TSF / qualitative-decision and 4,125 in total.)

Note: the repo's own `download_data.py` calls
`snapshot_download(repo_id="ParadiseYu/TSRBench", ...)`. On 2026-10-28 that id
returned HTTP 401 from the Hub API (private or gone); `umd-zhou-lab/TSRBench`
is the id the README, project page and dataset card link to.

### Record format (as found in the files)

All files except `perception.jsonl` and `abductive_reasoning.jsonl`:

| field | type | notes |
|---|---|---|
| `question` | str | task text; no placeholders |
| `timeseries` | list[list[float]] | one inner list per channel |
| `name_of_series` | list[str] | one name per channel (exceptions below) |
| `choices` | list[str] **or** dict{letter: value} | list → labelled A… in stored order; dict → sorted keys (A–D); values can be numbers or whole lists (deductive reasoning) |
| `answer` | str | one letter |
| `domain`, `type`, `task` | str | metadata; `task` is absent on 50/400 numerical-reasoning rows |

Exceptions seen in the data:

* `perception.jsonl`: **no `choices`, no `name_of_series`**. The `question`
  already contains one `<ts><ts/>` placeholder per channel (1 or 2) and the
  options inline (`A. …`), and ends with "You must start response from A) or B) or C) or D)".
* `abductive_reasoning.jsonl`: nested record (`context.history_events/history_times/future_events/future_times`,
  `multiple_choice_question.{question,choices,answer}`, `numerical_time_series.{"wp_Team A","wp_Team B",…}.{history,future}`).
* `temporal_relation_reasoning.jsonl`: `name_of_series == ["Time Stamps", "GoldSteinScale"]`
  and the "Time Stamps" channel holds ISO date **strings**.
* `time_series_forecasting.jsonl`: four channels but a single name (`["Closing Price"]`).
* `numerical_reasoning.jsonl`: 50 rows have no `name_of_series`.

## 2. Prompt protocol

The benchmark ships several inference pipelines. The one for models that take
time series as embeddings (the setting LOTUS is in) is
`inference/embed_opensource/inference.py`; its prompt builders are what this
adapter ports:

* **standard** (9 files):
  `{question} Here are the time series '{name_1}': <ts><ts/>. '{name_2}': <ts><ts/>. … Select from the options below:\nA. …\nB. …\n\nOutput your reasoning process in <think> tags and final answer as a single letter in <answer> tags. Format:\n<think>Your reasoning here (less than 2048 tokens)</think>\n<answer>A</answer>`
  (`format_choices`: list → `A.`–`G.`; dict → `key. value` in sorted key order.
  Missing names → `Series {i+1}`.)
* **temporal_relation_reasoning**: same, after dropping any channel named
  "Time Stamps" (case-insensitive).
* **abductive_reasoning**: the basketball preamble, a `--- CONTEXT ---` block
  (last 10 past events, `[A CRITICAL EVENT HAPPENED HERE]`, first 10 future
  events, a `Time | Team A Win Prob | Team B Win Prob` table over the 10 steps
  either side of the critical moment), `--- TASK ---`, the question, then the
  two win-probability windows as series named "Team A Win Probability" /
  "Team B Win Probability", then the shared answer instruction.

In our interface a prompt is `pre_prompt`, a list of `(caption, series)` blocks
and `post_prompt`; `OpenTSLMSPFixPos2D.pad_and_apply_batch` wraps each encoded
block in `<ts> … </ts>` itself. Each channel is z-scored (unbiased std, floored
at 1e-8 like the model's `_safe_std`) and captioned with the Phase-2 wording,
so the model sees

```
{question} Here are the time series 'Humidity (%)', it has mean 65.7917 and std 13.7997:<ts>…</ts>. 'Air Pressure (hPa)', it has mean 1034.2500 and std 2.3637:<ts>…</ts>. … Select from the options below:
A. Rain
B. No Rain
Output your reasoning process in <think> tags …
```

The exact text per item (with `<ts>[N values]</ts>` standing in for the soft
tokens) is written to `prompt_text` in every output record, and
`--dry_run` writes those records without loading a model.

Decoding is greedy (`model.generate`, no sampling), `--max_new_tokens 2048`
by default because the benchmark prompt itself asks for < 2048 reasoning tokens.

## 3. Running

```bash
# SLURM (defaults mirror eval_phase2_cot.sh; TSRBENCH_DIR is required)
TSRBENCH_DIR=/path/to/TSRBench \
EVAL_CKPT=/path/to/joint_phase2_downstream_from_pretrained_freq/checkpoints/best_model.pt \
sbatch bashfiles/evaluate/eval_tsrbench.sh

# direct
python evaluation/tsrbench/run_tsrbench.py \
  --ckpt /path/to/best_model.pt --llm_id google/gemma-3-270m --use_freq_embed \
  --data_dir /path/to/TSRBench --out_dir results/tsrbench \
  --subsets all --max_new_tokens 2048 --export_for_official_scorer

# quick smoke test: 3 rows per subset
python evaluation/tsrbench/run_tsrbench.py --ckpt … --data_dir … --out_dir /tmp/t --limit 3
```

`--subsets` takes file stems (`perception,causal_reasoning,…`) or the scorer's
aliases (`math_reasoning`, `event_forecast`, `pattern_decision`). Model-geometry
flags are the same as `evaluation/cot/eval_cot_phase2.py` (so the `EVAL_EXTRA`
string for the 1B transformer-encoder run works unchanged). `--resume` continues
an interrupted run from the per-subset jsonl files.

Outputs in `--out_dir`:

* `<subset>.jsonl` — one record per row: `id`, `idx`, `subset`, `category`,
  `prompt_text`, `prediction`, `gold`, `pred_letter`, `score` (1/0, or `null`
  when no letter could be recovered), `n_series`, `series_lengths`,
  `series_names`, `meta`.
* `summary.json` — `paper_columns` (14 paper columns + `DR`), `per_file`
  (with `by_category` for perception), `overall_paper14`, `overall_all`. Every
  entry carries both `accuracy_scorer` (correct / parsed; what the released
  scorer computes) and `accuracy_strict` (correct / all; unparseable = wrong).
* `official_scorer/` — when `--export_for_official_scorer` is set, see below.

## 4. Scoring with the released scorer

The TSRBench repo has two scorers:

* `evaluation/evaluate.py` — `eval()`s the response as a JSON dict and reads
  `["answer"]`; it is the scorer for the API/vLLM text pipelines, which request
  JSON output. It cannot read `<answer>` tags.
* `inference/embed_opensource/evaluate.py` — reads `<answer>(.*?)</answer>`
  first (then falls back to the JSON dict), drops responses with no recoverable
  answer, scores `answer[0].lower() == label.lower()` after stripping
  `, $ space * ( )`, prints per-task accuracy with a per-`category` breakdown
  for perception, and reports *Overall Accuracy* = Σ correct / Σ parsed over
  the tasks (micro-average). This is the scorer whose behaviour matches the
  `<think>/<answer>` prompt, and the one `--export_for_official_scorer` targets.

It expects `results/<task>_<model>/generated_answer*.json` (a list of
`{"idx": int, "response": str, …}`) and `dataset/<task>.jsonl`, with the task
names it hard-codes (`math_reasoning`, `event_forecast`, `pattern_decision`
instead of the Hub file names `numerical_reasoning`, `event_prediction`,
`qualitative_decision`). The export writes exactly that:

```
<out_dir>/official_scorer/
├── results/<scorer_task>_<model_name>/generated_answer_1_0.json
└── dataset/<scorer_task>.jsonl        -> symlink to the file in --data_dir
```

then

```bash
git clone https://github.com/tianyi-lab/TSRBench
python TSRBench/inference/embed_opensource/evaluate.py \
  --results_dir <out_dir>/official_scorer/results \
  --dataset_dir <out_dir>/official_scorer/dataset \
  --model_name lotus
```

Note the released scorer's JSON fallback uses Python `eval()` on model output;
our in-repo mirror (`run_tsrbench.py::extract_answer`) uses `json.loads` for
that branch and is otherwise identical, so `accuracy_scorer` in `summary.json`
should equal the official numbers.

## 5. Subset names ↔ paper columns

| paper column | Hub file (`--subsets` stem) | scorer name | selection |
|---|---|---|---|
| PR | `perception` | `perception` | `category == "Pattern Recognition"` |
| NU | `perception` | `perception` | `category == "Noise Understanding"` |
| AD | `perception` | `perception` | `category == "Anolmaly Detection"` (spelling as in the data) |
| CA | `perception` | `perception` | `category == "Similarity Analysis"` |
| ER | `etiological_reasoning` | same | |
| CD | `causal_reasoning` | same | |
| AR | `abductive_reasoning` | same | |
| TR | `temporal_relation_reasoning` | same | |
| NR | `numerical_reasoning` | `math_reasoning` | |
| IR | `inductive_reasoning` | same | |
| TSF | `time_series_forecasting` | same | |
| EP | `event_prediction` | `event_forecast` | |
| QualDM | `qualitative_decision` | `pattern_decision` | |
| QuantDM | `quantitative_decision` | same | |
| — (`DR` in `summary.json`) | `deductive_reasoning` | same | not a column in the paper's table |

TSRBench itself counts 15 tasks (the 4 perception categories + 11 files); the
paper's table has 14 columns and omits deductive reasoning. `summary.json`
therefore reports `overall_paper14` (all files except `deductive_reasoning`)
next to `overall_all`.

## 6. Caveats — things to check before trusting a number

1. **Perception prompt is our choice.** `perception.jsonl` has no `choices`
   or `name_of_series`, and the benchmark's embedding-pipeline builder
   (`build_prompt_standard`) indexes `sample["choices"]`, so on the public
   file it raises and the row is skipped; there is no released reference for
   how a time-series-embedding model is prompted on this file. Default
   `--perception_mode inplace` injects each series at the question's own
   `<ts><ts/>` placeholder (as the benchmark's *text* pipeline does) and
   appends the shared `<think>/<answer>` format instruction after the
   question's own option list. `--perception_mode append` removes the
   placeholder markers and uses the standard template with names
   `Series {i+1}` and the inline options re-listed. If the paper's numbers
   were produced with a different perception prompt, change `build_perception`.
2. **CA = "Similarity Analysis" is inferred.** The TSRBench paper abbreviates
   that category SA and the released scorer calls it `ComA`; "CA" is our
   paper's label for the only remaining perception category.
3. **Caption wording.** `'{name}', it has mean X and std Y:` follows the
   Phase-2 CoT datasets (`HARCoTQADataset` etc.). The benchmark's own
   template has `'{name}': <ts>`; the extra statistics caption is our
   interface's convention, not the benchmark's.
4. **Series/name mismatches** (TSF: 4 channels, 1 name; 50 numerical-reasoning
   rows without names) are filled with `Series {i+1}`, the benchmark's own
   default for a missing name list. What the four TSF channels are is not
   documented in the data.
5. **Unparseable responses.** The released scorer drops them from the
   denominator; `summary.json` also reports the strict rate. Check
   `n_unparsed` — with a 270M model a large gap between the two is possible.
6. **Text length.** `pad_and_apply_batch` tokenises each text segment with
   `truncation=True, max_length=4096`; the longest `question + choices` seen
   is ~10 k characters (time-series forecasting), which should fit, but
   nothing here guards against truncation. Soft tokens: one per 32-sample
   patch per channel (qualitative decision = 12 × 1000 samples → 384 tokens).
7. **Row counts** for `causal_reasoning`, `time_series_forecasting` and
   `qualitative_decision` were not verified locally (only the first 3 MB of
   each was inspected); the repo's README figures are quoted above.
8. **Dataset license**: none is shown on the dataset card and the GitHub repo
   has no LICENSE file (GitHub API `license: null`). Do not redistribute the
   data from this repo; download it from the Hub.
9. The `--download` helper and the `huggingface_hub` dependency are optional;
   nothing else here needs packages beyond what Phase-2 evaluation already uses.
