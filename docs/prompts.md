<!--
SPDX-FileCopyrightText: 2026 Deakin University and the LOTUS authors

SPDX-License-Identifier: MIT
-->

# Prompt templates

The exact prompts used in the paper (Appendix "Prompt Templates"). Every Phase-2 corpus
uses the same three-part schema: a **pre-prompt** stating the task and instructions, one
**captioned block per channel**, and a **post-prompt** fixing the answer format. In LOTUS
and the soft-prompt baselines each block holds the encoder's soft tokens, delimited by
`<ts>…</ts>`; in the text-tokenised baseline the same position holds the decimated values
as digits. Each caption names the channel and gives its mean and standard deviation, so the
normalisation statistics reach every interface as text.

The templates are built in `src/opentslm/time_series_datasets/Phase2DownstreamDataset.py`
and the per-corpus dataset classes under `src/opentslm/time_series_datasets/<task>/`.
Ellipses below mark truncation only; everything else is verbatim.

## Phase 1

### Forecasting (electricity demand and the synthetic corpus)

```
[pre-prompt]      (empty)
[channel block]   This is a univariate time series with 1 channel and 1024 time steps.
                  It has mean 6084.2410 and std 1174.1908. <ts>…</ts>
[post-prompt]     <eos>
```

A multivariate item from the synthetic corpus takes the same form with `multivariate`, its
channel count, and one block per channel. The answer field is the end-of-sequence token:
the training signal is the next-patch regression, not any token in this prompt. A
forecasting batch therefore produces no cross-entropy term, which is why each Phase-1 batch
is drawn from a single source.

### Language corpora

- **Time-series QA (TSQA)**: the item's question in the pre-prompt, `Predict the {task} Answer:`
  in the post-prompt, the answer letter as the target.
- **M4 captioning**: `You are an expert in time series analysis.` followed by
  `Please generate a detailed caption for this time-series, describing it as accurately as possible.`,
  with the reference caption as the target.

## Phase 2

### HAR

```
[pre-prompt]
You are given accelerometer data in all three dimensions. Your task is to classify the activity based on analysis of the data.

Instructions:
- Begin by analyzing the time series without assuming a specific label.
- Think step-by-step about what the observed patterns suggest regarding movement intensity and behavior.
- Write your rationale as a single, natural paragraph — do not use bullet points, numbered steps, or section headings.
- Do **not** mention any class label until the final sentence.

Possible activity labels are:
biking, lying, running, sitting, standing, walking, walking_down, walking_up.

- Make sure that your last word is the answer. You MUST end your response with "Answer: "

[channel blocks, one per axis]
The following is the accelerometer data on the x-axis, it has mean -3.2434 and std 0.0474: <ts>…</ts>
The following is the accelerometer data on the y-axis, it has mean 2.3132 and std 0.0550: <ts>…</ts>
The following is the accelerometer data on the z-axis, it has mean 9.2017 and std 0.0639: <ts>…</ts>

[post-prompt]
Rationale:
```

### PAMAP2

Identical to HAR except for the eighteen-way label set:

```
Possible activity labels are:
lying, sitting, standing, walking, running, cycling, nordic walking, watching TV, computer work, car driving, ascending stairs, descending stairs, vacuum cleaning, ironing, folding laundry, house cleaning, playing soccer, rope jumping.
```

### Sleep-EDF

```
[pre-prompt]
You are given a 30-second EEG time series segment. Your task is to classify the sleep stage based on analysis of the data.

Instructions:
- Analyze the data objectively without presuming a particular label.
- Reason carefully and methodically about what the signal patterns suggest regarding sleep stage.
- Write your reasoning as a single, coherent paragraph. Do not use bullet points, lists, or section headers.
- Only reveal the correct class at the very end.
- Never state that you are uncertain or unable to classify the data. You must always provide a rationale and a final answer.

[channel block]
This is the EEG signal, it has mean 0.0021 and std 0.9873: <ts>…</ts>

[post-prompt]
Possible sleep stages are:
Wake, Non-REM stage 1, Non-REM stage 2, Non-REM stage 3, Non-REM stage 4, REM sleep

- Please now write your rationale. Make sure that your last word is the answer. You MUST end your response with "Answer: "
```

Unlike the accelerometer corpora, the class list is placed in the post-prompt.

### ECG-QA

```
[pre-prompt]
You are an expert cardiologist analyzing an ECG (electrocardiogram).
Clinical Context: 82-year-old male patient. 12-lead ECG. clinical recording. recorded with AT-6 C 5.8. Signal quality: baseline drift noted, static noise present, burst noise present, electrode artifacts present. extra beats detected during recording. pacemaker present.
Your task is to examine the ECG signal and answer the following medical question:
Question: Does this ECG show burst noise?

Instructions:
- Begin by analyzing the time series without assuming a specific answer.
- Think step-by-step about what the observed patterns suggest regarding the cardiac condition.
- Write your rationale as a single, natural paragraph — do not use bullet points, numbered steps, or section headings.
- Do **not** mention any final answer until the very end.
- Consider the ECG morphology, intervals, and any abnormalities that relate to the question.

[channel blocks, one per lead, twelve in total]
This is ECG Lead I, it has mean -0.0063 and std 0.1198: <ts>…</ts>
… (Leads II, III, aVR, aVL, aVF, V1–V6)

[post-prompt]
Based on your analysis of the ECG data, select your answer from the following options:
no, yes
- Make sure that your last word is the answer. You MUST end your response with "Answer: "
```

The clinical-context paragraph is assembled from the record's metadata; the answer set in
the post-prompt is that of the question template.

### TimeMQA

```
[pre-prompt]
You are given a time series from the Healthcare domain. Answer the following question.
Which data segment shows more volatility? A) <the provided time series> or B) [0.71, 0.72, 0.73, 0.73, 0.73, 0.73, 0.72, 0.72, 0.71, 0.7, 0.68, 0.67]

[channel block]
This is the time series, mean 0.5300 std 0.1352. <ts>…</ts>

[post-prompt]
Answer:
```

The corpus is open-ended, so the post-prompt is a bare `Answer:`. In comparison questions
the first segment enters through the time-series interface while the second is inlined as
digits, so the model compares a soft-token series against a textual one.

## TSRBench (zero-shot)

TSRBench is evaluated zero-shot, so the prompts are the benchmark's own rather than ours.
All fourteen subsets are multiple choice and share one answer instruction; only the way
the signal is introduced differs between the three templates. Series enter at the
`<ts>…</ts>` placeholders exactly as in Phase 2. Parsing takes the single letter inside
`<answer>`; an item with no recoverable letter is scored incorrect. The runner is
`evaluation/tsrbench/run_tsrbench.py`.

### Standard multiple choice (twelve of the fourteen subsets)

```
[question and channel blocks]
{question} Here are the time series '{name_1}': <ts>…</ts>. '{name_2}': <ts>…</ts>. …

[answer instruction, shared by all subsets]
Select from the options below:
A. {choice A}
B. {choice B}
…
Output your reasoning process in <think> tags and final answer as a single letter in <answer> tags. Format:
<think>Your reasoning here (less than 2048 tokens)</think>
<answer>A</answer>
```

Options are labelled `A`–`G` in the order the benchmark stores them.

### Temporal relation reasoning

As above, but any channel named "Time Stamps" is dropped before the placeholders are
emitted, so only measured channels are encoded. The answer instruction is unchanged.

### Abductive reasoning

```
[pre-prompt]
You are an expert in basketball game analysis. Your task is to perform abductive reasoning.
Given a sequence of past events, future events, and corresponding time series data from a game, determine the most plausible event that occurred in between to link them.

[context]
--- CONTEXT ---
Past Events (History):
- {time}: {event} …

… [A CRITICAL EVENT HAPPENED HERE] …

Future Events:
- {time}: {event} …

Time Series Data (Win Probability):
Time | Team A Win Prob | Team B Win Prob
…

[task, channel blocks, then the shared answer instruction]
--- TASK ---
{question} Here are the time series 'Team A Win Probability': <ts>…</ts>. 'Team B Win Probability': <ts>…</ts>.
```

A window of ten steps either side of the critical moment is kept. The win-probability
channels are given twice, once serialised in the table and once through the encoder. This
is the benchmark's own design and the one setting where the interface receives the signal
redundantly.
