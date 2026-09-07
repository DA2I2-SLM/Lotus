#!/usr/bin/env python3
"""
Build finetune_comparison.xlsx comparing our fine-tuned models vs paper baselines.
Loads results from JSON files written by finetune_eval.py.

Paper setting (Time-MoE Table 4):
  H=96  → ctx=512
  H=192 → ctx=1024
  H=336 → ctx=2048
  H=720 → ctx=3072
"""
import json, os
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

RESULTS_DIR     = "output_job/finetune_results"
FULLFT_DIR      = "output_job/fullft_results"
EVAL_1B_DIR     = "output_job/eval_1b_results"
EVAL_1B_R8_DIR  = "output_job/eval_1b_r8_results"
EVAL_1B_R8V2_DIR= "output_job/eval_1b_r8v2_results"
OUT             = "results/finetune_comparison.xlsx"

HORIZONS  = [96, 192, 336, 720]
DATASETS  = ["ETTh1", "ETTh2", "Weather"]   # display names
CTX_MATCH = {96: 512, 192: 1024, 336: 2048, 720: 3072}  # paper standard

# ── Load JSON results ─────────────────────────────────────────────────────────

def load_json(tag):
    """Return dict[ds][ctx][H] = (mse, mae)  with None for missing entries."""
    path = os.path.join(RESULTS_DIR, f"ft_results_{tag}.json")
    rows = json.load(open(path))
    d = {}
    for r in rows:
        ds  = r["dataset"].capitalize() if r["dataset"].lower() == "weather" else r["dataset"]
        ctx = r["context_length"]
        H   = r["horizon"]
        d.setdefault(ds, {}).setdefault(ctx, {})[H] = (r["mse"], r["mae"])
    return d

def build_our_data(raw, ctx_map):
    """Build {ds: {H: (mse, mae)}} using ctx_map to select the context per horizon."""
    out = {}
    for ds in DATASETS:
        out[ds] = {}
        for H in HORIZONS:
            ctx = ctx_map(H)
            out[ds][H] = raw.get(ds, {}).get(ctx, {}).get(H, (None, None))
    return out

raw_baseline     = load_json("baseline")
raw_ts_only_lora = load_json("ts_only_lora")
raw_ts_lang_lora = load_json("ts_lang_lora")

# Standard setting: matched ctx per horizon (same as paper)
OUR_STANDARD = {
    "Baseline\n(no LoRA)":    build_our_data(raw_baseline,     lambda H: CTX_MATCH[H]),
    "TS-only LoRA\n(r=4)":    build_our_data(raw_ts_only_lora, lambda H: CTX_MATCH[H]),
    "TS+Lang LoRA\n(r=4)":    build_our_data(raw_ts_lang_lora, lambda H: CTX_MATCH[H]),
}

# Fixed ctx=512 for all horizons
OUR_CTX512 = {
    "Baseline\n(no LoRA)":    build_our_data(raw_baseline,     lambda H: 512),
    "TS-only LoRA\n(r=4)":    build_our_data(raw_ts_only_lora, lambda H: 512),
    "TS+Lang LoRA\n(r=4)":    build_our_data(raw_ts_lang_lora, lambda H: 512),
}

# Full fine-tuning results (all at ctx=512)
def load_fullft(model_tag):
    """Merge per-dataset JSONs from fullft_results into same format as load_json."""
    combined = {}
    for ds_raw, ds_disp in [("ETTh1","ETTh1"),("ETTh2","ETTh2"),("weather","Weather")]:
        path = os.path.join(FULLFT_DIR, f"ft_fullft_{model_tag}_{ds_raw}.json")
        rows = json.load(open(path))
        combined.setdefault(ds_disp, {}).setdefault(512, {})
        for r in rows:
            combined[ds_disp][512][r["horizon"]] = (r["mse"], r["mae"])
    return combined

raw_fullft_baseline     = load_fullft("baseline")
raw_fullft_ts_only_lora = load_fullft("ts_only_lora")
raw_fullft_ts_lang_lora = load_fullft("ts_lang_lora")

# ── 1B model results ──────────────────────────────────────────────────────────
def load_1b(mode):
    """Load 1B eval results for mode in ('zeroshot', 'finetune').
    Returns {ds: {ctx: {H: (mse, mae)}}} same format as load_json."""
    d = {}
    for ds_raw, ds_disp in [("ETTh1","ETTh1"), ("ETTh2","ETTh2"), ("weather","Weather")]:
        path = os.path.join(EVAL_1B_DIR, f"eval_1b_{mode}_{ds_raw}.json")
        rows = json.load(open(path))
        for r in rows:
            ctx = r["context_length"]; H = r["horizon"]
            d.setdefault(ds_disp, {}).setdefault(ctx, {})[H] = (r["mse"], r["mae"])
    return d

raw_1b_zeroshot = load_1b("zeroshot")
raw_1b_finetune = load_1b("finetune")

OUR_1B = {
    "1B Zero-shot":      build_our_data(raw_1b_zeroshot, lambda H: CTX_MATCH[H]),
    "1B Fine-tuned\n(LoRA FT)": build_our_data(raw_1b_finetune, lambda H: CTX_MATCH[H]),
}

# ── 1B LoRA r=8 results (job 3462055, epoch-2 snapshot) ──────────────────────
def load_1b_r8(mode, dirpath=None, pre="r8_"):
    """Same shape as load_1b, for a LoRA r=8 pretrained checkpoint snapshot."""
    dirpath = dirpath or EVAL_1B_R8_DIR
    d = {}
    for ds_raw, ds_disp in [("ETTh1","ETTh1"), ("ETTh2","ETTh2"), ("weather","Weather")]:
        path = os.path.join(dirpath, f"eval_1b_{pre}{mode}_{ds_raw}.json")
        rows = json.load(open(path))
        for r in rows:
            ctx = r["context_length"]; H = r["horizon"]
            d.setdefault(ds_disp, {}).setdefault(ctx, {})[H] = (r["mse"], r["mae"])
    return d

raw_1b_r8_zeroshot = load_1b_r8("zeroshot")
raw_1b_r8_finetune = load_1b_r8("finetune")
raw_1b_r8v2_zeroshot = load_1b_r8("zeroshot", EVAL_1B_R8V2_DIR, "r8v2_")
raw_1b_r8v2_finetune = load_1b_r8("finetune", EVAL_1B_R8V2_DIR, "r8v2_")

# Matched ctx per horizon (paper convention). Note H=336/720 use ctx 2048/3072,
# where forecast_windows() clips to MAX_CTX_TIMESTEPS=2048 while still reporting
# the un-clipped window's mean/std in the prompt — those two rows are inflated
# for every model in this table.
OUR_1B_R8 = {
    "1B no-LoRA\nZero-shot":   build_our_data(raw_1b_zeroshot,    lambda H: CTX_MATCH[H]),
    "1B no-LoRA\nFine-tuned":  build_our_data(raw_1b_finetune,    lambda H: CTX_MATCH[H]),
    "1B LoRA r8 v1\nZero-shot":   build_our_data(raw_1b_r8_zeroshot, lambda H: CTX_MATCH[H]),
    "1B LoRA r8 v1\nFine-tuned":  build_our_data(raw_1b_r8_finetune, lambda H: CTX_MATCH[H]),
    "1B LoRA r8 v2\nZero-shot":   build_our_data(raw_1b_r8v2_zeroshot, lambda H: CTX_MATCH[H]),
    "1B LoRA r8 v2\nFine-tuned":  build_our_data(raw_1b_r8v2_finetune, lambda H: CTX_MATCH[H]),
}

# Fixed ctx=512: every (ctx, H) here stays under the 2048 clip, so this is the
# only view of the 1B models free of the context-clipping artifact.
OUR_1B_R8_CTX512 = {
    "1B no-LoRA\nZero-shot":   build_our_data(raw_1b_zeroshot,    lambda H: 512),
    "1B no-LoRA\nFine-tuned":  build_our_data(raw_1b_finetune,    lambda H: 512),
    "1B LoRA r8 v1\nZero-shot":   build_our_data(raw_1b_r8_zeroshot, lambda H: 512),
    "1B LoRA r8 v1\nFine-tuned":  build_our_data(raw_1b_r8_finetune, lambda H: 512),
    "1B LoRA r8 v2\nZero-shot":   build_our_data(raw_1b_r8v2_zeroshot, lambda H: 512),
    "1B LoRA r8 v2\nFine-tuned":  build_our_data(raw_1b_r8v2_finetune, lambda H: 512),
}

OUR_FULLFT = {
    "Baseline\n(full FT)":      build_our_data(raw_fullft_baseline,     lambda H: 512),
    "TS-only LoRA\n(full FT)":  build_our_data(raw_fullft_ts_only_lora, lambda H: 512),
    "TS+Lang LoRA\n(full FT)":  build_our_data(raw_fullft_ts_lang_lora, lambda H: 512),
}

# Combined ctx=512 view: LoRA FT + Full FT side by side
OUR_COMBINED = {
    "Baseline\n(LoRA FT)":      build_our_data(raw_baseline,             lambda H: 512),
    "TS-only\n(LoRA FT)":       build_our_data(raw_ts_only_lora,         lambda H: 512),
    "TS+Lang\n(LoRA FT)":       build_our_data(raw_ts_lang_lora,         lambda H: 512),
    "Baseline\n(Full FT)":      build_our_data(raw_fullft_baseline,       lambda H: 512),
    "TS-only\n(Full FT)":       build_our_data(raw_fullft_ts_only_lora,   lambda H: 512),
    "TS+Lang\n(Full FT)":       build_our_data(raw_fullft_ts_lang_lora,   lambda H: 512),
}

# ── Paper baselines (Time-MoE Table 4) ───────────────────────────────────────
PAPER = {
    "TimeMoE-base\n(fine-tuned)": {
        "ETTh1":   {96:(0.345,0.375), 192:(0.388,0.411), 336:(0.414,0.427), 720:(0.425,0.433)},
        "ETTh2":   {96:(0.274,0.333), 192:(0.343,0.386), 336:(0.362,0.402), 720:(0.380,0.424)},
        "Weather": {96:(0.149,0.201), 192:(0.193,0.244), 336:(0.245,0.285), 720:(0.318,0.337)},
    },
    "TimeMoE-large\n(fine-tuned)": {
        "ETTh1":   {96:(0.335,0.374), 192:(0.390,0.415), 336:(0.411,0.430), 720:(0.411,0.430)},
        "ETTh2":   {96:(0.276,0.335), 192:(0.340,0.386), 336:(0.361,0.401), 720:(0.390,0.430)},
        "Weather": {96:(0.135,0.192), 192:(0.192,0.242), 336:(0.248,0.289), 720:(0.301,0.329)},
    },
    "iTransformer\n(full train)": {
        "ETTh1":   {96:(0.386,0.405), 192:(0.441,0.436), 336:(0.487,0.458), 720:(0.454,0.459)},
        "ETTh2":   {96:(0.297,0.349), 192:(0.380,0.400), 336:(0.428,0.432), 720:(0.454,0.452)},
        "Weather": {96:(0.148,0.230), 192:(0.182,0.230), 336:(0.213,0.269), 720:(0.267,0.329)},
    },
    "PatchTST\n(full train)": {
        "ETTh1":   {96:(0.370,0.400), 192:(0.413,0.429), 336:(0.422,0.440), 720:(0.447,0.468)},
        "ETTh2":   {96:(0.302,0.348), 192:(0.388,0.403), 336:(0.426,0.445), 720:(0.439,0.462)},
        "Weather": {96:(0.187,0.267), 192:(0.249,0.297), 336:(0.305,0.329), 720:(0.367,0.366)},
    },
    "TimeMixer\n(full train)": {
        "ETTh1":   {96:(0.405,0.419), 192:(0.433,0.444), 336:(0.434,0.449), 720:(0.425,0.433)},
        "ETTh2":   {96:(0.289,0.340), 192:(0.383,0.400), 336:(0.448,0.452), 720:(0.451,0.461)},
        "Weather": {96:(0.149,0.209), 192:(0.196,0.255), 336:(0.243,0.295), 720:(0.301,0.337)},
    },
    "DLinear\n(full train)": {
        "ETTh1":   {96:(0.386,0.400), 192:(0.437,0.459), 336:(0.481,0.459), 720:(0.457,0.466)},
        "ETTh2":   {96:(0.333,0.387), 192:(0.477,0.476), 336:(0.594,0.541), 720:(0.831,0.657)},
        "Weather": {96:(0.176,0.237), 192:(0.220,0.282), 336:(0.265,0.319), 720:(0.323,0.362)},
    },
    "FEDformer\n(full train)": {
        "ETTh1":   {96:(0.376,0.419), 192:(0.420,0.448), 336:(0.459,0.471), 720:(0.449,0.451)},
        "ETTh2":   {96:(0.358,0.397), 192:(0.429,0.441), 336:(0.496,0.487), 720:(0.463,0.474)},
        "Weather": {96:(0.238,0.314), 192:(0.275,0.329), 336:(0.339,0.380), 720:(0.389,0.409)},
    },
}

# ── Style helpers ─────────────────────────────────────────────────────────────
C_TITLE   = "1F4E79"
C_HEADER  = "2E75B6"
C_SUBHDR  = "BDD7EE"
C_ALT     = "EBF3FB"
C_WHITE   = "FFFFFF"
C_OUR0    = "D6E4F0"   # Baseline LoRA FT (light blue)
C_OUR1    = "F4B942"   # TS-only LoRA FT (amber)
C_OUR2    = "70AD47"   # TS+Lang LoRA FT (green)
C_OUR3    = "9DC3E6"   # Baseline Full FT (mid blue)
C_OUR4    = "FFD966"   # TS-only Full FT (yellow)
C_OUR5    = "A9D18E"   # TS+Lang Full FT (light green)
C_BEST    = "C6EFCE"
C_WORST   = "FFC7CE"
C_PENDING = "E0E0E0"
OUR_FILLS_3  = [C_OUR0, C_OUR1, C_OUR2]
OUR_FILLS_6  = [C_OUR0, C_OUR1, C_OUR2, C_OUR3, C_OUR4, C_OUR5]

def fill(h): return PatternFill("solid", fgColor=h)
def bold(size=10, white=False): return Font(bold=True, size=size, color="FFFFFF" if white else "000000")
def normal(size=10): return Font(size=size)
def center(): return Alignment(horizontal="center", vertical="center", wrap_text=True)

OUR_FILLS = OUR_FILLS_3   # default; overridden per sheet call

# ── Sheet builder ─────────────────────────────────────────────────────────────

def merge(ws, r1, c1, r2, c2):
    ws.merge_cells(start_row=r1, start_column=c1, end_row=r2, end_column=c2)

def write_sheet(ws, our_data, title_text, note_text, our_fills=None):
    fills        = our_fills if our_fills is not None else OUR_FILLS_3
    OUR_MODELS   = list(our_data.keys())
    PAPER_MODELS = list(PAPER.keys())
    all_models   = OUR_MODELS + PAPER_MODELS
    n_our        = len(OUR_MODELS)
    n_models     = len(all_models)
    COL_LABEL    = 1

    def model_col(i):
        return COL_LABEL + 1 + i * 2, COL_LABEL + 2 + i * 2

    total_cols = COL_LABEL + n_models * 2

    # Row 1 — title
    merge(ws, 1, 1, 1, total_cols)
    c = ws.cell(1, 1, title_text)
    c.fill = fill(C_TITLE); c.font = bold(12, white=True); c.alignment = center()

    # Row 2 — note
    merge(ws, 2, 1, 2, total_cols)
    c = ws.cell(2, 1, note_text)
    c.fill = fill("D6E4F0"); c.font = Font(italic=True, size=9); c.alignment = center()

    # Row 3 — group headers
    ws.cell(3, COL_LABEL, "").fill = fill(C_HEADER)
    our_start, _ = model_col(0)
    _, our_end   = model_col(n_our - 1)
    merge(ws, 3, our_start, 3, our_end)
    c = ws.cell(3, our_start, "Ours (fine-tuned 1 epoch)")
    c.fill = fill(C_OUR1); c.font = bold(10); c.alignment = center()

    pap_start, _ = model_col(n_our)
    _, pap_end   = model_col(n_our + len(PAPER_MODELS) - 1)
    merge(ws, 3, pap_start, 3, pap_end)
    c = ws.cell(3, pap_start, "Paper Baselines (Time-MoE Table 4)")
    c.fill = fill(C_HEADER); c.font = bold(10, white=True); c.alignment = center()

    # Row 4 — model names
    ws.cell(4, COL_LABEL, "").fill = fill(C_HEADER)
    for i, name in enumerate(all_models):
        mc, _ = model_col(i)
        merge(ws, 4, mc, 4, mc + 1)
        fc = fills[i] if i < n_our else C_SUBHDR
        c = ws.cell(4, mc, name)
        c.fill = fill(fc); c.font = bold(9); c.alignment = center()

    # Row 5 — MSE / MAE sub-headers
    ws.cell(5, COL_LABEL, "H").fill = fill(C_SUBHDR)
    ws.cell(5, COL_LABEL).font = bold(9); ws.cell(5, COL_LABEL).alignment = center()
    for i in range(n_models):
        mc, mec = model_col(i)
        fc = fills[i] if i < n_our else C_SUBHDR
        for col, lbl in [(mc, "MSE"), (mec, "MAE")]:
            c = ws.cell(5, col, lbl)
            c.fill = fill(fc); c.font = bold(9); c.alignment = center()

    # Data rows
    data_row = 6
    for dataset in DATASETS:
        n_h = len(HORIZONS)
        merge(ws, data_row, COL_LABEL, data_row + n_h - 1, COL_LABEL)
        c = ws.cell(data_row, COL_LABEL, dataset)
        c.fill = fill(C_HEADER); c.font = bold(10, white=True)
        c.alignment = Alignment(horizontal="center", vertical="center",
                                text_rotation=90, wrap_text=False)

        for h_idx, H in enumerate(HORIZONS):
            row   = data_row + h_idx
            row_bg = C_WHITE if h_idx % 2 == 0 else C_ALT

            # Gather MSE values for highlight
            mse_vals = {}
            for i, name in enumerate(all_models):
                d = our_data[name][dataset] if i < n_our else PAPER[name][dataset]
                mse_vals[i] = d[H][0]
            valid = {k: v for k, v in mse_vals.items() if v is not None}
            best  = min(valid.values()) if valid else None
            worst = max(valid.values()) if valid else None

            for i, name in enumerate(all_models):
                d = our_data[name][dataset] if i < n_our else PAPER[name][dataset]
                mse, mae = d[H]
                mc, mec = model_col(i)
                for col, val in [(mc, mse), (mec, mae)]:
                    if val is None:
                        c = ws.cell(row, col, "—")
                        c.fill = fill(C_PENDING)
                        c.font = Font(italic=True, size=9, color="888888")
                    else:
                        c = ws.cell(row, col, round(val, 4))
                        c.number_format = "0.0000"
                        if col == mc:
                            bg = C_BEST if val == best else (C_WORST if val == worst else row_bg)
                        else:
                            bg = row_bg
                        c.fill = fill(bg); c.font = normal(9)
                    c.alignment = center()

        # Average row
        avg_row = data_row + n_h
        merge(ws, avg_row, COL_LABEL, avg_row, COL_LABEL)
        c = ws.cell(avg_row, COL_LABEL, "Avg")
        c.fill = fill(C_SUBHDR); c.font = bold(9); c.alignment = center()

        for i, name in enumerate(all_models):
            d = our_data[name][dataset] if i < n_our else PAPER[name][dataset]
            mc, mec = model_col(i)
            for col, idx in [(mc, 0), (mec, 1)]:
                vals = [d[H][idx] for H in HORIZONS if d[H][idx] is not None]
                if vals:
                    c = ws.cell(avg_row, col, round(sum(vals) / len(vals), 4))
                    c.number_format = "0.0000"; c.font = bold(9)
                else:
                    c = ws.cell(avg_row, col, "—")
                    c.font = Font(italic=True, size=9, color="888888")
                c.fill = fill(C_SUBHDR); c.alignment = center()

        data_row += n_h + 2   # horizons + avg + gap row

    # Column widths
    ws.column_dimensions[get_column_letter(COL_LABEL)].width = 10
    for i in range(n_models):
        mc, mec = model_col(i)
        ws.column_dimensions[get_column_letter(mc)].width  = 9
        ws.column_dimensions[get_column_letter(mec)].width = 9
    ws.row_dimensions[4].height = 36
    ws.freeze_panes = ws.cell(6, 2)


# ── Build workbook ────────────────────────────────────────────────────────────
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

wb = openpyxl.Workbook()

ws1 = wb.active
ws1.title = "Standard Setting"
write_sheet(
    ws1, OUR_STANDARD,
    "Fine-tuning Evaluation — Standard Setting (matched ctx/horizon, Time-MoE Table 4)",
    "H=96→ctx=512 | H=192→ctx=1024 | H=336→ctx=2048 | H=720→ctx=3072  "
    "  Note: our models pretrained on MAX_CTX=2048; ctx=3072 is out-of-distribution.",
)

ws2 = wb.create_sheet("Fixed ctx=512 (LoRA FT)")
write_sheet(
    ws2, OUR_CTX512,
    "Fine-tuning Evaluation — Fixed ctx=512, LoRA fine-tuning (LLM frozen)",
    "All horizons at ctx=512 for our models.  "
    "Paper baselines use standard matched ctx (H=96→512, H=192→1024, …).",
    our_fills=OUR_FILLS_3,
)

ws3 = wb.create_sheet("Fixed ctx=512 (Full FT)")
write_sheet(
    ws3, OUR_FULLFT,
    "Fine-tuning Evaluation — Fixed ctx=512, Full fine-tuning (LLM unfrozen)",
    "LLM backbone fully trained: enc/dec/proj lr=1e-4, LLM lr=1e-5.  "
    "Paper baselines use standard matched ctx.",
    our_fills=OUR_FILLS_3,
)

ws4 = wb.create_sheet("LoRA vs Full FT (ctx=512)")
write_sheet(
    ws4, OUR_COMBINED,
    "Fine-tuning Comparison — LoRA FT vs Full FT at ctx=512",
    "Left 3 cols: LoRA FT (LLM frozen).  Right 3 cols: Full FT (LLM unfrozen, llm_lr=1e-5).  "
    "Paper baselines use standard matched ctx.",
    our_fills=OUR_FILLS_6,
)

C_OUR_1B0 = "7030A0"   # purple — 1B zero-shot
C_OUR_1B1 = "C00000"   # dark red — 1B fine-tuned

ws5 = wb.create_sheet("1B Model")
write_sheet(
    ws5, OUR_1B,
    "Gemma-3-1B — Zero-shot vs Fine-tuned (1 epoch, LoRA FT, matched ctx)",
    "H=96→ctx=512 | H=192→ctx=1024 | H=336→ctx=2048 | H=720→ctx=3072  "
    "  Note: 1B model pretrained on MAX_CTX=2048; ctx=3072 is OOD.",
    our_fills=[C_OUR_1B0, C_OUR_1B1],
)

C_R8_0 = "7030A0"   # purple   — no-LoRA zero-shot
C_R8_1 = "B084CC"   # lilac    — no-LoRA fine-tuned
C_R8_2 = "C00000"   # dark red — LoRA r8 zero-shot
C_R8_3 = "E88080"   # salmon   — LoRA r8 fine-tuned
C_R8_4 = "1F6F3C"   # green    — LoRA r8 v2 zero-shot
C_R8_5 = "8FCFA8"   # mint     — LoRA r8 v2 fine-tuned
R8_FILLS = [C_R8_0, C_R8_1, C_R8_2, C_R8_3, C_R8_4, C_R8_5]

ws6 = wb.create_sheet("1B LoRA r8 (matched ctx)")
write_sheet(
    ws6, OUR_1B_R8,
    "Gemma-3-1B — no-LoRA vs LoRA r=8 pretraining, zero-shot and fine-tuned (matched ctx)",
    "v1/v2 are two epoch-2 snapshots of job 3462055 96 min apart (val 0.6735 / 0.6480); no-LoRA is its epoch-5 best. The v1-v2 spread is the checkpoint noise floor: mean 3.0%, max 10.6% per cell. "
    "H=336/720 use ctx=2048/3072, where the eval harness clips context to 2048 but keeps the "
    "un-clipped mean/std in the prompt — those rows are inflated for both models.",
    our_fills=R8_FILLS,
)

ws7 = wb.create_sheet("1B LoRA r8 (ctx=512)")
write_sheet(
    ws7, OUR_1B_R8_CTX512,
    "Gemma-3-1B — no-LoRA vs LoRA r=8, all horizons at ctx=512 (no context clipping)",
    "Every cell here stays under the 2048-timestep clip, so this is the artifact-free "
    "comparison. Paper baselines still use their standard matched ctx.",
    our_fills=R8_FILLS,
)

os.makedirs("results", exist_ok=True)
wb.save(OUT)
print(f"Saved → {OUT}")
print("Sheets:", wb.sheetnames)
