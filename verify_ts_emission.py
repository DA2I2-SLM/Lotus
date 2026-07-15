#!/usr/bin/env python3
"""
Verify whether the trained model can generate <ts> and </ts> tokens
as part of free autoregressive generation (chat mode).

The model is a chat model — it can generate text at any time, and <ts> / </ts>
are special tokens it should emit naturally when producing a forecast response,
just like any other token in its vocabulary.

Test: give the model a natural-language forecasting prompt (with history patches),
let it generate freely for up to MAX_DECODE_STEPS tokens, and check whether
<ts> and </ts> appear anywhere in the output.
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from inference_phase2 import build_model, _normalize

# ---------------------------------------------------------------------------
CKPT = (
    "/scratch/s223540177/FoundationalTS/result/sincos_patch32/"
    "gemma_3_270m/OpenTSLMSPFixPos2D/"
    "joint_phase2_downstream_from_joint_phase1_extend/"
    "checkpoints/best_model.pt"
)
LLM_ID          = "google/gemma-3-270m"
PATCH_SIZE      = 32
HISTORY_LEN     = 512
PRED_LEN        = 96
N_CHANNELS      = 3
MAX_DECODE_STEPS = 200          # how many tokens to generate before giving up
TOP_K           = 10
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------------------------------------------------


def _build_prompt_sample(
    history_norm: torch.Tensor,   # [C, T]  already normalised
    norm_stats: list,
    pred_len: int,
    patch_size: int,
    ds_name: str = "ETTh1",
) -> dict:
    C = history_norm.shape[0]
    N_per_ch = pred_len // patch_size
    time_series, time_series_text = [], []
    for c in range(C):
        time_series.append(history_norm[c].tolist())
        m, s = norm_stats[c]
        time_series_text.append(
            f"This is the time series (ch{c}), mean {m:.4f} std {s:.4f}. <ts>"
        )
    return {
        "type":                  "chronos",
        "pre_prompt":            (
            f"This is {ds_name} data. You are given {HISTORY_LEN} timesteps "
            f"across {C} channel{'s' if C != 1 else ''}:"
        ),
        "post_prompt":           (
            f". Predict the next {pred_len} timesteps "
            f"as {N_per_ch} patch{'es' if N_per_ch != 1 else ''} per channel."
        ),
        "time_series":           time_series,
        "time_series_text":      time_series_text,
        "ts_already_normalized": True,
        "n_channels":            C,
        "_n_channels":           C,
    }


@torch.no_grad()
def run(model, sample: dict):
    ts_id  = model.ts_start_token_id
    ets_id = model.ts_end_token_id
    eos_id = model.tokenizer.eos_token_id

    print(f"  <ts>  token id : {ts_id}")
    print(f"  </ts> token id : {ets_id}")

    # ── Prompt forward pass ───────────────────────────────────────────────────
    inputs_embeds, attn_mask, _, _, _, _, _ = model.pad_and_apply_batch(
        [sample], return_targets=True
    )
    out = model.llm(
        inputs_embeds=inputs_embeds,
        attention_mask=attn_mask,
        use_cache=True,
        output_hidden_states=False,
    )

    valid_len = int(attn_mask[0].sum().item())
    first_logits = out.logits[0, valid_len - 1, :]

    # ── Report P(<ts>) and P(</ts>) at step 0 ────────────────────────────────
    probs = F.softmax(first_logits.float(), dim=-1)
    top_vals, top_ids = torch.topk(probs, TOP_K)

    print(f"\n{'='*60}")
    print(f"Step-0 token distribution (first generated token):")
    print(f"  P(<ts>)  = {probs[ts_id].item():.6f}   rank {(probs > probs[ts_id]).sum().item() + 1}")
    print(f"  P(</ts>) = {probs[ets_id].item():.6f}   rank {(probs > probs[ets_id]).sum().item() + 1}")
    print(f"\n  Top-{TOP_K}:")
    for rank, (tid, prob) in enumerate(zip(top_ids.tolist(), top_vals.tolist()), 1):
        tok = model.tokenizer.decode([tid]).replace("\n", "\\n")
        flag = ""
        if tid == ts_id:  flag = "  ← <ts>"
        if tid == ets_id: flag = "  ← </ts>"
        print(f"    {rank:2d}. p={prob:.6f}  id={tid:6d}  '{tok}'{flag}")

    # ── Free generation — up to MAX_DECODE_STEPS tokens ──────────────────────
    print(f"\n{'='*60}")
    print(f"Free generation (up to {MAX_DECODE_STEPS} tokens):\n")

    past_kv  = out.past_key_values
    cur_mask = attn_mask
    next_id  = first_logits.argmax().item()

    generated       = []
    ts_positions    = []
    ets_positions   = []

    for step in range(MAX_DECODE_STEPS):
        tok = model.tokenizer.decode([next_id]).replace("\n", "\\n")
        flag = ""
        if next_id == ts_id:
            flag = "  ←←← <ts>"
            ts_positions.append(step)
        elif next_id == ets_id:
            flag = "  ←←← </ts>"
            ets_positions.append(step)
        generated.append(next_id)

        print(f"  {step+1:3d}: id={next_id:6d}  '{tok}'{flag}")

        if next_id == eos_id:
            print("       [EOS — generation complete]")
            break

        next_emb = model.llm.get_input_embeddings()(
            torch.tensor([[next_id]], device=model.device)
        )
        cur_mask = torch.cat(
            [cur_mask, torch.ones(1, 1, device=model.device, dtype=cur_mask.dtype)],
            dim=1,
        )
        step_out = model.llm(
            inputs_embeds=next_emb,
            attention_mask=cur_mask,
            past_key_values=past_kv,
            use_cache=True,
            output_hidden_states=False,
        )
        past_kv = step_out.past_key_values
        next_id = step_out.logits[0, -1, :].argmax().item()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  <ts>  emitted: {len(ts_positions)}x  at steps {ts_positions}")
    print(f"  </ts> emitted: {len(ets_positions)}x  at steps {ets_positions}")

    if ts_positions:
        print(f"\n  PASS — model can emit <ts> during free generation.")
    else:
        print(f"\n  FAIL — <ts> never appeared in {len(generated)} generated tokens.")
        print(f"  The model has not learned to autonomously start a forecast block.")
    print(f"{'='*60}\n")


def main():
    model = build_model(
        ckpt_path=CKPT,
        llm_id=LLM_ID,
        patch_size=PATCH_SIZE,
        enc_dec_type="transformer",
        pe_type="sincos",
        device=DEVICE,
        lora_r=32,
        lora_alpha=64,
    )
    model.eval()

    torch.manual_seed(42)
    history_raw  = torch.randn(N_CHANNELS, HISTORY_LEN)
    history_norm, norm_stats = _normalize(history_raw)

    sample = _build_prompt_sample(
        history_norm=history_norm,
        norm_stats=norm_stats,
        pred_len=PRED_LEN,
        patch_size=PATCH_SIZE,
    )

    run(model, sample)


if __name__ == "__main__":
    main()
