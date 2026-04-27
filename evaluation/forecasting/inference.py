"""
Inference script for OpenTSLM-SP (pretrained checkpoint).

For each test case this script prints:
  1. The generated TEXT from the LLM
  2. The reconstructed TIME SERIES VALUES decoded from LLM hidden states at TS patch positions
  3. A direct Encoder->Decoder round-trip (no LLM) as a sanity check
"""

import torch
import numpy as np
import argparse

from opentslm.model.llm.OpenTSLMSPFixPos2D import OpenTSLMSPFixPos2D
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)


# ───────────────────────────────────────────────────────────────────────────────
#  Helper: build a single-sample batch dict
# ───────────────────────────────────────────────────────────────────────────────
def make_batch(time_series: torch.Tensor, question: str = "Describe this time series."):
    """time_series : [T] or [T, 1]"""
    if time_series.dim() == 1:
        time_series = time_series.unsqueeze(-1)   # [T, 1]

    return {
        "pre_prompt": f"You are a time-series analysis assistant.\n{question}\n",
        "time_series_text": ["<ts>"],
        "time_series": [time_series],
        "post_prompt": "</ts>\nAnswer:",
        "answer": "",
    }


# ───────────────────────────────────────────────────────────────────────────────
#  Core inference: text generation + TS reconstruction from LLM hidden states
# ───────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model: OpenTSLMSPFixPos2D, batch: list, args) -> dict:
    """
    Runs the full pipeline and returns a dict with:
      - 'text'        : str    — LLM generated text
      - 'ts_recon'    : list   — reconstructed time series values (float list)
      - 'n_patches'   : int    — number of TS patch positions in the sequence
      - 'patch_size'  : int    — values per patch
    """
    device = model.device

    # ── Step 1: build inputs (same as model.generate internaly) ──────────────
    inputs_embeds, attention_mask, _, _, _, _, ts_chan_id = \
        model.pad_and_apply_batch(batch, return_targets=False)
    # inputs_embeds : [1, L, H]   (batch size 1)
    # attention_mask: [1, L]
    # L = pre_prompt_tokens + <ts>_token + N_patches + </ts>_token + Answer:_tokens

    # ── Step 2: forward through LLM to get hidden states ─────────────────────
    outputs = model.llm(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    last_hidden = outputs.hidden_states[-1]   # [1, L, H]

    # ── Step 3: identify which positions are TS patch embeddings ─────────────
    #
    # The sequence layout (per pad_and_apply_batch) is:
    #   [pre_prompt tokens...] [ts_text tokens...] [patch_0 ... patch_N-1] [post_prompt tokens...]
    #
    # We detect patch positions by looking for the contiguous block of L positions
    # that are NOT from text tokens. We do this by re-running the tokeniser to
    # count text token lengths, then computing patch positions by subtraction.
    #
    # Simpler approach: re-use the ts_mask returned when return_targets=True.
    # We call pad_and_apply_batch again with return_targets=True to get ts_mask.
    # (cheap — no LLM forward, just embedding + encoder)
    _, _, _, ts_mask, _, _, _ = \
        model.pad_and_apply_batch(batch, return_targets=True)
    # ts_mask: [1, L] bool — True at every TS patch position

    ts_positions = ts_mask[0]                          # [L] bool
    n_patches    = int(ts_positions.sum().item())

    print(f"  Sequence length      : {inputs_embeds.size(1)} tokens/patches total")
    print(f"  TS patch positions   : {n_patches} patches  (patch_size={model.patch_size})")
    print(f"  TS values covered    : {n_patches * model.patch_size} time steps")

    # ── Step 4: extract hidden states at TS patch positions ──────────────────
    ts_hidden = last_hidden[0][ts_positions]           # [N_patches, H]

    # ── Step 5: project LLM hidden dim -> encoder dim ────────────────────────
    ts_dec_input = model.decoder_projector(ts_hidden.float())   # [N_patches, enc_dim]

    # ── Step 6: decode patches -> raw values ─────────────────────────────────
    # decoder expects [N_ts, N_patches, enc_dim] and a list of channel ids
    ts_dec_input_3d = ts_dec_input.unsqueeze(0)                 # [1, N_patches, enc_dim]
    ts_reconstructed = model.decoder(ts_dec_input_3d, ts_chan_id)
    # ts_reconstructed: [1, N_patches, patch_size]

    ts_values = ts_reconstructed.squeeze(0).reshape(-1).cpu().float().tolist()
    # ts_values is now a flat list of  N_patches * patch_size  floats

    # ── Step 7: generate text (autoregressive, no hidden-state access needed) ─
    gen_ids = model.llm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )
    generated_text = model.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]

    return {
        "text":       generated_text,
        "ts_recon":   ts_values,
        "n_patches":  n_patches,
        "patch_size": model.patch_size,
    }


# ───────────────────────────────────────────────────────────────────────────────
#  Encoder -> Decoder round-trip (no LLM at all) — baseline sanity check
# ───────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def encoder_decoder_roundtrip(model: OpenTSLMSPFixPos2D, ts: torch.Tensor) -> torch.Tensor:
    """
    Bypasses the LLM entirely.
    Encodes ts directly, then decodes back to values.
    Useful to verify the encoder/decoder pair in isolation.
    Returns flat tensor of reconstructed values, trimmed to original length.
    """
    device = model.device
    patch_size = model.patch_size

    if ts.dim() == 1:
        ts = ts.unsqueeze(-1)

    T = ts.size(0)
    rem = T % patch_size
    if rem:
        ts = torch.cat([ts, torch.zeros(patch_size - rem, 1)], dim=0)

    ts_in   = ts.squeeze(-1).unsqueeze(0).to(device)   # [1, T_padded]
    chan_id = torch.tensor([0], device=device)

    enc   = model.encoder(ts_in, chan_id=chan_id)       # [1, N_patches, enc_dim]
    recon = model.decoder(enc, [0])                     # [1, N_patches, patch_size]
    return recon.squeeze(0).reshape(-1).cpu()[:T]


# ───────────────────────────────────────────────────────────────────────────────
#  Main
# ───────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="OpenTSLM pretrain inference")

    # -- inference flags -------------------------------------------------------
    parser.add_argument("--checkpoint",      type=str, required=True)
    parser.add_argument("--llm_id",          type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--device",          type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_new_tokens",  type=int, default=200)
    parser.add_argument("--ts_length",       type=int, default=128)
    parser.add_argument("--normalize",       action="store_true")
    parser.add_argument("--print_all_values", action="store_true",
                        help="Print every reconstructed value; otherwise prints first 20")

    # -- model / encoder / decoder flags (passed straight through as args) -----
    parser.add_argument("--is_pretraining",      action="store_true", default=False)
    parser.add_argument("--use_only_time_pe",    action="store_true", default=False)
    parser.add_argument("--use_only_channel_pe", action="store_true", default=False)
    # add more here as needed to match your training config, e.g.:
    # parser.add_argument("--d_model", type=int, default=256)

    args = parser.parse_args()

    print(f"Device     : {args.device}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"LLM        : {args.llm_id}\n")

    # -- 1. Build model --------------------------------------------------------
    model = OpenTSLMSPFixPos2D(args=args, llm_id=args.llm_id, device=args.device)

    # -- 2. Load weights -------------------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)

    model.encoder.load_state_dict(ckpt["encoder_state"])
    print("  encoder loaded")

    model.projector.load_state_dict(ckpt["projector_state"])
    print("  projector loaded")

    if "decoder_state" in ckpt:
        model.decoder.load_state_dict(ckpt["decoder_state"])
        print("  decoder loaded")
    else:
        raise ValueError("No decoder_state found in checkpoint -- cannot run inference without decoder weights")

    if "decoder_projector_state" in ckpt:
        model.decoder_projector.load_state_dict(ckpt["decoder_projector_state"])
        print("  decoder_projector loaded")
    else:
        raise ValueError("No decoder_projector_state found in checkpoint -- cannot run inference without decoder_projector weights")

    model.eval()

    # -- 3. Test cases ---------------------------------------------------------
    t         = torch.linspace(0, 4 * np.pi, args.ts_length)
    ts_sine   = torch.sin(t).float()
    ts_random = torch.cumsum(torch.randn(args.ts_length), dim=0).float()

    test_cases = [
        ("Sine wave",   ts_sine,   "What pattern do you see in this time series?"),
        ("Random walk", ts_random, "Is this time series stationary?"),
    ]

    # -- 4. Run inference for each test case -----------------------------------
    for name, ts, question in test_cases:
        print(f"\n{'='*70}")
        print(f"TEST CASE : {name}")
        print(f"INPUT TS  : length={ts.size(0)}, min={ts.min():.4f}, max={ts.max():.4f}")
        print(f"QUESTION  : {question}")
        print(f"{'='*70}")

        batch = [make_batch(ts, question=question)]
        batch = extend_time_series_to_match_patch_size_and_aggregate(
            batch, normalize=args.normalize
        )

        result = run_inference(model, batch, args)

        # ── Text output ───────────────────────────────────────────────────────
        # print(f"\n[GENERATED TEXT]")
        # print(f"  {result['text']}")

        # ── Reconstructed TS values (from LLM hidden states -> decoder) ───────
        n   = result["n_patches"]
        ps  = result["patch_size"]
        vals = result["ts_recon"]

        print(f"\n[RECONSTRUCTED TIME SERIES VALUES]")
        print(f"  Source      : LLM last hidden state at {n} TS patch positions")
        print(f"                -> decoder_projector -> decoder")
        print(f"  Patches     : {n}  x  patch_size {ps}  =  {len(vals)} total values")
        print(f"  Value range : min={min(vals):.4f}, max={max(vals):.4f}, mean={sum(vals)/len(vals):.4f}")

        display = vals if args.print_all_values else vals[:20]
        label   = "all values" if args.print_all_values else "first 20 values"
        print(f"  ({label})")
        for i, v in enumerate(display):
            patch_idx = i // ps
            pos_in_patch = i % ps
            print(f"    [{i:4d}]  patch={patch_idx:3d}  pos={pos_in_patch}  value={v:+.6f}")

        if not args.print_all_values and len(vals) > 20:
            print(f"    ... ({len(vals) - 20} more values; use --print_all_values to see all)")

        # ── Compare to original (if same length) ──────────────────────────────
        orig = ts.tolist()
        trim = min(len(orig), len(vals))
        mse_llm = sum((vals[i] - orig[i])**2 for i in range(trim)) / trim
        print(f"\n  MSE vs original (LLM path) : {mse_llm:.6f}")

    # -- 5. Encoder->Decoder round-trip sanity check (no LLM) -----------------
    print(f"\n{'='*70}")
    print("SANITY CHECK: Encoder -> Decoder round-trip (NO LLM)")
    print("(This tells you how good the encoder/decoder pair is in isolation)")
    print(f"{'='*70}")

    recon_direct = encoder_decoder_roundtrip(model, ts_sine)
    orig = ts_sine[:recon_direct.size(0)].tolist()
    recon_list = recon_direct.tolist()
    mse_direct = sum((r - o)**2 for r, o in zip(recon_list, orig)) / len(orig)

    print(f"  Input  first 5 : {[f'{v:+.4f}' for v in orig[:5]]}")
    print(f"  Recon  first 5 : {[f'{v:+.4f}' for v in recon_list[:5]]}")
    print(f"  MSE            : {mse_direct:.6f}")
    print()
    print("  Interpretation:")
    print("  - Low MSE here = encoder+decoder work well together")
    print("  - If LLM-path MSE >> direct MSE, the LLM hidden states are not yet")
    print("    well-aligned with the decoder (expected early in pretraining)")


if __name__ == "__main__":
    main()