#!/usr/bin/env python3
"""Evaluate a curriculum-trained OFFICIAL OpenTSLM-SP stage checkpoint with the
eval_cot_phase2 harness (same datasets/metrics/manifest as every other model).

Mirrors train_official_sp.py's environment exactly (dims 128/patch4, pre-norm
encoder, eager+bf16 LLM, right-padded tokenizers), then replaces the harness's
FixPos2D load_model with one that builds the vanilla OpenTSLMSP and loads the
stage checkpoint (encoder/projector/LoRA states, LoRA config read from ckpt).

Usage: same CLI as eval_cot_phase2.py, e.g.
  python eval_official_sp.py --ckpt <stage best_model.pt> --llm_id <gemma snap> \
      --tasks har sleep pamap2 tsqa --split test --max_samples 0 --out_dir <dir>
Set OFFICIAL_EVAL_SMOKE=1 to only construct+load on CPU and exit (wiring test).
"""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # repo root, clone-relative
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO + "/src")
sys.path.insert(0, HERE)  # encoder_compat / model_compat
sys.path.insert(0, REPO + "/evaluation/cot")

# official dims — before any opentslm import binds them
os.environ["OPENTSLM_EMBED_DIM"] = "128"
os.environ["OPENTSLM_PATCH_SIZE"] = "4"
import opentslm.model_config as mc
mc.EMBED_DIM = 128
mc.ENCODER_OUTPUT_DIM = 128
mc.TRANSFORMER_INPUT_DIM = 128
mc.PATCH_SIZE = 4

import encoder_compat  # noqa: F401  (pre-norm TransformerCNNEncoder)
import torch
import model_compat
model_compat.apply(attn_implementation="eager", torch_dtype=torch.bfloat16)

import transformers
_ORIG_TOK = transformers.AutoTokenizer.from_pretrained.__func__


def _tok_right(cls, *a, **k):
    tok = _ORIG_TOK(cls, *a, **k)
    try:
        tok.padding_side = "right"
    except Exception:
        pass
    return tok


transformers.AutoTokenizer.from_pretrained = classmethod(_tok_right)

from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP as _OrigSP  # noqa: E402
from opentslm.time_series_datasets.util import (  # noqa: E402
    extend_time_series_to_match_patch_size_and_aggregate as _collate,
)


class _SPEval(_OrigSP):
    # harness expects generate(batch, max_new_tokens=...) -> (texts, ts).
    # The harness passes RAW dataset rows (time_series = list of numpy arrays);
    # pad_and_apply_batch needs the trainer's collated form (stacked tensor +
    # lengths) — without it every batch dies with AttributeError ('numpy.ndarray'
    # has no attribute 'dim'), silently swallowed into '<error>' preds and an
    # all-zero table. Collate on shallow copies so dataset rows stay unmutated.
    def generate(self, batch, max_new_tokens=200, **kw):
        cb = _collate([dict(r) for r in batch], patch_size=4)
        assert len(cb) == len(batch), \
            f"collate dropped {len(batch) - len(cb)} rows — preds would misalign"
        # Force greedy. Neither class passes do_sample, so both inherit
        # google/gemma-3-270m's generation_config {do_sample: true, top_k: 64,
        # top_p: 0.95} and decode by SAMPLING -- while our own model decodes
        # greedily via a hand-rolled argmax loop. Re-running one cell three
        # times gave 0.2222 / 0.2308 / 0.2564, i.e. ~3 pt of pure decode noise
        # against the official rows. Caller-supplied kwargs still win.
        kw.setdefault("do_sample", False)
        return _OrigSP.generate(self, cb, max_new_tokens=max_new_tokens, **kw), None


def _load_model(ckpt_path, llm_id, device, lora_r=4, lora_alpha=8, use_freq=False,
                arch=None, **_ignored):
    # arch/**_ignored: eval_cot_phase2.load_model grew an `arch` kwarg so the
    # gemma-3-1b transformer-encoder checkpoints could be reconstructed. This
    # shim replaces that function wholesale and hardcodes the official
    # architecture, so it must ACCEPT and discard the new kwargs -- otherwise
    # every official-variant eval dies with
    # "TypeError: _load_model() got an unexpected keyword argument 'arch'".
    print(f"[eval_official_sp] building vanilla OpenTSLMSP llm={llm_id}")
    model = _SPEval(llm_id=llm_id, device=device)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ck.get("lora_enabled", False) and "lora_state" in ck:
        cfg = ck.get("lora_config")
        pc = next(iter(cfg.values())) if isinstance(cfg, dict) and cfg else cfg
        r = int(getattr(pc, "r", lora_r))
        a = int(getattr(pc, "lora_alpha", lora_alpha))
        print(f"[eval_official_sp] enabling LoRA r={r} alpha={a} (from ckpt config)")
        model.enable_lora(lora_r=r, lora_alpha=a)
    model.encoder.load_state_dict(ck["encoder_state"])
    model.projector.load_state_dict(ck["projector_state"])
    model.load_lora_state_from_checkpoint(ck, allow_missing=False)
    print(f"[eval_official_sp] loaded {ckpt_path} epoch={ck.get('epoch')} "
          f"val_loss={ck.get('val_loss')}")
    model.eval()
    return model


if os.environ.get("OFFICIAL_EVAL_SMOKE") == "1":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--llm_id", required=True)
    a, _ = p.parse_known_args()
    m = _load_model(a.ckpt, a.llm_id, "cpu")
    print(f"[SMOKE OK] eos={m.get_eos_token()!r} "
          f"trainable={sum(p.numel() for p in m.parameters() if p.requires_grad)/1e6:.1f}M")
    sys.exit(0)

import eval_cot_phase2 as E  # noqa: E402
E.load_model = _load_model
E.main()
