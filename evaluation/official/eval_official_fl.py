#!/usr/bin/env python3
"""Evaluate a curriculum-trained OFFICIAL OpenTSLM-Flamingo stage checkpoint
with the eval_cot_phase2 harness (same datasets/metrics/manifest as every
other model).

Mirrors train_official_flamingo.py's environment exactly: open_flamingo 2.0.1
shadow tree wins over the installed 0.0.2 (0.0.2's generate() takes no
**kwargs -> TypeError -> empty preds), dims 128/patch4, vendored original
class, right-padded tokenizers. The stage checkpoint's `model_state` is the
full state dict of the class instance — loaded with a hard assert that nothing
is missing/unexpected.

Usage: same CLI as eval_cot_phase2.py.
Set OFFICIAL_EVAL_SMOKE=1 to only construct+load on CPU and exit (wiring test).
"""
import importlib.util
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # repo root, clone-relative
# Directory holding official_flamingo_class.py (defaults to this directory).
HERE = os.environ.get("FTS_CIK_VARIANTS", os.path.dirname(os.path.abspath(__file__)))
# Optional shadow checkout of open_flamingo 2.0.1; leave unset to use the installed package.
OF201 = os.environ.get("FTS_OF201_SHADOW", "")

assert "open_flamingo" not in sys.modules, "open_flamingo already imported"
if OF201:
    sys.path.insert(0, OF201)
sys.path.insert(1, REPO + "/src")
sys.path.insert(0, REPO + "/evaluation/cot")

os.environ["OPENTSLM_EMBED_DIM"] = "128"
os.environ["OPENTSLM_PATCH_SIZE"] = "4"
import opentslm.model_config as mc
mc.EMBED_DIM = 128
mc.ENCODER_OUTPUT_DIM = 128
mc.TRANSFORMER_INPUT_DIM = 128
mc.PATCH_SIZE = 4

import open_flamingo  # noqa: E402
_of_path = os.path.dirname(open_flamingo.__file__)
assert _of_path.startswith(OF201), f"wrong open_flamingo resolved: {_of_path}"
print(f"[eval_official_fl] open_flamingo from {_of_path} (2.0.1 shadow)", flush=True)

import torch  # noqa: E402

import transformers  # noqa: E402
_ORIG_TOK = transformers.AutoTokenizer.from_pretrained.__func__


def _tok_right(cls, *a, **k):
    tok = _ORIG_TOK(cls, *a, **k)
    try:
        tok.padding_side = "right"
    except Exception:
        pass
    return tok


transformers.AutoTokenizer.from_pretrained = classmethod(_tok_right)

_spec = importlib.util.spec_from_file_location(
    "official_flamingo_class", HERE + "/official_flamingo_class.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["official_flamingo_class"] = _mod
_spec.loader.exec_module(_mod)
_Official = _mod.OpenTSLMFlamingo


from opentslm.time_series_datasets.util import (  # noqa: E402
    extend_time_series_to_match_patch_size_and_aggregate as _collate,
)


class _FLEval(_Official):
    # harness expects generate(batch, max_new_tokens=...) -> (texts, ts).
    # Same as SP: raw harness rows must be collated (patch4) before
    # pad_and_apply_batch, else AttributeError -> silent all-zero eval.
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
        return _Official.generate(self, cb, max_new_tokens=max_new_tokens, **kw), None


def _load_model(ckpt_path, llm_id, device, lora_r=4, lora_alpha=8, use_freq=False,
                arch=None, **_ignored):
    # arch/**_ignored: eval_cot_phase2.load_model grew an `arch` kwarg so the
    # gemma-3-1b transformer-encoder checkpoints could be reconstructed. This
    # shim replaces that function wholesale and hardcodes the official
    # architecture, so it must ACCEPT and discard the new kwargs -- otherwise
    # every official-variant eval dies with
    # "TypeError: _load_model() got an unexpected keyword argument 'arch'".
    print(f"[eval_official_fl] building official Flamingo llm={llm_id}")
    model = _FLEval(device=device, llm_id=llm_id)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ck["model_state"], strict=False)
    assert not missing and not unexpected, (
        f"state dict mismatch: missing={len(missing)} unexpected={len(unexpected)} "
        f"(first missing: {missing[:3]}, first unexpected: {unexpected[:3]})")
    print(f"[eval_official_fl] loaded {ckpt_path} epoch={ck.get('epoch')} "
          f"val_loss={ck.get('val_loss')} (missing=0 unexpected=0)")
    model.to(device)
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
          f"params={sum(p.numel() for p in m.parameters())/1e6:.0f}M")
    sys.exit(0)

import eval_cot_phase2 as E  # noqa: E402
E.load_model = _load_model
E.main()
