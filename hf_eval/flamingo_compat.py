"""Runtime compatibility shim for evaluating OpenTSLM Flamingo checkpoints.

This module performs NO edits to repo source. It monkey-patches the installed
`open_flamingo` (0.0.2) `Flamingo.__init__` so that when OpenTSLM passes a
`SimpleNamespace(visual=time_series_encoder)` as the vision encoder, we unwrap
it to the real encoder module before the base constructor stores it.

Why this is needed:
  * open_flamingo 0.0.2 stores `self.vision_encoder = vision_encoder` verbatim.
  * TimeSeriesFlamingoWithTrainableEncoder overrides `_encode_vision_x` to call
    `self.vision_encoder(x)` DIRECTLY (not `.visual(x)`), and OpenTSLMFlamingo
    calls `model.vision_encoder.requires_grad_(...)`. Both require the raw
    nn.Module, not a SimpleNamespace.
  * The published checkpoints store encoder weights under `*.vision_encoder.*`
    (a registered submodule). With the SimpleNamespace these weights silently
    fail to load under strict=False. Unwrapping registers the module so the
    weights load correctly.

Import this module BEFORE constructing the Flamingo model.
"""
from types import SimpleNamespace
import torch.nn as nn
import open_flamingo.src.flamingo as _ofm

_PATCHED_FLAG = "_opentslm_eval_visual_unwrap"

if not getattr(_ofm.Flamingo.__init__, _PATCHED_FLAG, False):
    _orig_init = _ofm.Flamingo.__init__

    def _patched_init(self, vision_encoder, *args, **kwargs):
        if isinstance(vision_encoder, SimpleNamespace) and hasattr(vision_encoder, "visual"):
            visual = vision_encoder.visual
            if isinstance(visual, nn.Module):
                vision_encoder = visual
        return _orig_init(self, vision_encoder, *args, **kwargs)

    setattr(_patched_init, _PATCHED_FLAG, True)
    _ofm.Flamingo.__init__ = _patched_init
    # keep the re-exported symbol in sync
    try:
        import open_flamingo as _of
        _of.Flamingo.__init__ = _patched_init
    except Exception:
        pass
    print("[flamingo_compat] patched open_flamingo.Flamingo.__init__ "
          "to unwrap SimpleNamespace(visual=...) -> nn.Module", flush=True)
