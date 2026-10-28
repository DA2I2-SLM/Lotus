"""Runtime shim (no repo edits): fix the OpenTSLM-SP time-series encoder.

The published SP checkpoints were trained with a PRE-norm transformer encoder
(`nn.TransformerEncoderLayer(norm_first=True)`), but the local
`TransformerCNNEncoder` builds it with the default POST-norm
(`norm_first=False`). The two share identical parameter names, so the checkpoint
loads with strict=True WITHOUT error — but the forward computation is wrong and
collapses every time series to (nearly) the same embedding, which makes the LLM
emit degenerate garbage.

Empirically, for two very different Sleep series the relative output difference
is 4e-6 with norm_first=False vs 1.8e-1 with norm_first=True.

This module monkey-patches `TransformerCNNEncoder.__init__` so that every
instance (including the one rebuilt inside OpenTSLMSP.load_from_file) uses a
pre-norm transformer. Import BEFORE constructing the model.
"""
import torch.nn as nn
from opentslm.model.encoder.TransformerCNNEncoder import TransformerCNNEncoder

_FLAG = "_opentslm_eval_norm_first_fixed"

if not getattr(TransformerCNNEncoder.__init__, _FLAG, False):
    _orig_init = TransformerCNNEncoder.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        old = self.encoder  # nn.TransformerEncoder built post-norm
        layer0 = old.layers[0]
        d_model = layer0.self_attn.embed_dim
        nhead = layer0.self_attn.num_heads
        ff_dim = layer0.linear1.out_features
        num_layers = len(old.layers)
        # gelu activation matches the repo default
        new_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim,
            dropout=0.0, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(new_layer, num_layers=num_layers)

    setattr(_patched_init, _FLAG, True)
    TransformerCNNEncoder.__init__ = _patched_init
    print("[encoder_compat] patched TransformerCNNEncoder to use norm_first=True "
          "(pre-norm) to match the published checkpoints", flush=True)
