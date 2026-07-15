#!/usr/bin/env python3
"""
TimeMixer rolling-window forecasting baseline.

Faithful self-contained implementation of TimeMixer (ICLR 2024) architecture.

Reference: Wang et al., "TimeMixer: Decomposable Multiscale Mixing for
           Time Series Forecasting", ICLR 2024.
           https://github.com/kwuking/TimeMixer

Architecture matches the official code:
  - TokenEmbedding: Conv1d(1, d_model, kernel_size=3, circular padding)
  - Normalize (RevIN): learnable per-channel instance norm at each scale
  - MultiScaleSeasonMixing: bottom-up (fine → coarse) with Linear projectors
  - MultiScaleTrendMixing: top-down (coarse → fine) with Linear projectors
  - PastDecomposableMixing (PDM): decompose → mix → residual
  - FutureMixing: scale-wise Linear(seq_len//window^i → pred_len) + projection
  - Channel-independence mode (each channel processed separately)
  - Input: [B, T, N]; forward(x_enc, None, None, None)

Usage:
    python timemixer_prediction.py \
        --dataset-path /scratch/.../monash_australian_electricity \
        --context-lengths 336,672 \
        --prediction-lengths 48,96,336 \
        --num-test-windows 8
"""

import argparse
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# ── Official-faithful TimeMixer components ────────────────────────────────────

class _SeriesDecomp(nn.Module):
    """Moving-average series decomposition (mirrors series_decomp in Autoformer_EncDec.py).

    Input/output: [B, T, C].
    """

    def __init__(self, kernel_size: int):
        super().__init__()
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Symmetric padding on the time axis
        front = x[:, :1, :].expand(-1, (self.kernel_size - 1) // 2, -1)
        end   = x[:, -1:, :].expand(-1, (self.kernel_size - 1) // 2, -1)
        x_pad = torch.cat([front, x, end], dim=1)           # [B, T+pad, C]
        # AvgPool1d operates on [B, C, T]; permute in/out
        x_trend  = self.avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)  # [B, T, C]
        x_season = x - x_trend
        return x_season, x_trend


class _Normalize(nn.Module):
    """RevIN-style learnable instance normalization (mirrors StandardNorm.Normalize).

    Stores per-(batch, channel) mean/stdev from the 'norm' call so that
    a subsequent 'denorm' call can undo the transformation.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias   = nn.Parameter(torch.zeros(num_features))
        self._mean:  Optional[torch.Tensor] = None
        self._stdev: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == 'norm':
            # x: [B, T, C] — compute stats over T
            self._mean  = x.mean(dim=1, keepdim=True).detach()
            self._stdev = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
            x = (x - self._mean) / self._stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps ** 2)
            x = x * self._stdev + self._mean
        else:
            raise ValueError(f"Unknown mode: {mode!r}")
        return x


class _TokenEmbedding(nn.Module):
    """Conv1d value embedding (mirrors TokenEmbedding in Embed.py).

    Projects [B, T, c_in] → [B, T, d_model] via a kernel-3 circular conv.
    """

    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(
            c_in, d_model, kernel_size=3, padding=1,
            padding_mode='circular', bias=False,
        )
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, c_in] → [B, c_in, T] → conv → [B, d_model, T] → [B, T, d_model]
        return self.conv(x.permute(0, 2, 1)).transpose(1, 2)


class _MultiScaleSeasonMixing(nn.Module):
    """Bottom-up mixing of seasonal patterns: fine scale informs coarser scales.

    Mirrors MultiScaleSeasonMixing in the official code.
    Each layer: Linear(T//w^i → T//w^(i+1)) + GELU + Linear(same → same).
    """

    def __init__(self, seq_len: int, down_sampling_layers: int, down_sampling_window: int):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(seq_len // (down_sampling_window ** i),
                          seq_len // (down_sampling_window ** (i + 1))),
                nn.GELU(),
                nn.Linear(seq_len // (down_sampling_window ** (i + 1)),
                          seq_len // (down_sampling_window ** (i + 1))),
            )
            for i in range(down_sampling_layers)
        ])

    def forward(self, season_list: List[torch.Tensor]) -> List[torch.Tensor]:
        # Inputs: season_list[i] = [B, d_model, T//w^i]  (already permuted)
        out_high = season_list[0]
        out_low  = season_list[1]
        out = [out_high.permute(0, 2, 1)]   # [B, T_fine, d_model]

        for i in range(len(season_list) - 1):
            res     = self.layers[i](out_high)          # [B, d_model, T_coarser]
            out_low = out_low + res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out.append(out_high.permute(0, 2, 1))       # [B, T_scale, d_model]

        return out     # list of [B, T//w^i, d_model], same order as input


class _MultiScaleTrendMixing(nn.Module):
    """Top-down mixing of trend patterns: coarse scale informs finer scales.

    Mirrors MultiScaleTrendMixing in the official code.
    Each layer: Linear(T//w^(i+1) → T//w^i) + GELU + Linear(same → same).
    """

    def __init__(self, seq_len: int, down_sampling_layers: int, down_sampling_window: int):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(seq_len // (down_sampling_window ** (i + 1)),
                          seq_len // (down_sampling_window ** i)),
                nn.GELU(),
                nn.Linear(seq_len // (down_sampling_window ** i),
                          seq_len // (down_sampling_window ** i)),
            )
            for i in reversed(range(down_sampling_layers))
        ])

    def forward(self, trend_list: List[torch.Tensor]) -> List[torch.Tensor]:
        # Inputs: trend_list[i] = [B, d_model, T//w^i]  (already permuted)
        trend_rev = list(reversed(trend_list))
        out_low  = trend_rev[0]   # coarsest
        out_high = trend_rev[1]
        out = [out_low.permute(0, 2, 1)]    # [B, T_coarsest, d_model]

        for i in range(len(trend_rev) - 1):
            res      = self.layers[i](out_low)          # [B, d_model, T_finer]
            out_high = out_high + res
            out_low  = out_high
            if i + 2 <= len(trend_rev) - 1:
                out_high = trend_rev[i + 2]
            out.append(out_low.permute(0, 2, 1))        # [B, T_scale, d_model]

        out.reverse()
        return out     # list of [B, T//w^i, d_model], fine-to-coarse order (matches season)


class _PDMBlock(nn.Module):
    """Past Decomposable Mixing block — channel-independence=1 path.

    Mirrors PastDecomposableMixing in the official code.
    decompose → mix season (bottom-up) + trend (top-down) → residual + out_cross_layer.
    """

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        d_ff: int,
        down_sampling_layers: int,
        down_sampling_window: int,
        decomp_kernel: int,
    ):
        super().__init__()
        self.decomp = _SeriesDecomp(decomp_kernel)
        self.mix_season = _MultiScaleSeasonMixing(seq_len, down_sampling_layers, down_sampling_window)
        self.mix_trend  = _MultiScaleTrendMixing(seq_len, down_sampling_layers, down_sampling_window)
        self.out_cross  = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x_list: List[torch.Tensor]) -> List[torch.Tensor]:
        # x_list[i]: [B, T//w^i, d_model]
        lengths = [x.size(1) for x in x_list]

        season_list, trend_list = [], []
        for x in x_list:
            s, t = self.decomp(x)
            season_list.append(s.permute(0, 2, 1))   # [B, d_model, T//w^i]
            trend_list.append(t.permute(0, 2, 1))

        out_season = self.mix_season(season_list)     # [B, T//w^i, d_model] each
        out_trend  = self.mix_trend(trend_list)

        out_list = []
        for orig, os, ot, T in zip(x_list, out_season, out_trend, lengths):
            mixed = os + ot                                    # [B, T//w^i, d_model]
            out   = orig + self.out_cross(mixed)               # residual connection
            out_list.append(out[:, :T, :])
        return out_list


class TimeMixer(nn.Module):
    """
    TimeMixer forecasting model — channel-independence mode.

    Matches the official Model class from https://github.com/kwuking/TimeMixer
    for the long_term_forecast task with channel_independence=1.

    Args:
        seq_len: Context (input) length.
        pred_len: Forecast horizon.
        enc_in: Number of input channels.
        d_model: Token embedding dimension (default 16 per official defaults).
        d_ff: PDM feedforward dimension (default 32).
        e_layers: Number of stacked PDM blocks (default 2).
        down_sampling_layers: Number of additional coarser scales beyond the original.
        down_sampling_window: Temporal pooling factor between adjacent scales.
        decomp_kernel: Moving-average kernel for seasonal-trend decomposition.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int = 1,
        d_model: int = 16,
        d_ff: int = 32,
        e_layers: int = 2,
        down_sampling_layers: int = 2,
        down_sampling_window: int = 2,
        decomp_kernel: int = 25,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len   = seq_len
        self.pred_len  = pred_len
        self.enc_in    = enc_in
        self.down_sampling_window = down_sampling_window
        self.num_scales = down_sampling_layers + 1  # number of temporal scales

        # Per-scale RevIN normalisation (one per scale)
        self.normalize_layers = nn.ModuleList([
            _Normalize(enc_in, affine=True)
            for _ in range(self.num_scales)
        ])

        # Channel-independence embedding: projects [B*N, T, 1] → [B*N, T, d_model]
        self.enc_embedding = _TokenEmbedding(c_in=1, d_model=d_model)
        self.dropout       = nn.Dropout(dropout)

        # Pre-processing decomposition (for ci=1 the pre_enc just returns the list)
        self.preprocess = _SeriesDecomp(decomp_kernel)

        # Stacked PDM blocks
        self.pdm_blocks = nn.ModuleList([
            _PDMBlock(seq_len, d_model, d_ff, down_sampling_layers, down_sampling_window, decomp_kernel)
            for _ in range(e_layers)
        ])

        # FMM: one linear predictor per scale + a shared projection to 1 output channel
        scale_lens = [seq_len // (down_sampling_window ** i) for i in range(self.num_scales)]
        self.predict_layers = nn.ModuleList([
            nn.Linear(scale_lens[i], pred_len)
            for i in range(self.num_scales)
        ])
        self.projection = nn.Linear(d_model, 1, bias=True)

    def _multi_scale_inputs(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Downsample x ([B, T, N]) into a list of [B, T//w^i, N] tensors."""
        down_pool = nn.AvgPool1d(self.down_sampling_window)
        x_list = [x]
        xi = x.permute(0, 2, 1)   # [B, N, T]
        for _ in range(self.num_scales - 1):
            xi = down_pool(xi)
            x_list.append(xi.permute(0, 2, 1))
        return x_list

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ) -> torch.Tensor:
        """
        Args:
            x_enc: [B, T, N] input context (temporal features ignored).
        Returns:
            [B, pred_len, N] forecast.
        """
        B, T, N = x_enc.shape

        # Multi-scale downsampling
        x_list = self._multi_scale_inputs(x_enc)   # list of [B, T//w^i, N]

        # Per-scale normalisation + channel-independence reshape
        enc_list = []
        for i, x in enumerate(x_list):
            x = self.normalize_layers[i](x, 'norm')           # [B, T//w^i, N]
            # Channel-independence: treat each channel as an independent series
            _, Ti, _ = x.shape
            x = x.permute(0, 2, 1).reshape(B * N, Ti, 1)      # [B*N, T//w^i, 1]
            enc_list.append(x)

        # Token embedding at each scale
        emb_list = [self.dropout(self.enc_embedding(x)) for x in enc_list]  # [B*N, T//w^i, d_model]

        # Stacked PDM blocks
        for block in self.pdm_blocks:
            emb_list = block(emb_list)

        # FMM: predict from each scale, reshape, project to 1 channel, sum
        dec_out_list = []
        for i, enc_out in enumerate(emb_list):
            # enc_out: [B*N, T//w^i, d_model]
            # predict_layers[i]: Linear(T//w^i → pred_len) applied to permuted (time) dim
            dec = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
            # dec: [B*N, pred_len, d_model]
            dec = self.projection(dec)                          # [B*N, pred_len, 1]
            dec = dec.reshape(B, N, self.pred_len).permute(0, 2, 1)   # [B, pred_len, N]
            dec_out_list.append(dec)

        # Sum across scales (FMM)
        dec_out = torch.stack(dec_out_list, dim=-1).sum(dim=-1)  # [B, pred_len, N]

        # Denormalize with scale-0 (finest) statistics
        dec_out = self.normalize_layers[0](dec_out, 'denorm')

        return dec_out   # [B, pred_len, N]


# ── Data helpers ─────────────────────────────────────────────────────────────────

def parse_int_list(raw: str) -> List[int]:
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    if any(v <= 0 for v in values):
        raise ValueError("All lengths must be positive integers.")
    return sorted(set(values))


def is_non_scalar_sequence(x) -> bool:
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x)
        return arr.ndim > 0 and arr.size > 1
    return False


def to_scalar(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x)
        if arr.size == 0:
            return np.nan
        if arr.size == 1:
            return arr.reshape(-1)[0]
        return np.nan
    return x


def load_long_dataframe(
    context_df: pd.DataFrame,
    id_column: str = "id",
    timestamp_column: str = "timestamp",
    target_column: str = "target",
) -> pd.DataFrame:
    required_cols = {id_column, timestamp_column, target_column}
    missing = required_cols - set(context_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    base = context_df[[id_column, timestamp_column, target_column]].rename(
        columns={id_column: "id", timestamp_column: "timestamp", target_column: "target"}
    )
    base = base.explode(["timestamp", "target"]).reset_index(drop=True)

    if base["target"].map(is_non_scalar_sequence).any():
        vectors = base["target"].apply(
            lambda x: np.asarray(x).reshape(-1)
            if isinstance(x, (list, tuple, np.ndarray))
            else np.asarray([x])
        )
        expanded = pd.DataFrame(vectors.tolist(), index=base.index)
        df = base[["id", "timestamp"]].join(expanded)
        df = df.melt(id_vars=["id", "timestamp"], var_name="channel", value_name="target")
        df["id"] = df["id"].astype(str) + "_ch" + df["channel"].astype(int).astype(str)
        df = df.drop(columns=["channel"])
    else:
        df = base
        df["target"] = df["target"].apply(to_scalar)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df = df.dropna(subset=["id", "timestamp", "target"])
    df["target"] = df["target"].astype("float32")
    return df


# ── Sliding-window dataset ────────────────────────────────────────────────────────

def make_windows(
    series: np.ndarray,
    context_len: int,
    pred_len: int,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Slice a 1-D series into (context, target) windows.

    Returns arrays of shape [n_windows, context_len] and [n_windows, pred_len].
    """
    total = context_len + pred_len
    if len(series) < total:
        return np.empty((0, context_len), dtype=np.float32), np.empty((0, pred_len), dtype=np.float32)
    xs, ys = [], []
    for start in range(0, len(series) - total + 1, stride):
        xs.append(series[start: start + context_len])
        ys.append(series[start + context_len: start + total])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


# ── Training ──────────────────────────────────────────────────────────────────────

def train_model(
    model: TimeMixer,
    x_train: np.ndarray,
    y_train: np.ndarray,
    num_epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> TimeMixer:
    """Train TimeMixer on (context, target) windows.

    Data is fed as [B, T, 1] (channel-independence mode, 1 channel per window).
    x_train and y_train are expected to be already z-score normalised with the
    context statistics, so loss is in normalised space (expected O(1)).
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=num_epochs,
        steps_per_epoch=max(1, math.ceil(len(x_train) / batch_size)),
    )
    criterion = nn.MSELoss()

    # Shape: [N, T] → [N, T, 1] for the channel-independence interface
    x_t = torch.tensor(x_train[:, :, None], dtype=torch.float32)   # [N, T, 1]
    y_t = torch.tensor(y_train[:, :, None], dtype=torch.float32)   # [N, P, 1]
    loader = DataLoader(TensorDataset(x_t, y_t), batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)   # [B, T, 1], [B, P, 1]
            optimizer.zero_grad()
            pred = model(xb)                          # [B, P, 1] in normalised space
            loss = criterion(pred, yb)                # MSE in normalised space
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item() * len(xb)
        if epoch == 1 or epoch % max(1, num_epochs // 5) == 0:
            avg = epoch_loss / len(x_train)
            print(f"    epoch {epoch}/{num_epochs}  train_mse_norm={avg:.4f}")

    return model


# ── Evaluation ────────────────────────────────────────────────────────────────────

def compute_mase(
    pred: np.ndarray,
    target: np.ndarray,
    train_series: np.ndarray,
    seasonal_period: int,
) -> float:
    if len(train_series) <= seasonal_period:
        return float("nan")
    naive_errors = np.abs(train_series[seasonal_period:] - train_series[:-seasonal_period])
    scale = naive_errors.mean()
    if scale < 1e-8:
        return float("nan")
    return float(np.abs(pred - target).mean() / scale)


def evaluate_rolling(
    model: TimeMixer,
    full_series_scaled: np.ndarray,
    context_len: int,
    pred_len: int,
    num_windows: int,
    seasonal_period: int,
    device: torch.device,
) -> Optional[dict]:
    """Evaluate on rolling test windows.

    full_series_scaled is already StandardScaler-normalised (fit on training portion).
    MSE/MAE are reported in that standardised space — matching how papers compute metrics.
    """
    model.eval()
    required = context_len + num_windows * pred_len
    if len(full_series_scaled) < required:
        return None

    train_cutoff = len(full_series_scaled) - num_windows * pred_len

    all_mse, all_mae = [], []
    with torch.no_grad():
        for w in range(num_windows):
            ctx_start = train_cutoff - context_len + w * pred_len
            ctx = full_series_scaled[ctx_start: ctx_start + context_len]
            tgt = full_series_scaled[ctx_start + context_len: ctx_start + context_len + pred_len]

            x = torch.tensor(ctx[None, :, None], dtype=torch.float32).to(device)  # [1, T, 1]
            pred = model(x).squeeze(0).squeeze(-1).cpu().numpy()   # [pred_len] in scaled space

            L = min(len(pred), len(tgt))
            pred, tgt_t = pred[:L], tgt[:L]

            all_mse.append(float(np.mean((pred - tgt_t) ** 2)))
            all_mae.append(float(np.mean(np.abs(pred - tgt_t))))

    return {
        "mse_mean": float(np.mean(all_mse)),
        "mse_std":  float(np.std(all_mse)),
        "mae_mean": float(np.mean(all_mae)),
        "mae_std":  float(np.std(all_mae)),
    }


# ── Grid runner ───────────────────────────────────────────────────────────────────

def run_grid(
    df_long: pd.DataFrame,
    context_lengths: List[int],
    prediction_lengths: List[int],
    num_test_windows: int,
    seasonal_period: int,
    args: argparse.Namespace,
    device: torch.device,
) -> pd.DataFrame:
    series_by_id = {sid: grp["target"].values for sid, grp in df_long.groupby("id")}
    lengths = {sid: len(v) for sid, v in series_by_id.items()}

    rows = []
    for pred_len in prediction_lengths:
        for ctx_len in context_lengths:
            # TimeMixer requires seq_len divisible by down_sampling_window^down_sampling_layers
            divisor = args.down_sampling_window ** args.down_sampling_layers
            if ctx_len % divisor != 0:
                print(
                    f"[SKIP] ctx_len={ctx_len} not divisible by "
                    f"down_sampling_window^down_sampling_layers={divisor}"
                )
                continue

            min_required = ctx_len + num_test_windows * pred_len
            valid_ids = [sid for sid, l in lengths.items() if l >= min_required]
            if not valid_ids:
                print(f"[SKIP] pred_len={pred_len}, ctx_len={ctx_len}: no series long enough (need {min_required})")
                continue

            print(f"\n[RUN] pred_len={pred_len}, ctx_len={ctx_len}, series={len(valid_ids)}")

            # ── StandardScaler per series, fitted on training portion only ──────
            # Mirrors the official data_loader.py: scaler.fit(train_data) then
            # scaler.transform(all_data).  MSE is then in this standardised space,
            # which is what papers report.
            all_x, all_y = [], []
            scaled_series: dict[str, np.ndarray] = {}
            for sid in valid_ids:
                s = series_by_id[sid]
                train_cutoff = len(s) - num_test_windows * pred_len
                train_part = s[:train_cutoff]

                s_mean = float(train_part.mean())
                s_std  = max(float(train_part.std()), 1e-5)
                s_scaled = (s - s_mean) / s_std          # entire series, scaled with train stats
                scaled_series[sid] = s_scaled

                x, y = make_windows(s_scaled[:train_cutoff], ctx_len, pred_len, stride=pred_len)
                if len(x) > 0:
                    all_x.append(x)
                    all_y.append(y)

            if not all_x:
                print("[SKIP] No training windows available.")
                continue

            x_train = np.concatenate(all_x, axis=0)   # already in standardised space
            y_train = np.concatenate(all_y, axis=0)
            print(f"  Training windows: {len(x_train):,}")

            model = TimeMixer(
                seq_len=ctx_len,
                pred_len=pred_len,
                enc_in=1,
                d_model=args.d_model,
                d_ff=args.d_ff,
                e_layers=args.e_layers,
                down_sampling_layers=args.down_sampling_layers,
                down_sampling_window=args.down_sampling_window,
                decomp_kernel=args.decomp_kernel,
                dropout=args.dropout,
            )
            model = train_model(
                model, x_train, y_train,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=device,
            )

            mses, maes = [], []
            for sid in valid_ids:
                metrics = evaluate_rolling(
                    model, scaled_series[sid], ctx_len, pred_len,
                    num_test_windows, seasonal_period, device,
                )
                if metrics is not None:
                    mses.append(metrics["mse_mean"])
                    maes.append(metrics["mae_mean"])

            if not mses:
                continue

            row = {
                "context_length":    ctx_len,
                "prediction_length": pred_len,
                "num_series":        len(mses),
                "mse_mean":          float(np.mean(mses)),
                "mse_std":           float(np.std(mses)),
                "mae_mean":          float(np.mean(maes)),
                "mae_std":           float(np.std(maes)),
            }
            rows.append(row)
            print(f"  MSE={row['mse_mean']:.4f}  MAE={row['mae_mean']:.4f}")

    if not rows:
        raise RuntimeError("No valid (context_length, prediction_length) configuration was evaluated.")
    return pd.DataFrame(rows).sort_values(["prediction_length", "context_length"])


# ── CLI ───────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TimeMixer rolling-window forecasting baseline.")

    # Data
    parser.add_argument(
        "--dataset-path", "--dataset_path", dest="dataset_path", type=str,
        default="/scratch/Projects/rl_timeseries/pretraining_data/multivariate/"
                "chronos_datasets/monash_australian_electricity",
    )
    parser.add_argument("--parquet-filename", type=str, default="train-00000-of-00001.parquet")
    parser.add_argument("--id-column",        type=str, default="id")
    parser.add_argument("--timestamp-column", type=str, default="timestamp")
    parser.add_argument("--target-column",    type=str, default="target")

    # Evaluation
    parser.add_argument("--context-lengths",   type=str, default="336,672")
    parser.add_argument("--prediction-lengths", type=str, default="48,96,336")
    parser.add_argument("--num-test-windows",  type=int, default=8)
    parser.add_argument("--seasonal-period",   type=int, default=48,
                        help="For MASE: half-hourly data → 48-step daily cycle")

    # Model architecture (official defaults from run.py)
    parser.add_argument("--d-model",              type=int, default=16)
    parser.add_argument("--d-ff",                 type=int, default=32)
    parser.add_argument("--e-layers",             type=int, default=2)
    parser.add_argument("--down-sampling-layers", type=int, default=2)
    parser.add_argument("--down-sampling-window", type=int, default=2)
    parser.add_argument("--decomp-kernel",        type=int, default=25)
    parser.add_argument("--dropout",              type=float, default=0.1)

    # Training
    parser.add_argument("--num-epochs",  type=int,   default=10)
    parser.add_argument("--batch-size",  type=int,   default=256)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--device",      type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    # Output
    parser.add_argument("--grid-out",  type=str, default="timemixer_grid.csv")
    parser.add_argument("--pivot-out", type=str, default="timemixer_pivot.csv")

    args = parser.parse_args()
    # Map hyphenated CLI names to underscored attribute names expected by run_grid/TimeMixer
    args.down_sampling_layers = args.down_sampling_layers
    args.down_sampling_window = args.down_sampling_window
    args.d_model = args.d_model
    args.d_ff    = args.d_ff

    if args.num_test_windows < 2:
        raise ValueError("--num-test-windows must be >= 2 for stable rolling-window evaluation.")

    device = torch.device(args.device)
    context_lengths   = parse_int_list(args.context_lengths)
    prediction_lengths = parse_int_list(args.prediction_lengths)

    parquet_path = Path(args.dataset_path) / args.parquet_filename
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")
    print(f"Loading dataset from: {parquet_path}")

    context_df = pd.read_parquet(parquet_path)
    df_long = load_long_dataframe(
        context_df,
        id_column=args.id_column,
        timestamp_column=args.timestamp_column,
        target_column=args.target_column,
    )
    print(f"  {len(df_long):,} rows, {df_long['id'].nunique():,} series")

    results = run_grid(
        df_long=df_long,
        context_lengths=context_lengths,
        prediction_lengths=prediction_lengths,
        num_test_windows=args.num_test_windows,
        seasonal_period=args.seasonal_period,
        args=args,
        device=device,
    )

    print("\nGrid Results:\n")
    print(results.to_string(index=False))

    mse_pivot = results.pivot(index="context_length", columns="prediction_length", values="mse_mean")
    mae_pivot = results.pivot(index="context_length", columns="prediction_length", values="mae_mean")

    print("\nMSE (standardised, comparable to paper numbers):\n")
    print(mse_pivot)
    print("\nMAE (standardised):\n")
    print(mae_pivot)

    results.to_csv(args.grid_out, index=False)
    mse_pivot.to_csv(args.pivot_out)
    print(f"\nSaved grid:  {args.grid_out}")
    print(f"Saved pivot: {args.pivot_out}")


if __name__ == "__main__":
    main()
