# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

from abc import abstractmethod
import torch
import torch.nn as nn
from opentslm.model_config import TRANSFORMER_INPUT_DIM, PATCH_SIZE


class TimeSeriesDecoderBase(nn.Module):
    """
    Base class for time series decoders.
    
    Decoders map from latent representations back to time series space.
    """
    
    def __init__(
        self,
        input_dim: int = TRANSFORMER_INPUT_DIM,
        patch_size: int = PATCH_SIZE,
        dropout: float = 0.0,
    ):
        """
        Args:
            input_dim: Dimension of input latent representation
            patch_size: Size of time series patches to reconstruct
            dropout: Dropout probability
        """
        super().__init__()
        self.input_dim = input_dim
        self.patch_size = patch_size
        self.dropout = dropout

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to time series.
        
        Args:
            x: Latent representation from encoder
            
        Returns:
            Reconstructed time series
        """
        pass