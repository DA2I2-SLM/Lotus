# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""LOTUS: forecast-grounded time-series language model.

The training entry point is ``curriculum_learning.py`` at the repo root; the
model class is :class:`opentslm.model.llm.OpenTSLMSPFixPos2D.OpenTSLMSPFixPos2D`.
Submodules are imported lazily so that ``import opentslm`` stays cheap.
"""

__all__ = ["OpenTSLMSPFixPos2D"]


def __getattr__(name):
    if name == "OpenTSLMSPFixPos2D":
        from opentslm.model.llm.OpenTSLMSPFixPos2D import OpenTSLMSPFixPos2D
        return OpenTSLMSPFixPos2D
    raise AttributeError(f"module 'opentslm' has no attribute {name!r}")
