# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

# constants.py

import os

# Path to this file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to raw data directory — kept on /scratch to avoid filling /home quota
RAW_DATA = "/scratch/s223540177/FoundationalTS/data"
