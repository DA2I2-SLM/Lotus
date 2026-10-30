# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

# constants.py

import getpass
import os

# Path to this file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Root for the downstream raw data (HAR, Sleep-EDF, ECG-QA, M4, TSQA, ...).
# Auto-downloaded corpora land here. Read from LOTUS_DATA_DIR; the fallback
# mirrors the default that bashfiles/_paths.sh exports.
RAW_DATA = os.environ.get("LOTUS_DATA_DIR") or os.path.join(
    os.environ.get("LOTUS_SCRATCH", f"/scratch/{getpass.getuser()}"),
    "Lotus",
    "data",
)


def get_chatts_dir() -> str:
    """Return the ChatTS corpus root (LOTUS_CHATTS_DIR).

    There is deliberately no default: the corpus is not auto-downloaded, so a
    guessed path would only fail later with a less helpful message. Raise at
    the point of use instead of at import time so code paths that do not
    touch ChatTS keep working with the variable unset.
    """
    chatts_dir = os.environ.get("LOTUS_CHATTS_DIR", "").strip()
    if not chatts_dir:
        raise RuntimeError(
            "LOTUS_CHATTS_DIR is not set. Point it at the ChatTS corpus root "
            "(the directory holding sft/, ift/ and process/align_256), e.g. "
            "export LOTUS_CHATTS_DIR=/path/to/ChatTS"
        )
    if not os.path.isdir(chatts_dir):
        raise FileNotFoundError(
            f"LOTUS_CHATTS_DIR={chatts_dir!r} does not exist or is not a directory."
        )
    return chatts_dir
