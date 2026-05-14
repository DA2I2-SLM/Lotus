# Copyright 2024 Tsinghua University and ByteDance.
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/license/mit
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import openai
import base64
import matplotlib.pyplot as plt
from loguru import logger
import numpy as np
import json
import os
from tqdm import tqdm
import traceback
from typing import *
import io
import sys

import getpass
USER_NAME = getpass.getuser()
sys.path.append(f"/home/{USER_NAME}/FoundationalTS")

from chatts.evaluate_qa import evaluate_batch_qa

# CONFIG
EXP = 'chatts_dataset_a'
DATASET = f'/scratch/{USER_NAME}/FoundationalTS/evaluation/dataset/dataset_a.json'


if __name__ == '__main__':
    dataset = json.load(open(DATASET))

    # Load from existing generation
    generated_answer = [{} for _ in range(len(dataset))]
    for file in os.listdir(f"exp/{EXP}"):
        if 'generated_answer' in file and file.endswith('.json'):
            cur_answer = json.load(open(os.path.join(f"exp/{EXP}", file)))
            for ans in cur_answer:
                generated_answer[ans['idx']] = ans
    
    # Evaluation
    evaluate_batch_qa(dataset, generated_answer, EXP, num_workers=1) # num_workers should be 16