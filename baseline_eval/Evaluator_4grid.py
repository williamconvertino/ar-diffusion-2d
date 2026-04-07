"""
    Evaluator for 4_grid sudoku puzzle set
"""

import sys
# import os
import pandas as pd
from pathlib import Path

import re
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np

from Evaluatorall import DifficultyFilter, NineGrid, LlamaBackend, LladaBackend, GridEvaluator, InferenceMode

class FourGrid(NineGrid):

    name="FourGrid"
    _META_COLS = [
        "board_id",
        "root_hash",
        "solution_id",
        "size",
        "difficulty",
        "missing_cells",
    ]

if __name__ == "__main__":

    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    #define the difficulty filter
    diff = DifficultyFilter(
        missing_cells=(6, 8),
        # initial_resolution_rate=(0.0, 0.5),
        )

    data_path = Path(__file__).resolve() / '../../data' / 'four_grid.csv'
    # 1. Load the dataset
    dataset = NineGrid(
        parquet_path=str(data_path.resolve()),
        n_samples=5,                      # start small for a smoke-test
        difficulty=diff,
        few_shot_strategy="easiest",        # use simplest puzzles as demos
    )
    print(f"Loaded {len(dataset)} problems")
    print(dataset.difficulty_summary())

    evaluator = GridEvaluator(
        problem_set=dataset,
        inference_mode=InferenceMode.ZERO_SHOT,
        # model="meta-llama/Meta-Llama-3-8B-Instruct",
        model = "google/byt5-small",
        max_new_tokens=512,
        compute_perplexity=True,
    )
    print(f" \n Using {next(evaluator.backend._model.parameters()).device} \n")

    metrics = evaluator.evaluate()
    print(metrics.summary())

    failures = [r for r in metrics.per_problem if not r.exact_match]
    print(f"\n{len(failures)} failures out of {len(dataset)}")

    for r in failures[:3]:
        print(f"\n--- {r.problem_id} ---")
        print(f"  format_valid : {r.format_valid}")
        print(f"  cell_accuracy: {r.cell_accuracy:.2f}")
        print(f"  raw_output   : {r.raw_output[:200]!r}")

