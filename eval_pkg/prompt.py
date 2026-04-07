"""
Prompt construction and output parsing for Sudoku / 2D grid problems.
Matches the format used in the NineGrid benchmark notebook.
"""
from __future__ import annotations

import json
from typing import Any

from .eval_types import InferenceMode, Problem


class PromptBuilder:
    """
    Serialises puzzles as Python list-of-lists (0 = blank) — the same
    representation the dataset uses. This avoids tokenisation artifacts
    that arise with space-separated plain-text grids.
    """

    @staticmethod
    def grid_to_list(grid: list[list[Any]]) -> list[list[int]]:
        """Convert internal grid (None = blank) to list-of-lists with 0 for blanks."""
        return [[0 if v is None else int(v) for v in row] for row in grid]

    @classmethod
    def build(
        cls,
        problem: Problem,
        mode: InferenceMode,
        few_shot_examples: list[Problem] | None = None,
    ) -> str:
        puzzle_as_list = cls.grid_to_list(problem.grid)

        few_shot_prefix = ""
        if mode == InferenceMode.FEW_SHOT and few_shot_examples:
            examples_block = ""
            for ex in few_shot_examples:
                ex_puzzle = cls.grid_to_list(ex.grid)
                examples_block += (
                    f"\nExample:\nPuzzle:\n{ex_puzzle}\n\n"
                    f"SOLUTION (follow the solution structure above)\n{ex.solution}\n"
                )
            few_shot_prefix = (
                f"Here are some solved examples for reference:{examples_block}\n"
                "Now solve the following:\n"
            )

        return (
            f"{few_shot_prefix}"
            "Solve the following Sudoku puzzle.\n"
            "The following grid is a 9x9 list of lists. Empty cells are represented by 0. \n"
            "INSTRUCTION: Fill only the cells containing 0. Preserving the initial values is mandatory while following all Sudoku rules.\n"
            f"\nPuzzle:\n{puzzle_as_list}\n\n"
            "Return ONLY one solution, NO additional words, NO code, NO thinking steps or similar ONLY ONE solution in 9x9 list of lists completely solved.\n"
            "\nSOLUTION STRUCTURE\n"
            "[[row1],[row2],[row3],[row4],[row5],[row6],[row7],[row8],[row9]]\n"
            "\nSOLUTION (follow the solution structure above)\n"
        )


class GridParser:
    """
    Parses a model's text output back into a 2-D list.

    Mirrors the NineGrid notebook's parse_response: scan the full output for
    all [[...]] candidates and return the last valid N×N grid. Taking the
    *last* candidate handles models that emit chain-of-thought grids before
    their final answer.
    """

    @staticmethod
    def parse(text: str, expected_rows: int, expected_cols: int) -> list[list[Any]] | None:
        candidates = []
        start = 0
        while True:
            pos = text.find("[[", start)
            if pos == -1:
                break
            end = text.find("]]", pos)
            if end == -1:
                break
            candidates.append(text[pos : end + 2])
            start = pos + 1

        last_valid = None
        for candidate in candidates:
            try:
                grid = json.loads(candidate)
                if (
                    len(grid) == expected_rows
                    and all(len(r) == expected_cols for r in grid)
                ):
                    last_valid = grid
            except Exception:
                continue

        return last_valid