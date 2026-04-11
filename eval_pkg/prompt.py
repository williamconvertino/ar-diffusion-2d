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
        rows = ",".join(f"[row{i+1}]" for i in range(problem.rows))
        return (
            f"{few_shot_prefix}"
            "Solve the following Sudoku puzzle.\n"
            f"The following grid is a {problem.rows}x{problem.cols} list of lists. Empty cells are represented by 0. \n"
            "INSTRUCTION: Fill only the cells containing 0. Preserving the initial values is mandatory while following all Sudoku rules.\n"
            f"\nPuzzle:\n{puzzle_as_list}\n\n"
            f"Return ONLY one solution, NO additional words, NO code, NO thinking steps or similar ONLY ONE solution in {problem.rows}x{problem.cols} list of lists completely solved.\n"
            "\nSOLUTION STRUCTURE\n"
            # "[[row1],[row2],[row3],[row4],[row5],[row6],[row7],[row8],[row9]]\n"
            f"[{rows}]\n"
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


# ---------------------------------------------------------------------------
# Grid visualisation
# ---------------------------------------------------------------------------

def render_grid(grid: list[list[Any]], title: str = "") -> str:
    """
    Render a 9×9 Sudoku grid with box borders, e.g.:

      ┌───────┬───────┬───────┐
      │ · 7 · │ · · · │ · 4 3 │
      ...
      └───────┴───────┴───────┘

    Parameters
    ----------
    grid  : 9×9 list-of-lists; None or 0 = empty cell (shown as ·)
    title : optional label printed above the grid
    """
    TOP    = "  ┌───────┬───────┬───────┐"
    MID    = "  ├───────┼───────┼───────┤"
    BOT    = "  └───────┴───────┴───────┘"
    lines  = []

    if title:
        lines.append(f"  {title}")

    for r, row in enumerate(grid):
        if r == 0:
            lines.append(TOP)
        elif r % 3 == 0:
            lines.append(MID)

        cells = []
        for c, val in enumerate(row):
            cells.append("·" if (val is None or val == 0) else str(val))

        line = "  │ {0} {1} {2} │ {3} {4} {5} │ {6} {7} {8} │".format(*cells)
        lines.append(line)

    lines.append(BOT)
    return "\n".join(lines)


def render_result(
    problem: "Problem",
    predicted: list[list[Any]] | None,
    show_diff: bool = True,
) -> str:
    """
    Print the puzzle and the model's predicted solution side by side,
    with incorrect cells marked with * when show_diff=True.

    Example output:

      PUZZLE                    PREDICTION  (* = wrong)
      ┌───────┬───────┬───────┐ ┌───────┬───────┬───────┐
      │ · 7 · │ · · · │ · 4 3 │ │ 6 7 9 │ 5 1 8 │ 2 4 3 │
      ...

    Parameters
    ----------
    problem   : Problem instance (has .grid and .solution)
    predicted : model output grid, or None if parsing failed
    show_diff : mark cells that differ from solution with *
    """
    TOP  = "┌───────┬───────┬───────┐"
    MID  = "├───────┼───────┼───────┤"
    BOT  = "└───────┴───────┴───────┘"
    SEP  = "   "   # gap between the two grids

    def row_str(row, solution_row=None, pred_row=None):
        """Build one │ ... │ ... │ ... │ line, optionally marking errors."""
        cells = []
        for c, val in enumerate(row):
            ch = "·" if (val is None or val == 0) else str(val)
            if show_diff and solution_row is not None and pred_row is not None:
                # mark wrong predicted cells
                sv = solution_row[c] if solution_row else None
                pv = pred_row[c]     if (pred_row and c < len(pred_row)) else None
                if pv is not None and sv is not None and pv != sv:
                    ch = "*"
            cells.append(ch)
        return "│ {0} {1} {2} │ {3} {4} {5} │ {6} {7} {8} │".format(*cells)

    puzzle    = problem.grid
    solution  = problem.solution

    if predicted is None:
        pred_label = "PREDICTION  (parse failed)"
        pred_grid  = [[0]*9 for _ in range(9)]
    elif show_diff:
        pred_label = "PREDICTION  (* = wrong)"
        pred_grid  = predicted
    else:
        pred_label = "PREDICTION"
        pred_grid  = predicted

    # header line — pad puzzle label to align with grid width (25 chars)
    puzzle_label = "PUZZLE"
    header = f"  {puzzle_label:<25}{SEP}{pred_label}"

    lines = [header]
    for r in range(9):
        if r == 0:
            lines.append(f"  {TOP}{SEP}{TOP}")
        elif r % 3 == 0:
            lines.append(f"  {MID}{SEP}{MID}")

        p_row = puzzle[r]   if r < len(puzzle)    else [0]*9
        s_row = solution[r] if r < len(solution)  else None
        d_row = pred_grid[r] if r < len(pred_grid) else None

        puzzle_line = row_str(p_row)
        pred_line   = row_str(
            d_row or [0]*9,
            solution_row=s_row,
            pred_row=d_row,
        )
        lines.append(f"  {puzzle_line}{SEP}{pred_line}")

    lines.append(f"  {BOT}{SEP}{BOT}")
    return "\n".join(lines)