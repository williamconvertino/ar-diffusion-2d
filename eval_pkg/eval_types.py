"""
Shared data types: enums, dataclasses, and the ProblemSet protocol.
"""
from __future__ import annotations
 
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable
 
 
# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
 
class InferenceMode(str, Enum):
    ZERO_SHOT = "zero_shot"
    FEW_SHOT  = "few_shot"
 
 
class ModelFamily(str, Enum):
    LLAMA = "llama"
    LLADA = "llada"
 
 
class Difficulty(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"
 
 
# ---------------------------------------------------------------------------
# Core data classes
# ---------------------------------------------------------------------------
 
@dataclass
class Problem:
    """A single 2D spatial-reasoning problem."""
    problem_id: str
    grid: list[list[Any]]       # 2-D list, None = masked cell
    solution: list[list[Any]]   # full ground-truth grid
    metadata: dict[str, Any] = field(default_factory=dict)
 
    @property
    def rows(self) -> int:
        return len(self.grid)
 
    @property
    def cols(self) -> int:
        return len(self.grid[0]) if self.grid else 0
 
 
@dataclass
class EvalResult:
    """Metrics for a single problem."""
    problem_id: str
    predicted: list[list[Any]] | None
    raw_output: str
 
    exact_match: bool = False
    cell_accuracy: float = 0.0
    row_accuracy: float = 0.0
    col_accuracy: float = 0.0
    format_valid: bool = False
    inference_time_s: float = 0.0
    perplexity: float | None = None
    position_accuracy: dict[tuple[int, int], bool] = field(default_factory=dict)
 
 
@dataclass
class AggregateMetrics:
    """Dataset-level aggregation of EvalResult objects."""
    n_problems: int
    exact_match_rate: float
    mean_cell_accuracy: float
    mean_row_accuracy: float
    mean_col_accuracy: float
    format_validity_rate: float
    mean_inference_time_s: float
    mean_perplexity: float | None
    position_accuracy_map: dict[tuple[int, int], float]
    per_problem: list[EvalResult]
 
    def summary(self) -> str:
        lines = [
            f"Problems evaluated : {self.n_problems}",
            f"Exact-match rate   : {self.exact_match_rate:.3f}",
            f"Cell accuracy      : {self.mean_cell_accuracy:.3f}",
            f"Row accuracy       : {self.mean_row_accuracy:.3f}",
            f"Col accuracy       : {self.mean_col_accuracy:.3f}",
            f"Format validity    : {self.format_validity_rate:.3f}",
            f"Mean infer. time   : {self.mean_inference_time_s:.2f}s",
        ]
        if self.mean_perplexity is not None:
            lines.append(f"Mean perplexity    : {self.mean_perplexity:.2f}")
        return "\n".join(lines)
 
 
# ---------------------------------------------------------------------------
# ProblemSet protocol
# ---------------------------------------------------------------------------
 
@runtime_checkable
class ProblemSet(Protocol):
    """Minimal interface any problem set must satisfy."""
 
    @property
    def name(self) -> str: ...
 
    def __len__(self) -> int: ...
 
    def __iter__(self): ...
 
    def few_shot_examples(self, k: int = 3) -> list[Problem]: ...