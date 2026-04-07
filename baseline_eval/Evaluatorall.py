"""
Evaluator for LLaMA and LLaDA models on 2D spatial reasoning tasks (e.g. NineGrid).

Supports:
  - Problem sets: any object implementing the ProblemSet protocol
  - Inference modes: zero-shot, few-shot
  - Models: llama, llada (or any HuggingFace-compatible causal / masked-diffusion LM)
  - Metrics: exact match, cell accuracy, row/col accuracy, format validity,
             token perplexity (optional), position-wise accuracy
"""

from __future__ import annotations

import re
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & constants
# ---------------------------------------------------------------------------

class InferenceMode(str, Enum):
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"


class ModelFamily(str, Enum):
    LLAMA = "llama"
    LLADA = "llada"


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class Problem:
    """A single 2D spatial-reasoning problem."""
    problem_id: str
    grid: list[list[Any]]          # 2-D list (rows × cols), None = masked cell
    solution: list[list[Any]]      # full ground-truth grid
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
    predicted: list[list[Any]] | None   # parsed model output
    raw_output: str                      # raw text from the model

    exact_match: bool = False
    cell_accuracy: float = 0.0          # fraction of cells correct
    row_accuracy: float = 0.0           # fraction of rows fully correct
    col_accuracy: float = 0.0           # fraction of cols fully correct
    format_valid: bool = False          # did the output parse cleanly?
    inference_time_s: float = 0.0
    perplexity: float | None = None     # only if model exposes log-probs
    position_accuracy: dict[tuple[int,int], bool] = field(default_factory=dict)


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
    position_accuracy_map: dict[tuple[int,int], float]   # (r,c) → avg accuracy
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
# Problem-set protocol  (implement this for NineGrid, Sudoku, etc.)
# ---------------------------------------------------------------------------

@runtime_checkable
class ProblemSet(Protocol):
    """Minimal interface a problem set must satisfy."""

    @property
    def name(self) -> str: ...

    def __len__(self) -> int: ...

    def __iter__(self): ...           # yields Problem

    def few_shot_examples(self, k: int = 3) -> list[Problem]:
        """Return k representative solved problems for few-shot prompting."""
        ...


# ---------------------------------------------------------------------------
# Built-in: NineGrid problem set  (Kaggle benchmark by beta3logic)
# ---------------------------------------------------------------------------

class DifficultyFilter:
    """
    Convenience filter passed to NineGrid to restrict problems by difficulty.

    All thresholds are inclusive on both ends.  Any field left as None is
    unconstrained.

    Difficulty fields (all match the dataset column names):
      missing_cells          int   [17, 64]
      given_ratio            float [0.28, 0.95]
      naked_singles_count    int   [0, 26]
      hidden_singles_count   int   [0, 27]
      initial_resolution_rate float [0.0, 1.0]
    """
    def __init__(
        self,
        missing_cells: tuple[int, int] | None = None,
        given_ratio: tuple[float, float] | None = None,
        naked_singles_count: tuple[int, int] | None = None,
        hidden_singles_count: tuple[int, int] | None = None,
        initial_resolution_rate: tuple[float, float] | None = None,
    ):
        self.missing_cells = missing_cells
        self.given_ratio = given_ratio
        self.naked_singles_count = naked_singles_count
        self.hidden_singles_count = hidden_singles_count
        self.initial_resolution_rate = initial_resolution_rate

    def matches(self, meta: dict) -> bool:
        checks = [
            ("missing_cells", self.missing_cells),
            ("given_ratio", self.given_ratio),
            ("naked_singles_count", self.naked_singles_count),
            ("hidden_singles_count", self.hidden_singles_count),
            ("initial_resolution_rate", self.initial_resolution_rate),
        ]
        for key, rng in checks:
            if rng is not None and key in meta:
                lo, hi = rng
                if not (lo <= meta[key] <= hi):
                    return False
        return True


class NineGrid:
    """
    Loads the NineGrid Sudoku Reasoning Benchmark for LLMs from Kaggle
    (https://www.kaggle.com/benchmarks/beta3logic/ninegrid-sudoku-reasoning-benchmark-for-llms).

    Dataset schema
    --------------
    puzzle_grid              list[list[int]]   9×9, 0 = empty cell
    solution_grid            list[list[int]]   9×9, fully solved
    missing_cells            int               number of empty cells [17, 64]
    given_cells              int               81 − missing_cells
    given_ratio              float             given_cells / 81
    row_given_counts         list[int]         given cells per row (9 values)
    col_given_counts         list[int]         given cells per col (9 values)
    naked_singles_count      int               trivially-solvable empty cells
    hidden_singles_count     int               single-candidate-in-unit cells
    initial_resolution_rate  float             fraction solvable by singles alone

    Parameters
    ----------
    parquet_path : str
        Path to the downloaded dataset file.  Parquet is preferred (faster),
        but a CSV with the same columns is also accepted.
    n_samples : int | None
        Cap the number of problems loaded (useful for quick tests).
    difficulty : DifficultyFilter | None
        Optional filter on difficulty metadata fields.
    few_shot_strategy : str
        How to pick few-shot demos:
          "easiest"  – lowest missing_cells (most given clues)
          "random"   – uniform random sample (reproducible via seed)
          "hardest"  – highest missing_cells
    seed : int
        RNG seed used when few_shot_strategy="random".

    Download
    --------
    pip install kaggle
    kaggle datasets download beta3logic/ninegrid-sudoku-reasoning-benchmark-for-llms
    unzip ninegrid-sudoku-reasoning-benchmark-for-llms.zip
    """

    name = "NineGrid"

    # Metadata columns preserved verbatim from the dataset
    _META_COLS = [
        "missing_cells", "given_cells", "given_ratio",
        "row_given_counts", "col_given_counts",
        "naked_singles_count", "hidden_singles_count",
        "initial_resolution_rate",
    ]

    def __init__(
        self,
        parquet_path: str,
        n_samples: int | None = None,
        difficulty: DifficultyFilter | None = None,
        few_shot_strategy: str = "easiest",
        seed: int = 42,
    ):
        self._rng = np.random.default_rng(seed)
        self._few_shot_strategy = few_shot_strategy
        self._problems = self._load(parquet_path, n_samples, difficulty)
        if not self._problems:
            raise ValueError(
                "No problems loaded — check the file path and difficulty filter."
            )

    # ---- ProblemSet protocol ------------------------------------------------

    def __len__(self) -> int:
        return len(self._problems)

    def __iter__(self):
        return iter(self._problems)

    def few_shot_examples(self, k: int = 3) -> list[Problem]:
        """
        Return k demonstration problems for few-shot prompting.

        Strategy controls selection:
          "easiest" → fewest missing cells (clearest examples for the model)
          "hardest" → most missing cells
          "random"  → reproducible random sample
        """
        k = min(k, len(self._problems))
        if self._few_shot_strategy == "easiest":
            sorted_probs = sorted(
                self._problems,
                key=lambda p: p.metadata.get("missing_cells", 0)
            )
            return sorted_probs[:k]
        elif self._few_shot_strategy == "hardest":
            sorted_probs = sorted(
                self._problems,
                key=lambda p: p.metadata.get("missing_cells", 0),
                reverse=True,
            )
            return sorted_probs[:k]
        else:  # "random"
            indices = self._rng.choice(len(self._problems), size=k, replace=False)
            return [self._problems[i] for i in indices]

    # ---- difficulty helpers -------------------------------------------------

    def filter(self, difficulty: DifficultyFilter) -> "NineGrid":
        """Return a new NineGrid containing only problems matching the filter."""
        filtered = [p for p in self._problems if difficulty.matches(p.metadata)]
        clone = object.__new__(NineGrid)
        clone._rng = self._rng
        clone._few_shot_strategy = self._few_shot_strategy
        clone._problems = filtered
        return clone

    def difficulty_summary(self) -> dict:
        """Descriptive stats over difficulty metadata fields."""
        if not self._problems:
            return {}
        stats = {}
        for col in ["missing_cells", "given_ratio", "naked_singles_count",
                    "hidden_singles_count", "initial_resolution_rate"]:
            vals = [p.metadata[col] for p in self._problems if col in p.metadata]
            if vals:
                arr = np.array(vals, dtype=float)
                stats[col] = {
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                    "mean": float(arr.mean()),
                    "median": float(np.median(arr)),
                }
        return stats

    # ---- loading ------------------------------------------------------------

    def _load(
        self,
        path: str,
        n_samples: int | None,
        difficulty: DifficultyFilter | None,
    ) -> list[Problem]:
        import ast

        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("Install `pandas` (and `pyarrow` for parquet) to load NineGrid.") from e

        if path.endswith(".parquet"):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)

        problems = []
        for idx, row in df.iterrows():
            # --- parse grids -------------------------------------------------
            # Columns may arrive as Python list objects (parquet) or
            # string-serialised lists (CSV).
            puzzle_grid = self._coerce_grid(row["puzzle_grid"])
            solution_grid = self._coerce_grid(row["solution_grid"])

            # Replace 0s with None (our internal "masked" sentinel)
            grid = [[v if v != 0 else None for v in r] for r in puzzle_grid]

            # --- build metadata dict -----------------------------------------
            meta: dict[str, Any] = {}
            for col in self._META_COLS:
                if col in row.index:
                    val = row[col]
                    # row/col count lists may also be strings in CSV
                    if isinstance(val, str):
                        try:
                            val = ast.literal_eval(val)
                        except Exception:
                            pass
                    meta[col] = val

            # --- difficulty filter -------------------------------------------
            if difficulty is not None and not difficulty.matches(meta):
                continue

            problems.append(Problem(
                problem_id=str(row.get("id", idx)),
                grid=grid,
                solution=solution_grid,
                metadata=meta,
            ))

            if n_samples is not None and len(problems) >= n_samples:
                break

        return problems

    @staticmethod
    def _coerce_grid(value: Any) -> list[list[int]]:
        """Accept list-of-lists (parquet) or string repr (CSV)."""
        import ast
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return ast.literal_eval(value)
        raise ValueError(f"Cannot parse grid from type {type(value)}: {value!r}")


# ---------------------------------------------------------------------------
# Model back-ends
# ---------------------------------------------------------------------------

class _BaseModelBackend(ABC):
    """Common interface for LLaMA and LLaDA backends."""

    def __init__(self, model_name: str, device: str = "auto", **kwargs):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None
        self._load(model_name, device, **kwargs)

    @abstractmethod
    def _load(self, model_name: str, device: str, **kwargs) -> None: ...

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 256, **kwargs) -> str: ...

    @abstractmethod
    def log_prob(self, prompt: str, completion: str) -> float | None:
        """Return sum of token log-probs for `completion` given `prompt`, or None."""
        ...


class LlamaBackend(_BaseModelBackend):
    """Causal-LM back-end (LLaMA-3, Mistral, etc.)."""

    def _load(self, model_name: str, device: str, **kwargs) -> None:
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            logger.info("Loading LLaMA-family model: %s", model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                torch_dtype=torch.float16,
                **kwargs,
            )
        except ImportError as e:
            raise ImportError("Install `transformers` and `torch` to use LlamaBackend.") from e

    def generate(self, prompt: str, max_new_tokens: int = 256, **kwargs) -> str:
        import torch
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                **kwargs,
            )
        new_tokens = out[0][inputs["input_ids"].shape[-1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def log_prob(self, prompt: str, completion: str) -> float | None:
        import torch
        full_text = prompt + completion
        enc_full = self._tokenizer(full_text, return_tensors="pt").to(self._model.device)
        enc_prompt = self._tokenizer(prompt, return_tensors="pt")
        prompt_len = enc_prompt["input_ids"].shape[-1]
        with torch.no_grad():
            logits = self._model(**enc_full).logits[0]   # (T, V)
        log_probs = torch.log_softmax(logits, dim=-1)
        ids = enc_full["input_ids"][0]
        # sum log-probs of the completion tokens
        total = sum(
            log_probs[i - 1, ids[i]].item()
            for i in range(prompt_len, len(ids))
        )
        return total


class LladaBackend(_BaseModelBackend):
    """
    Masked-diffusion LM back-end (LLaDA / MDLM family).

    LLaDA uses iterative unmasking at inference time.  We expose a simple
    wrapper that runs the standard `generate` API provided by the official
    LLaDA codebase (https://github.com/ML-GSAI/LLaDA).  If the model does
    not expose `generate` in the HuggingFace sense, override `generate` in
    a subclass.
    """

    def _load(self, model_name: str, device: str, **kwargs) -> None:
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch

            logger.info("Loading LLaDA model: %s", model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self._model = AutoModel.from_pretrained(
                model_name,
                device_map=device,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                **kwargs,
            )
        except ImportError as e:
            raise ImportError("Install `transformers` and `torch` to use LladaBackend.") from e

    def generate(self, prompt: str, max_new_tokens: int = 256, **kwargs) -> str:
        """
        Calls `model.generate` with LLaDA-style kwargs.
        Override for custom decoding (e.g. semi-autoregressive).
        """
        import torch
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                # LLaDA-specific defaults (override via **kwargs)
                steps=kwargs.pop("steps", 64),
                temperature=kwargs.pop("temperature", 0.0),
                **kwargs,
            )
        new_tokens = out[0][inputs["input_ids"].shape[-1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def log_prob(self, prompt: str, completion: str) -> float | None:
        # LLaDA does not naturally expose token log-probs; return None.
        return None


# ---------------------------------------------------------------------------
# Prompt builder  (matches the NineGrid benchmark notebook format)
# ---------------------------------------------------------------------------

class PromptBuilder:
    """
    Constructs prompts matching the NineGrid benchmark notebook format.
    The puzzle is serialised as a Python list-of-lists (0 = blank), which
    is the same representation the dataset uses and what LLMs handle best
    for Sudoku — avoiding tokenisation artifacts of space-separated rows.
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


# ---------------------------------------------------------------------------
# Output parser  (matches the NineGrid benchmark notebook parse_response)
# ---------------------------------------------------------------------------

class GridParser:
    """
    Parses a model's text output back into a 2-D list.

    Mirrors the notebook's parse_response logic: scan the full output for all
    [[...]] candidates and return the last one that is a valid N×N grid.
    Taking the *last* valid candidate handles models that self-correct or emit
    chain-of-thought grids before their final answer.
    """

    @staticmethod
    def parse(text: str, expected_rows: int, expected_cols: int) -> list[list[Any]] | None:
        import json

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
# Core evaluator
# ---------------------------------------------------------------------------

class GridEvaluator:
    """
    Evaluates LLaMA or LLaDA models on 2D spatial-reasoning problem sets.

    Parameters
    ----------
    problem_set : ProblemSet
        Any iterable of Problem objects with a `few_shot_examples` method.
    inference_mode : InferenceMode | str
        "zero_shot" or "few_shot".
    model : str | _BaseModelBackend
        HuggingFace model-name string (auto-detects llama vs llada from name),
        or a pre-constructed backend instance.
    n_few_shot : int
        Number of demonstrations for few-shot mode (default 3).
    max_new_tokens : int
        Token budget for the model's response.
    compute_perplexity : bool
        If True and the backend supports it, compute perplexity on gold completions.
    device : str
        Device for model loading ("auto", "cuda", "cpu", etc.).
    model_kwargs : dict
        Extra kwargs forwarded to the model backend constructor.
    """

    def __init__(
        self,
        problem_set: ProblemSet,
        inference_mode: InferenceMode | str = InferenceMode.ZERO_SHOT,
        model: str | _BaseModelBackend = "meta-llama/Llama-3.2-1B-Instruct",
        n_few_shot: int = 3,
        max_new_tokens: int = 256,
        compute_perplexity: bool = False,
        device: str = "auto",
        model_kwargs: dict | None = None,
    ):
        self.problem_set = problem_set
        self.inference_mode = InferenceMode(inference_mode)
        self.n_few_shot = n_few_shot
        self.max_new_tokens = max_new_tokens
        self.compute_perplexity = compute_perplexity

        # ---- resolve model backend ----------------------------------------
        if isinstance(model, _BaseModelBackend):
            self.backend = model
        else:
            family = self._detect_family(model)
            kw = model_kwargs or {}
            if family == ModelFamily.LLADA:
                self.backend = LladaBackend(model, device=device, **kw)
            else:
                self.backend = LlamaBackend(model, device=device, **kw)

        # ---- cache few-shot demos -----------------------------------------
        self._few_shot_pool: list[Problem] = (
            problem_set.few_shot_examples(n_few_shot)
            if self.inference_mode == InferenceMode.FEW_SHOT
            else []
        )

    # ---- public API --------------------------------------------------------

    def evaluate(self, max_problems: int | None = None) -> AggregateMetrics:
        """
        Run the full evaluation loop and return aggregated metrics.
 
        Parameters
        ----------
        max_problems : int | None
            Cap the number of problems (useful for quick smoke-tests).
        """
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None
 
        results: list[EvalResult] = []
        problems = list(self.problem_set)
        if max_problems is not None:
            problems = problems[:max_problems]
 
        iterator = (
            tqdm(problems, desc="Evaluating", unit="puzzle", dynamic_ncols=True)
            if tqdm is not None
            else problems
        )
 
        for prob in iterator:
            result = self._evaluate_one(prob)
            results.append(result)
            logger.info(
                "[%s] exact=%s cell_acc=%.2f format=%s",
                prob.problem_id,
                result.exact_match,
                result.cell_accuracy,
                result.format_valid,
            )
            if tqdm is not None:
                n_done = len(results)
                em_so_far = sum(r.exact_match for r in results) / n_done
                ca_so_far = sum(r.cell_accuracy for r in results) / n_done
                iterator.set_postfix(exact=f"{em_so_far:.2f}", cell_acc=f"{ca_so_far:.2f}")
 
        return self._aggregate(results, rows=problems[0].rows, cols=problems[0].cols)

    # ---- internals ---------------------------------------------------------

    def _evaluate_one(self, problem: Problem) -> EvalResult:
        prompt = PromptBuilder.build(
            problem,
            mode=self.inference_mode,
            few_shot_examples=self._few_shot_pool,
        )

        t0 = time.perf_counter()
        raw = self.backend.generate(prompt, max_new_tokens=self.max_new_tokens)
        elapsed = time.perf_counter() - t0

        predicted = GridParser.parse(raw, problem.rows, problem.cols)
        format_valid = predicted is not None

        result = EvalResult(
            problem_id=problem.problem_id,
            predicted=predicted,
            raw_output=raw,
            format_valid=format_valid,
            inference_time_s=elapsed,
        )

        if format_valid:
            self._fill_accuracy_metrics(result, problem.solution)

        if self.compute_perplexity:
            gold_text = str(PromptBuilder.grid_to_list(problem.solution))
            lp = self.backend.log_prob(prompt, gold_text)
            if lp is not None:
                n_tokens = sum(len(row) for row in problem.solution)
                result.perplexity = float(np.exp(-lp / max(n_tokens, 1)))

        return result

    @staticmethod
    def _fill_accuracy_metrics(result: EvalResult, solution: list[list[Any]]) -> None:
        pred = result.predicted
        rows, cols = len(solution), len(solution[0])

        # cell-level
        total = rows * cols
        correct = 0
        for r in range(rows):
            for c in range(cols):
                match = (r < len(pred) and c < len(pred[r]) and
                         pred[r][c] == solution[r][c])
                result.position_accuracy[(r, c)] = match
                if match:
                    correct += 1
        result.cell_accuracy = correct / total

        # exact match
        result.exact_match = (correct == total)

        # row accuracy
        row_correct = sum(
            all(result.position_accuracy.get((r, c), False) for c in range(cols))
            for r in range(rows)
        )
        result.row_accuracy = row_correct / rows

        # col accuracy
        col_correct = sum(
            all(result.position_accuracy.get((r, c), False) for r in range(rows))
            for c in range(cols)
        )
        result.col_accuracy = col_correct / cols

    @staticmethod
    def _aggregate(
        results: list[EvalResult],
        rows: int,
        cols: int,
    ) -> AggregateMetrics:
        n = len(results)
        if n == 0:
            raise ValueError("No results to aggregate.")

        em = sum(r.exact_match for r in results) / n
        ca = sum(r.cell_accuracy for r in results) / n
        ra = sum(r.row_accuracy for r in results) / n
        cola = sum(r.col_accuracy for r in results) / n
        fv = sum(r.format_valid for r in results) / n
        ti = sum(r.inference_time_s for r in results) / n

        perps = [r.perplexity for r in results if r.perplexity is not None]
        mean_ppl = float(np.mean(perps)) if perps else None

        # position-wise accuracy map  (r,c) → mean over problems
        pos_acc: dict[tuple[int, int], float] = {}
        for r in range(rows):
            for c in range(cols):
                vals = [res.position_accuracy.get((r, c), False) for res in results]
                pos_acc[(r, c)] = float(np.mean(vals))

        return AggregateMetrics(
            n_problems=n,
            exact_match_rate=em,
            mean_cell_accuracy=ca,
            mean_row_accuracy=ra,
            mean_col_accuracy=cola,
            format_validity_rate=fv,
            mean_inference_time_s=ti,
            mean_perplexity=mean_ppl,
            position_accuracy_map=pos_acc,
            per_problem=results,
        )

    @staticmethod
    def _detect_family(model_name: str) -> ModelFamily:
        name_lower = model_name.lower()
        if "llada" in name_lower or "mdlm" in name_lower or "diffusion" in name_lower:
            return ModelFamily.LLADA
        return ModelFamily.LLAMA


# ---------------------------------------------------------------------------
# Quick demo / smoke-test (no GPU needed — uses a mock backend)
# ---------------------------------------------------------------------------

class _MockBackend(_BaseModelBackend):
    """Deterministic back-end that always returns the correct answer (for testing)."""

    def _load(self, *a, **kw): pass

    def generate(self, prompt: str, **kw) -> str:
        # Parse the grid from the prompt and return the solution
        lines = prompt.strip().splitlines()
        grid_lines = [l for l in lines if re.match(r'^[\d\?\s]+$', l.strip()) and l.strip()]
        if not grid_lines:
            return ""
        # For demo: replace '?' with '5'
        return "\n".join(l.replace("?", "5") for l in grid_lines[-3:])

    def log_prob(self, *a, **kw) -> float | None:
        return -6.0

def save_results(
    metrics: AggregateMetrics,
    model_name: str,
    inference_mode: InferenceMode | str,
    problem_set: ProblemSet,
    difficulty: "DifficultyFilter | None" = None,
    output_path: str | None = None,
    extra: dict | None = None,
) -> str:
    """
    Save experiment results to a JSON file.
 
    Parameters
    ----------
    metrics       : AggregateMetrics returned by evaluator.evaluate()
    model_name    : HuggingFace model string or identifier
    inference_mode: InferenceMode used
    problem_set   : the ProblemSet that was evaluated
    difficulty    : DifficultyFilter used (optional)
    output_path   : explicit output path; if None, auto-generates a timestamped filename
    extra         : any additional key/value pairs to include (e.g. n_few_shot, notes)
 
    Returns
    -------
    Path of the saved file.
    """
    import json
    import datetime
 
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
 
    if output_path is None:
        safe_model = model_name.replace("/", "_").replace("-", "_")
        output_path = f"./experiments/results_{safe_model}_{inference_mode}_{timestamp}.json"
 
    # difficulty filter config
    diff_config = None
    if difficulty is not None:
        diff_config = {
            k: v for k, v in {
                "missing_cells": difficulty.missing_cells,
                "given_ratio": difficulty.given_ratio,
                "naked_singles_count": difficulty.naked_singles_count,
                "hidden_singles_count": difficulty.hidden_singles_count,
                "initial_resolution_rate": difficulty.initial_resolution_rate,
            }.items() if v is not None
        }
 
    # dataset difficulty summary if available
    diff_summary = None
    if hasattr(problem_set, "difficulty_summary"):
        diff_summary = problem_set.difficulty_summary()
 
    # # per-problem results
    # per_problem = [
    #     {
    #         "problem_id": r.problem_id,
    #         "exact_match": r.exact_match,
    #         "cell_accuracy": r.cell_accuracy,
    #         "row_accuracy": r.row_accuracy,
    #         "col_accuracy": r.col_accuracy,
    #         "format_valid": r.format_valid,
    #         "inference_time_s": r.inference_time_s,
    #         "perplexity": r.perplexity,
    #     }
    #     for r in metrics.per_problem
    # ]
 
    doc = {
        "timestamp": timestamp,
        "experiment": {
            "model": model_name,
            "inference_mode": str(inference_mode),
            "problem_set": problem_set.name,
            "n_problems": metrics.n_problems,
            "difficulty_filter": diff_config,
            **(extra or {}),
        },
        # "dataset_difficulty_summary": diff_summary,
        "aggregate_metrics": {
            "exact_match_rate": metrics.exact_match_rate,
            "mean_cell_accuracy": metrics.mean_cell_accuracy,
            "mean_row_accuracy": metrics.mean_row_accuracy,
            "mean_col_accuracy": metrics.mean_col_accuracy,
            "format_validity_rate": metrics.format_validity_rate,
            "mean_inference_time_s": metrics.mean_inference_time_s,
            "mean_perplexity": metrics.mean_perplexity,
        },
        # "position_accuracy_map": pos_acc,
        # "per_problem": per_problem,
    }
 
    with open(output_path, "w") as f:
        json.dump(doc, f, indent=2)
 
    print(f"Results saved to: {output_path}")
    return output_path
        
if __name__ == "__main__":

    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    #define the difficulty filter
    diff = DifficultyFilter(
        missing_cells=(30, 45),
        initial_resolution_rate=(0.0, 0.5),
        )

    # 1. Load the dataset
    dataset = NineGrid(
        parquet_path="/data/evan/NineGrid/ninegrid.parquet",
        n_samples=2,                      # start small for a smoke-test
        difficulty=diff,
        few_shot_strategy="easiest",        # use simplest puzzles as demos
    )

    print(f"Loaded {len(dataset)} problems")
    print(dataset.difficulty_summary())

    # 2. Run zero-shot with LLaMA-3-8B
    evaluator = GridEvaluator(
        problem_set=dataset,
        inference_mode=InferenceMode.ZERO_SHOT,
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        max_new_tokens=512,
        compute_perplexity=True,
    )
    print(f" \n Using {next(evaluator.backend._model.parameters()).device} \n")

    metrics = evaluator.evaluate()
    print(metrics.summary())

    print("\n SAVING RESULTS \n")

    save_results(
        metrics=metrics,
        model_name="Meta-Llama-3-8B-Instruct",
        inference_mode=InferenceMode.ZERO_SHOT,
        problem_set=dataset,
        difficulty=diff,
        extra={"n_few_shot": 0, "max_new_tokens": 512, "notes": "baseline run"},
    )

    # 3. Run few-shot with the same model and compare
    #takes too much memory and fails on my machine
    # evaluator_fs = GridEvaluator(
    #     problem_set=dataset,
    #     inference_mode=InferenceMode.FEW_SHOT,
    #     model="meta-llama/Meta-Llama-3-8B-Instruct",
    #     n_few_shot=3,
    #     max_new_tokens=512,
    # )

    # metrics_fs = evaluator_fs.evaluate()
    # print(metrics_fs.summary())

    # 4. Inspect failures
    # failures = [r for r in metrics.per_problem if not r.exact_match]
    # print(f"\n{len(failures)} failures out of {len(dataset)}")

    # for r in failures[:3]:
    #     print(f"\n--- {r.problem_id} ---")
    #     print(f"  format_valid : {r.format_valid}")
    #     print(f"  cell_accuracy: {r.cell_accuracy:.2f}")
    #     print(f"  raw_output   : {r.raw_output[:200]!r}")