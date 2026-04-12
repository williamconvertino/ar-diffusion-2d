"""
Core evaluation loop: runs a model backend against a problem set and
aggregates per-problem metrics into dataset-level statistics.
"""
from __future__ import annotations

import logging
import time

import numpy as np
from functools import partial

from .eval_types import (
    AggregateMetrics, EvalResult, InferenceMode,
    Problem, ProblemSet,
)
from .models import _BaseModelBackend, build_backend
from .prompt import GridParser, PromptBuilder

logger = logging.getLogger(__name__)


class GridEvaluator:
    """
    Evaluates LLaMA or LLaDA models on 2D spatial-reasoning problem sets.

    Parameters
    ----------
    problem_set       : any ProblemSet (e.g. NineGrid)
    inference_mode    : "zero_shot" or "few_shot"
    model             : HuggingFace model-name string or pre-built backend
    n_few_shot        : number of demonstrations for few-shot mode
    max_new_tokens    : token budget for model responses
    compute_perplexity: compute perplexity on gold completions (LLaMA only)
    device            : "auto", "cuda:0", "cpu", etc.
    model_kwargs      : extra kwargs forwarded to the backend constructor
    """

    def __init__(
        self,
        problem_set: ProblemSet,
        inference_mode: InferenceMode | str = InferenceMode.ZERO_SHOT,
        model: str | _BaseModelBackend = "meta-llama/Meta-Llama-3-8B-Instruct",
        n_few_shot: int = 3,
        max_new_tokens: int = 512,
        compute_perplexity: bool = False,
        device: str = "auto",
        model_kwargs: dict | None = None,
    ):
        self.problem_set       = problem_set
        self.inference_mode    = InferenceMode(inference_mode)
        self.n_few_shot        = n_few_shot
        self.max_new_tokens    = max_new_tokens
        self.compute_perplexity = compute_perplexity

        self.backend = build_backend(model, device=device, model_kwargs=model_kwargs)

        self._few_shot_pool: list[Problem] = (
            problem_set.few_shot_examples(n_few_shot)
            if self.inference_mode == InferenceMode.FEW_SHOT
            else []
        )

    # ---- public API --------------------------------------------------------

    def evaluate(self, max_problems: int | None = None) -> AggregateMetrics:
        """Run the full evaluation loop and return aggregated metrics."""
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        problems = list(self.problem_set)
        if max_problems is not None:
            problems = problems[:max_problems]

        results: list[EvalResult] = []
        iterator = (
            tqdm(problems, desc="Evaluating", unit="puzzle", dynamic_ncols=True)
            if tqdm is not None else problems
        )

        for prob in iterator:
            # result = self._evaluate_one(prob)
            result = self.evaluate_one(prob)
            results.append(result)
            logger.info(
                "[%s] exact=%s cell_acc=%.2f format=%s",
                prob.problem_id, result.exact_match,
                result.cell_accuracy, result.format_valid,
            )
            if tqdm is not None:
                n = len(results)
                iterator.set_postfix(
                    exact=f"{sum(r.exact_match for r in results)/n:.2f}",
                    cell_acc=f"{sum(r.cell_accuracy for r in results)/n:.2f}",
                )

        return self._aggregate(results, rows=problems[0].rows, cols=problems[0].cols)

    def evaluate_one(self, problem: Problem) -> EvalResult:
        return self._evaluate_one(problem)

    # ---- internals ---------------------------------------------------------

    def _evaluate_one(self, problem: Problem, **gen_kwargs) -> EvalResult:
        prompt = PromptBuilder.build(
            problem,
            mode=self.inference_mode,
            few_shot_examples=self._few_shot_pool,
        )

        t0 = time.perf_counter()
        raw = self.backend.generate(prompt, max_new_tokens=self.max_new_tokens, **gen_kwargs)
        elapsed = time.perf_counter() - t0

        predicted    = GridParser.parse(raw, problem.rows, problem.cols)
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
    def _fill_accuracy_metrics(result: EvalResult, solution: list) -> None:
        pred = result.predicted
        rows, cols = len(solution), len(solution[0])
        total, correct = rows * cols, 0

        for r in range(rows):
            for c in range(cols):
                match = (
                    r < len(pred) and c < len(pred[r])
                    and pred[r][c] == solution[r][c]
                )
                result.position_accuracy[(r, c)] = match
                if match:
                    correct += 1

        result.cell_accuracy  = correct / total
        result.exact_match    = (correct == total)
        result.row_accuracy   = sum(
            all(result.position_accuracy.get((r, c), False) for c in range(cols))
            for r in range(rows)
        ) / rows
        result.col_accuracy   = sum(
            all(result.position_accuracy.get((r, c), False) for r in range(rows))
            for c in range(cols)
        ) / cols

    @staticmethod
    def _aggregate(results: list[EvalResult], rows: int, cols: int) -> AggregateMetrics:
        n = len(results)
        if n == 0:
            raise ValueError("No results to aggregate.")

        perps    = [r.perplexity for r in results if r.perplexity is not None]
        pos_acc  = {
            (r, c): float(np.mean([res.position_accuracy.get((r, c), False) for res in results]))
            for r in range(rows) for c in range(cols)
        }

        return AggregateMetrics(
            n_problems           = n,
            exact_match_rate     = sum(r.exact_match    for r in results) / n,
            mean_cell_accuracy   = sum(r.cell_accuracy  for r in results) / n,
            mean_row_accuracy    = sum(r.row_accuracy   for r in results) / n,
            mean_col_accuracy    = sum(r.col_accuracy   for r in results) / n,
            format_validity_rate = sum(r.format_valid   for r in results) / n,
            mean_inference_time_s= sum(r.inference_time_s for r in results) / n,
            mean_perplexity      = float(np.mean(perps)) if perps else None,
            position_accuracy_map= pos_acc,
            per_problem          = results,
        )

class EvaluatorViz(GridEvaluator):
    def __init__(self, viz_path, **eval_kwargs):
        super().__init__(**eval_kwargs)
        self.viz_path =         viz_path        # Path to store the generation text for LLaDA visualization
    
    def evaluate_one(self, problem: Problem, **kwargs):
        return super()._evaluate_one(problem, viz_path=self.viz_path, **kwargs)