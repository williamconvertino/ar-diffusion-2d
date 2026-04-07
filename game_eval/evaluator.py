from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Protocol, Sequence, Tuple
import random
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
import statistics


# =========================
# Configs
# =========================

@dataclass
class FewShotExample:
    input_text: str
    target_text: str


@dataclass
class EvaluationConfig:
    problem_set_name: str                    # e.g. "NineGrid"
    inference_kind: str                      # "zero-shot" or "few-shot"
    model_name: str
    batch_size: int
    seeds: List[int]
    prompt_template: str
    max_batches: Optional[int] = None
    few_shot_examples: List[FewShotExample] = field(default_factory=list)
    min_board_size_for_board_accuracy: int = 4


# =========================
# Dataset / Model Protocols
# =========================

class ProblemExample(Protocol):
    """
    Minimal interface expected from each example in the problem set.
    """

    @property
    def uid(self) -> str: ...
    
    @property
    def input_text(self) -> str: ...
    
    @property
    def target_text(self) -> str: ...
    
    @property
    def metadata(self) -> Dict[str, Any]: ...
    # metadata may include things like:
    # - "board_size": int
    # - "answer": Any
    # - "board_target": Any


class ProblemSet(Protocol):
    """
    Minimal interface expected from the dataset/problem set.
    """

    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> ProblemExample: ...


class InferenceModel(Protocol):
    """
    Minimal interface for any model wrapper.
    """

    def generate_batch(
        self,
        prompts: Sequence[str],
        seed: int,
        **kwargs: Any,
    ) -> List[str]:
        """
        Returns one prediction string per prompt.
        """
        ...


# =========================
# Concrete Helper Types
# =========================

@dataclass
class PromptRecord:
    uid: str
    prompt: str
    target_text: str
    metadata: Dict[str, Any]


@dataclass
class InferenceBatch:
    seed: int
    batch_index: int
    records: List[PromptRecord]


@dataclass
class PredictionRecord:
    uid: str
    prompt: str
    prediction: str
    target_text: str
    metadata: Dict[str, Any]


@dataclass
class BatchMetrics:
    seed: int
    batch_index: int
    num_examples: int
    exact_match: float
    token_f1: float
    board_accuracy: Optional[float]
    parse_success_rate: float


@dataclass
class AggregateMetrics:
    exact_match_mean: float
    exact_match_std: float
    token_f1_mean: float
    token_f1_std: float
    parse_success_rate_mean: float
    parse_success_rate_std: float
    board_accuracy_mean: Optional[float]
    board_accuracy_std: Optional[float]
    num_batches: int
    num_predictions: int


# =========================
# Default Metric Utilities
# =========================

def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def exact_match_score(prediction: str, target: str) -> float:
    return float(normalize_text(prediction) == normalize_text(target))


def token_f1_score(prediction: str, target: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    tgt_tokens = normalize_text(target).split()

    if not pred_tokens and not tgt_tokens:
        return 1.0
    if not pred_tokens or not tgt_tokens:
        return 0.0

    pred_counts: Dict[str, int] = {}
    tgt_counts: Dict[str, int] = {}

    for tok in pred_tokens:
        pred_counts[tok] = pred_counts.get(tok, 0) + 1
    for tok in tgt_tokens:
        tgt_counts[tok] = tgt_counts.get(tok, 0) + 1

    overlap = 0
    for tok, count in pred_counts.items():
        overlap += min(count, tgt_counts.get(tok, 0))

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(tgt_tokens)
    return 2 * precision * recall / (precision + recall)


def default_prediction_parser(prediction: str) -> Tuple[bool, Any]:
    """
    Generic parser: returns the normalized prediction itself.
    Replace this with a task-specific parser if needed.
    """
    pred = normalize_text(prediction)
    return (len(pred) > 0), pred


def default_target_parser(target_text: str, metadata: Dict[str, Any]) -> Any:
    """
    Generic target parser: prefers metadata['answer'] if present,
    otherwise falls back to normalized target text.
    """
    if "answer" in metadata:
        return metadata["answer"]
    return normalize_text(target_text)


def default_board_accuracy_fn(
                                parsed_pred: Any, 
                                metadata: Dict[str, Any]
                            ) -> Optional[float]:
    """
    Default board accuracy:
    - if metadata contains `board_target` and parsed_pred is same type/shape-ish, compare elementwise
    - otherwise return None

    Expected metadata example:
        metadata["board_target"] = [[...], [...], ...]
    """
    board_target = metadata.get("board_target")
    if board_target is None:
        return None

    if not isinstance(parsed_pred, list):
        return None
    if not isinstance(board_target, list):
        return None
    if len(parsed_pred) != len(board_target):
        return None
    if len(board_target) == 0:
        return None

    total = 0
    correct = 0
    for pred_row, tgt_row in zip(parsed_pred, board_target):
        if not isinstance(pred_row, list) or not isinstance(tgt_row, list):
            return None
        if len(pred_row) != len(tgt_row):
            return None
        for p, t in zip(pred_row, tgt_row):
            total += 1
            correct += int(p == t)

    return correct / total if total > 0 else None


# =========================
# Evaluation Class
# =========================

class EvaluationRunner:
    """
    Config-driven evaluation class.

    Responsibilities:
    - construct prompts from problem examples
    - yield inference batches with complete prompt structure
    - run model inference over seeds and batches
    - compute per-batch and aggregate metrics
    """

    def __init__(
        self,
        config: EvaluationConfig,
        problem_set: ProblemSet,
        model: InferenceModel,
        *,
        prediction_parser: Callable[[str], Tuple[bool, Any]] = default_prediction_parser,
        target_parser: Callable[[str, Dict[str, Any]], Any] = default_target_parser,
        metrics_list: List[Callable[Any]] = [default_board_accuracy_fn],
    ) -> None:
        self.config = config
        self.problem_set = problem_set
        self.model = model
        self.prediction_parser = prediction_parser
        self.target_parser = target_parser
        self.board_accuracy_fn = board_accuracy_fn

        allowed = {"zero-shot", "few-shot"}
        if self.config.inference_kind not in allowed:
            raise ValueError(
                f"inference_kind must be one of {allowed}, got {self.config.inference_kind!r}"
            )

        if self.config.inference_kind == "few-shot" and not self.config.few_shot_examples:
            raise ValueError("few-shot inference requires few_shot_examples in config")

        if self.config.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        if not self.config.seeds:
            raise ValueError("seeds must be non-empty")

    # -------------------------
    # Prompt construction
    # -------------------------

    def _format_few_shot_block(self) -> str:
        if self.config.inference_kind != "few-shot":
            return ""

        blocks = []
        for ex in self.config.few_shot_examples:
            blocks.append(
                f"Example Input:\n{ex.input_text}\n\nExample Output:\n{ex.target_text}"
            )
        return "\n\n".join(blocks)

    def build_prompt(self, example: ProblemExample) -> str:
        """
        Required placeholders in prompt_template can include:
        - {problem_set_name}
        - {inference_kind}
        - {few_shot_block}
        - {input_text}
        - {metadata}
        """
        few_shot_block = self._format_few_shot_block()
        return self.config.prompt_template.format(
            problem_set_name=self.config.problem_set_name,
            inference_kind=self.config.inference_kind,
            few_shot_block=few_shot_block,
            input_text=example.input_text,
            metadata=example.metadata,
        )

    # -------------------------
    # Batch generation
    # -------------------------

    def iter_inference_batches(self) -> Generator[InferenceBatch, None, None]:
        """
        Yields all batches across all seeds with fully formatted prompts.
        """
        n = len(self.problem_set)
        all_indices = list(range(n))

        for seed in self.config.seeds:
            rng = random.Random(seed)
            indices = all_indices[:]
            rng.shuffle(indices)

            if self.config.max_batches is None:
                max_items = len(indices)
            else:
                max_items = min(len(indices), self.config.max_batches * self.config.batch_size)

            indices = indices[:max_items]

            batch_index = 0
            for start in range(0, len(indices), self.config.batch_size):
                batch_indices = indices[start:start + self.config.batch_size]
                records: List[PromptRecord] = []

                for idx in batch_indices:
                    ex = self.problem_set[idx]
                    records.append(
                        PromptRecord(
                            uid=ex.uid,
                            prompt=self.build_prompt(ex),
                            target_text=ex.target_text,
                            metadata=ex.metadata,
                        )
                    )

                yield InferenceBatch(
                    seed=seed,
                    batch_index=batch_index,
                    records=records,
                )
                batch_index += 1

    # -------------------------
    # Inference
    # -------------------------

    def run_batch(self, batch: InferenceBatch, **model_kwargs: Any) -> Tuple[List[PredictionRecord], BatchMetrics]:
        prompts = [r.prompt for r in batch.records]
        predictions = self.model.generate_batch(prompts=prompts, seed=batch.seed, **model_kwargs)

        if len(predictions) != len(batch.records):
            raise RuntimeError(
                f"Model returned {len(predictions)} predictions for {len(batch.records)} prompts."
            )

        pred_records: List[PredictionRecord] = []
        for rec, pred in zip(batch.records, predictions):
            pred_records.append(
                PredictionRecord(
                    uid=rec.uid,
                    prompt=rec.prompt,
                    prediction=pred,
                    target_text=rec.target_text,
                    metadata=rec.metadata,
                )
            )

        metrics = self.compute_batch_metrics(
            seed=batch.seed,
            batch_index=batch.batch_index,
            predictions=pred_records,
        )
        return pred_records, metrics

    def run(self, **model_kwargs: Any) -> Dict[str, Any]:
        all_predictions: List[PredictionRecord] = []
        all_batch_metrics: List[BatchMetrics] = []

        for batch in self.iter_inference_batches():
            pred_records, batch_metrics = self.run_batch(batch, **model_kwargs)
            all_predictions.extend(pred_records)
            all_batch_metrics.append(batch_metrics)

        aggregate = self.aggregate_metrics(all_batch_metrics)

        return {
            "config": self.config,
            "predictions": all_predictions,
            "batch_metrics": all_batch_metrics,
            "aggregate_metrics": aggregate,
        }

    # -------------------------
    # Metrics
    # -------------------------

    def compute_batch_metrics(
        self,
        seed: int,
        batch_index: int,
        predictions: Sequence[PredictionRecord],
    ) -> BatchMetrics:
        em_scores: List[float] = []
        f1_scores: List[float] = []
        parse_successes: List[float] = []
        board_scores: List[float] = []

        for record in predictions:
            em_scores.append(exact_match_score(record.prediction, record.target_text))
            f1_scores.append(token_f1_score(record.prediction, record.target_text))

            parse_ok, parsed_pred = self.prediction_parser(record.prediction)
            parse_successes.append(float(parse_ok))

            board_size = record.metadata.get("board_size")
            if board_size is not None and board_size < self.config.min_board_size_for_board_accuracy:
                continue

            if parse_ok:
                board_acc = self.board_accuracy_fn(parsed_pred, record.metadata)
                if board_acc is not None:
                    board_scores.append(board_acc)

        return BatchMetrics(
            seed=seed,
            batch_index=batch_index,
            num_examples=len(predictions),
            exact_match=sum(em_scores) / len(em_scores) if em_scores else 0.0,
            token_f1=sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
            board_accuracy=(sum(board_scores) / len(board_scores)) if board_scores else None,
            parse_success_rate=sum(parse_successes) / len(parse_successes) if parse_successes else 0.0,
        )

    def aggregate_metrics(self, batch_metrics: Sequence[BatchMetrics]) -> AggregateMetrics:
        if not batch_metrics:
            return AggregateMetrics(
                exact_match_mean=0.0,
                exact_match_std=0.0,
                token_f1_mean=0.0,
                token_f1_std=0.0,
                parse_success_rate_mean=0.0,
                parse_success_rate_std=0.0,
                board_accuracy_mean=None,
                board_accuracy_std=None,
                num_batches=0,
                num_predictions=0,
            )

        em_vals = [m.exact_match for m in batch_metrics]
        f1_vals = [m.token_f1 for m in batch_metrics]
        parse_vals = [m.parse_success_rate for m in batch_metrics]
        board_vals = [m.board_accuracy for m in batch_metrics if m.board_accuracy is not None]

        return AggregateMetrics(
            exact_match_mean=statistics.mean(em_vals),
            exact_match_std=statistics.pstdev(em_vals) if len(em_vals) > 1 else 0.0,
            token_f1_mean=statistics.mean(f1_vals),
            token_f1_std=statistics.pstdev(f1_vals) if len(f1_vals) > 1 else 0.0,
            parse_success_rate_mean=statistics.mean(parse_vals),
            parse_success_rate_std=statistics.pstdev(parse_vals) if len(parse_vals) > 1 else 0.0,
            board_accuracy_mean=statistics.mean(board_vals) if board_vals else None,
            board_accuracy_std=statistics.pstdev(board_vals) if len(board_vals) > 1 else (0.0 if board_vals else None),
            num_batches=len(batch_metrics),
            num_predictions=sum(m.num_examples for m in batch_metrics),
        )

# class Evaluator:
#     def __init__(self,
#                 problem_set_path,
#                 inference_type,
#                 model_path,
#                 batches,
#                 seed,
#                 prompt_template
#                 ):
#         self.model_path = model_path
#         self.problem_set_path = porblem_set_path
#         self.batches = batches
#         self.seed = seed
#         self.prompt_template = prompt_template
#         self.inference_type = inference_type

#     def load_model(self, model_path):
#         return None
    
#     def build_data_generator(self, 
#                             problem_set_path,
#                             batches
#                             ):
#         return None
    
#     def compute_metrics(self, 
#                         y_true,
#                         y_pred,
#                         metrics_list
#                         ):
        
#         return [None]
    
#     def 