
"""
LLaMA / LLaDA evaluator for 2D spatial reasoning benchmarks.
"""
 
from .eval_types import (
    InferenceMode,
    ModelFamily,
    Difficulty,
    Problem,
    EvalResult,
    AggregateMetrics,
    ProblemSet,
)
from .dataset import DifficultyFilter, NineGrid
from .models import LlamaBackend, LladaBackend, MockBackend, build_backend
from .prompt import GridParser, PromptBuilder
from .evaluator import GridEvaluator
from .results import save_results
 
__all__ = [
    # enums
    "InferenceMode", "ModelFamily", "Difficulty",
    # data types
    "Problem", "EvalResult", "AggregateMetrics", "ProblemSet",
    # dataset
    "DifficultyFilter", "NineGrid",
    # models
    "LlamaBackend", "LladaBackend", "MockBackend", "build_backend",
    # prompt / parse
    "PromptBuilder", "GridParser",
    # evaluation
    "GridEvaluator",
    # results
    "save_results",
]