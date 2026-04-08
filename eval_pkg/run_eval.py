"""
run_eval.py — entry point for the NineGrid evaluation pipeline.

All parameters are passed as CLI arguments (set them in run.sh).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

# allow running from the repo root without installing the package
sys.path.insert(0, os.path.dirname(__file__))

from eval_pkg import (
    Difficulty,
    DifficultyFilter,
    FourGridDifficultyFilter,
    GridEvaluator,
    InferenceMode,
    LlamaBackend,
    NineGrid,
    FourGrid,
    save_results,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate LLaMA / LLaDA on the NineGrid Sudoku benchmark."
    )

    # --- dataset ---
    p.add_argument("--parquet",     required=True,
                   help="Path to ninegrid .parquet (or .csv) file")
    p.add_argument("--n-samples",   type=int, default=100,
                   help="Number of problems to evaluate (default: 100)")
    p.add_argument("--difficulty",  choices=["easy", "medium", "hard", "all"],
                   default="medium",
                   help="Difficulty tier: easy | medium | hard | all (default: medium)")
    p.add_argument("--problem",   choices=["ninegrid", "fourgrid", "minesweeper9", "all"],
                   default="ninegrid",
                   help="Problem Name: ninegrid, fourgrid, minesweeper9")
    p.add_argument("--from_preset",   type=bool, default=True,
                   help="Flag whether to load from preset")

    # --- model ---
    p.add_argument("--model",       required=True,
                   help="HuggingFace model name, e.g. meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--device",      default="auto",
                   help="Device for model loading: auto | cuda:0 | cpu (default: auto)")
    p.add_argument("--backend",     choices=["auto", "llama", "llada"], default="auto",
                   help="Force a specific backend: auto | llama | llada (default: auto)")

    # --- inference ---
    p.add_argument("--mode",        choices=["zero_shot", "few_shot"], default="zero_shot",
                   help="Inference mode (default: zero_shot)")
    p.add_argument("--n-few-shot",  type=int, default=3,
                   help="Number of few-shot examples (default: 3, ignored for zero_shot)")
    p.add_argument("--max-new-tokens", type=int, default=512,
                   help="Max tokens to generate per puzzle (default: 512)")
    p.add_argument("--perplexity",  action="store_true",
                   help="Also compute perplexity on gold solutions (LLaMA only)")

    # --- output ---
    p.add_argument("--output-dir",  default="results",
                   help="Directory to save result JSON files (default: ./results)")
    p.add_argument("--notes",       default="",
                   help="Free-text notes appended to the saved JSON")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # logging — silence HuggingFace noise, keep our own INFO lines
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    logging.getLogger("eval_pkg").setLevel(logging.INFO)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("accelerate").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 1. dataset --------------------------------------------------------
    # if args.difficulty == "all":
    #     difficulty = None
    # elif args.problem == "ninegrid" and args.from_preset:
    #     difficulty = DifficultyFilter.from_preset(Difficulty(args.difficulty))
    # else:
    #     difficulty = DifficultyFilter.from_difficulty(Difficulty(args.difficulty))
    
    if args.problem == "ninegrid":
        difficulty = (
            None if args.difficulty == "all"
            else DifficultyFilter.from_preset(Difficulty(args.difficulty))
        )
    
    if args.problem == "fourgrid":
        difficulty = (
            None if args.difficulty == "all"
            else FourGridDifficultyFilter.from_preset(Difficulty(args.difficulty))
        )

    print(f"\n{'='*60}")
    print(f"  Model      : {args.model}")
    print(f"  Mode       : {args.mode}")
    print(f"  Difficulty : {args.difficulty}")
    print(f"  N samples  : {args.n_samples}")
    print(f"{'='*60}\n")

    print("Loading dataset...")
    if args.problem == "ninegrid":
        dataset = NineGrid(
            parquet_path=args.parquet,
            n_samples=args.n_samples,
            difficulty=difficulty,
            few_shot_strategy="easiest",
        )
    
    if args.problem == "fourgrid":
        dataset = FourGrid(
            parquet_path=args.parquet,
            n_samples=args.n_samples,
            difficulty=difficulty,
            few_shot_strategy="easiest",
        )
    dataset.info()

    # ---- 2. load model once ------------------------------------------------
    print(f"\nLoading model: {args.model} (backend: {args.backend})")
    from eval_pkg.models import LladaBackend
    if args.backend == "llada":
        backend = LladaBackend(args.model, device=args.device)
    elif args.backend == "llama":
        backend = LlamaBackend(args.model, device=args.device)
    else:
        from eval_pkg.models import build_backend
        backend = build_backend(args.model, device=args.device)

    # ---- 3. evaluate -------------------------------------------------------
    evaluator = GridEvaluator(
        problem_set=dataset,
        inference_mode=InferenceMode(args.mode),
        model=backend,
        n_few_shot=args.n_few_shot,
        max_new_tokens=args.max_new_tokens,
        compute_perplexity=args.perplexity,
    )

    print(f"\nRunning {args.mode} evaluation on {len(dataset)} problems...\n")
    metrics = evaluator.evaluate()

    print(f"\n{'='*60}")
    print(metrics.summary())
    print(f"{'='*60}\n")

    # ---- 4. save results ---------------------------------------------------
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = args.model.replace("/", "_").replace("-", "_")
    output_path = os.path.join(
        args.output_dir,
        f"results_{safe_model}_{args.mode}_{args.difficulty}_{timestamp}.json"
    )

    save_results(
        metrics=metrics,
        model_name=args.model,
        inference_mode=args.mode,
        problem_set=dataset,
        difficulty=difficulty,
        output_path=output_path,
        extra={
            "n_few_shot":      args.n_few_shot if args.mode == "few_shot" else 0,
            "max_new_tokens":  args.max_new_tokens,
            "device":          args.device,
            "notes":           args.notes,
        },
    )


if __name__ == "__main__":
    main()