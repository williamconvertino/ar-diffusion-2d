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
    GridEvaluator,
    InferenceMode,
    LlamaBackend,
    NineGrid,
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
    p.add_argument("--difficulty",  choices=["easy", "medium", "hard", "expert", "all"],
                   default="medium",
                   help="Difficulty tier: easy | medium | hard | all (default: medium)")

    # --- model ---
    p.add_argument("--model",       required=True,
                   help="HuggingFace model name, e.g. meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--device",      default="auto",
                   help="Device for model loading: auto | cuda:0 | cpu (default: auto)")
    p.add_argument("--backend",     choices=["auto", "llama", "llada", "deepseek"], default="auto",
                   help="Force a specific backend: auto | llama | llada | deepseek (default: auto)")

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

    # --- visualisation ---
    p.add_argument("--visualize",   type=int, default=0, metavar="N",
                   help="Print before/after boards for N problems after evaluation (default: 0 = off)")
    p.add_argument("--visualize-filter", choices=["all", "correct", "wrong"], default="all",
                   help="Which problems to visualize: all | correct | wrong (default: all)")

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
    difficulty = (
        None if args.difficulty == "all"
        else DifficultyFilter.from_preset(Difficulty(args.difficulty))
    )

    print(f"\n{'='*60}")
    print(f"  Model      : {args.model}")
    print(f"  Mode       : {args.mode}")
    print(f"  Difficulty : {args.difficulty}")
    print(f"  N samples  : {args.n_samples}")
    print(f"{'='*60}\n")

    print("Loading dataset...")
    dataset = NineGrid(
        parquet_path=args.parquet,
        n_samples=args.n_samples,
        difficulty=difficulty,
        few_shot_strategy="easiest",
    )
    dataset.info()

    # ---- 2. load model once ------------------------------------------------
    print(f"\nLoading model: {args.model} (backend: {args.backend})")
    from eval_pkg.models import LladaBackend, DeepSeekBackend
    if args.backend == "llada":
        backend = LladaBackend(args.model, device=args.device)
    elif args.backend == "llama":
        backend = LlamaBackend(args.model, device=args.device)
    elif args.backend == "deepseek":
        backend = DeepSeekBackend(args.model, device=args.device)
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

    # ---- 5. visualise (optional) ------------------------------------------
    if args.visualize > 0:
        _visualize(
            metrics=metrics,
            problems=list(dataset),
            n=args.visualize,
            filter_mode=args.visualize_filter,
        )


def _visualize(
    metrics,
    problems: list,
    n: int,
    filter_mode: str,
) -> None:
    from eval_pkg.prompt import render_result

    # build a lookup from problem_id -> Problem
    prob_lookup = {p.problem_id: p for p in problems}

    # filter results
    if filter_mode == "correct":
        pool = [r for r in metrics.per_problem if r.exact_match]
        label = "correct"
    elif filter_mode == "wrong":
        pool = [r for r in metrics.per_problem if not r.exact_match]
        label = "wrong"
    else:
        pool = metrics.per_problem
        label = "all"

    to_show = pool[:n]
    if not to_show:
        print(f"  (no {label} results to visualize)")
        return

    print(f"\n{'='*60}")
    print(f"  Visualizing {len(to_show)} {label} problem(s)")
    print(f"{'='*60}")

    for r in to_show:
        prob = prob_lookup.get(r.problem_id)
        if prob is None:
            continue
        status = "✓ CORRECT" if r.exact_match else f"✗ WRONG  (cell acc: {r.cell_accuracy:.0%})"
        print(f"\n  Problem {r.problem_id}  —  {status}  —  {r.inference_time_s:.1f}s")
        if not r.format_valid:
            print("  (model output could not be parsed as a valid grid)")
            print(f"  Raw output: {r.raw_output[:300]!r}")
        else:
            print(render_result(prob, r.predicted))
        print()


if __name__ == "__main__":
    main()