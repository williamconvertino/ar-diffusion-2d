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
        output_path = f"results_{safe_model}_{inference_mode}_{timestamp}.json"
 
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
 
    # position accuracy: convert tuple keys to "r,c" strings for JSON
    pos_acc = {
        f"{r},{c}": v
        for (r, c), v in metrics.position_accuracy_map.items()
    }
 
    # per-problem results
    per_problem = [
        {
            "problem_id": r.problem_id,
            "exact_match": r.exact_match,
            "cell_accuracy": r.cell_accuracy,
            "row_accuracy": r.row_accuracy,
            "col_accuracy": r.col_accuracy,
            "format_valid": r.format_valid,
            "inference_time_s": r.inference_time_s,
            "perplexity": r.perplexity,
        }
        for r in metrics.per_problem
    ]
 
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
        "dataset_difficulty_summary": diff_summary,
        "aggregate_metrics": {
            "exact_match_rate": metrics.exact_match_rate,
            "mean_cell_accuracy": metrics.mean_cell_accuracy,
            "mean_row_accuracy": metrics.mean_row_accuracy,
            "mean_col_accuracy": metrics.mean_col_accuracy,
            "format_validity_rate": metrics.format_validity_rate,
            "mean_inference_time_s": metrics.mean_inference_time_s,
            "mean_perplexity": metrics.mean_perplexity,
        },
        "position_accuracy_map": pos_acc,
        "per_problem": per_problem,
    }
 
    with open(output_path, "w") as f:
        json.dump(doc, f, indent=2)
 
    print(f"Results saved to: {output_path}")
    return output_path