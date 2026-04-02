MODELS = [
    {
        "name": "Llama_3_8B",
        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "model_type": "hf",
        "dtype": "bfloat16",
        "extra_args": {"trust_remote_code": True},
    },
    {
        "name": "LLaDA_8B",
        "model_id": "GSAI-ML/LLaDA-8B-Instruct",
        "model_type": "hf",
        "dtype": "bfloat16",
        "extra_args": {"trust_remote_code": True},
    },
    {
        "name": "Deepseek_8B",
        "model_id": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        "model_type": "hf",
        "dtype": "bfloat16",
        "extra_args": {"trust_remote_code": True},
    }
]

TASKS = [
    "winogrande",     # Winogrande
    "piqa",           # Physical Intuition QA
    "openbookqa",     # OpenBook QA
    "hellaswag",      # HellaSwag
    "arc_easy",       # ARC Easy
    "arc_challenge",  # ARC Challenge (hard)
    "mmlu",           # Massive Multitask Language Understanding
]

NUM_FEWSHOT_LIST = [0, 1, 5]
GPUS = [0]

EVAL_CONFIG = {
    "batch_size": "auto",
    "limit": None,
    "output_dir": "results",
    "log_samples": False
}