#!/bin/bash
#SBATCH --job-name=baseline_eval_array
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --nodelist=dcc-h200-gpu-02
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --account=scavenger-h200
#SBATCH --partition=scavenger-h200
#SBATCH --array=0-2
#SBATCH --output=../../logs/%x-%A_%a.out
#SBATCH --error=../../logs/%x-%A_%a.err

source /work/wac20/miniconda3/bin/activate diffusion2d

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

cd ../../baseline_eval

MODELS=(
    "Llama_3_8B"
    "LLaDA_8B"
    "Deepseek_8B"
)

MODEL_NAME="${MODELS[$SLURM_ARRAY_TASK_ID]}"

echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Running model: $MODEL_NAME"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES before launch: ${CUDA_VISIBLE_DEVICES}"

python run_eval.py --models "$MODEL_NAME"