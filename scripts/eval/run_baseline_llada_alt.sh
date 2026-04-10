#!/bin/bash
#SBATCH --job-name=llada_baseline_eval
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --nodelist=dcc-h200-gpu-01,dcc-h200-gpu-02
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --account=scavenger-h200
#SBATCH --partition=scavenger-h200
#SBATCH --output=../../logs/%x-%A_%a.out
#SBATCH --error=../../logs/%x-%A_%a.err

source /work/wac20/miniconda3/bin/activate llada

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

cd ../../llada_eval/LLaDA

./eval_llada_lm_eval_alt.sh
