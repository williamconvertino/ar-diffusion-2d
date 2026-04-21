#!/bin/bash
#SBATCH --job-name=5shot_deepseek
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --nodelist=dcc-h200-gpu-01,dcc-h200-gpu-02,dcc-h200-gpu-06
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --account=h200ea
#SBATCH --partition=h200-hp
#SBATCH --output=../../logs/%x-%A_%a.out
#SBATCH --error=../../logs/%x-%A_%a.err

source /work/wac20/miniconda3/bin/activate diffusion2d

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

cd ../../llada_eval/LLaDA

# ./run_0shot_med.sh
# ./run_deepseek_0shot.sh
# ./run_llama3_0shot.sh


# ./run_5shot.sh
./run_deepseek_5shot.sh
# ./run_llama3_5shot.sh

# ./run_5shot_med.sh
# ./run_deepseek_5shot_med.sh
# ./run_llama3_5shot_med.sh