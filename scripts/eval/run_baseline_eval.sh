#!/bin/bash
#SBATCH --job-name=baseline_eval
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --exclude=dcc-h200-gpu-04,dcc-h200-gpu-05
#SBATCH --nodelist=dcc-h200-gpu-01,dcc-h200-gpu-02,dcc-h200-gpu-03,dcc-h200-gpu-06,dcc-h200-gpu-07
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --account=scavenger-h200
#SBATCH --partition=scavenger-h200
#SBATCH --output=../../logs/%x-%j.out
#SBATCH --error=../../logs/%x-%j.err

source /work/wac20/miniconda3/bin/activate

cd ../../baseline_eval

python run_eval.py