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

# Ensure log directory exists (SLURM won't create it automatically)
mkdir -p ../../logs

# Derive GPU count directly from the --gres line above.
# Change --gres=gpu:h200:N in the header and everything below adjusts automatically.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false

N_GPUS=$SLURM_GPUS_ON_NODE

echo "========================================"
echo "Job:    $SLURM_JOB_NAME ($SLURM_JOB_ID)"
echo "Node:   $SLURMD_NODENAME"
echo "GPUs:   $N_GPUS ($CUDA_VISIBLE_DEVICES)"
echo "Start:  $(date)"
echo "========================================"

torchrun \
    --standalone \
    --nproc_per_node=$N_GPUS \
    train_model.py \
        model=fst_1_3B \
        training=pretrain_1_3B \
        training.use_ddp=true \
        training.num_devices_per_node=$N_GPUS

echo "========================================"
echo "End: $(date)"
echo "========================================"