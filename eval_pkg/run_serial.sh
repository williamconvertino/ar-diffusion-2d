#!/usr/bin/env bash

PARQUET="./data/four_grid.csv"
N_SAMPLES=288
BACKEND="auto"
DEVICE="auto"

MAX_NEW_TOKENS=512
N_FEW_SHOT=3
PERPLEXITY=false
PROBLEM="fourgrid"

OUTPUT_DIR="experiments_sweep_diffusion"
NOTES="4grid eval sweep"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUTPUT_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODES=("few_shot")
# MODES=("zero_shot")
DIFFICULTIES=("easy" "medium" "hard")
# MODELS=(
#   "meta-llama/Meta-Llama-3-8B-Instruct"
#   "GSAI-ML/LLaDA-8B-Instruct"
#   "deepseek-ai/deepseek-math-7b-instruct"
# )

# AR_Models
# MODELS=(
#   "meta-llama/Meta-Llama-3-8B-Instruct"
#   "deepseek-ai/deepseek-math-7b-instruct"
# )

# Diffusion_Models
MODELS=(
  "GSAI-ML/LLaDA-8B-Instruct"
  "Dream-org/Dream-v0-Instruct-7B"
)
PERPLEXITY_FLAG=""
if [ "$PERPLEXITY" = "true" ]; then
    PERPLEXITY_FLAG="--perplexity"
fi

for MODE in "${MODES[@]}"; do
  echo ""
  echo "============================================================"
  echo "  Starting mode sweep: $MODE"
  echo "============================================================"
  echo ""

  for MODEL in "${MODELS[@]}"; do
    for DIFFICULTY in "${DIFFICULTIES[@]}"; do

      MODEL_TAG="${MODEL##*/}"
      LOG_FILE="$OUTPUT_DIR/run_${MODEL_TAG}_${DIFFICULTY}_${MODE}_${PROBLEM}_output.log"

      echo ""
      echo "------------------------------------------------------------"
      echo "  Model      : $MODEL"
      echo "  Mode       : $MODE"
      echo "  Difficulty : $DIFFICULTY"
      echo "  N samples  : $N_SAMPLES"
      echo "  Output dir : $OUTPUT_DIR"
      echo "------------------------------------------------------------"
      echo ""

      python -m eval_pkg.run_eval \
          --parquet        "$PARQUET"        \
          --n-samples      "$N_SAMPLES"      \
          --difficulty     "$DIFFICULTY"     \
          --model          "$MODEL"          \
          --backend        "$BACKEND"        \
          --device         "$DEVICE"         \
          --problem        "$PROBLEM"        \
          --mode           "$MODE"           \
          --n-few-shot     "$N_FEW_SHOT"     \
          --max-new-tokens "$MAX_NEW_TOKENS" \
          --output-dir     "$OUTPUT_DIR"     \
          --notes          "$NOTES"          \
          $PERPLEXITY_FLAG 2>&1 | tee "$LOG_FILE"

    done
  done
done

echo ""
echo "Done. Results saved to: $OUTPUT_DIR/"