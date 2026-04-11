#!/usr/bin/env bash
# =============================================================================
#  run.sh  —  NineGrid evaluation runner
#  Edit the parameters below, then: bash run.sh
# =============================================================================

# ---- Dataset ----------------------------------------------------------------
PARQUET="/data/evan/NineGrid/ninegrid.parquet"   # path to the .parquet file
N_SAMPLES=10000                                     # how many puzzles to evaluate
DIFFICULTY="medium"                               # easy | medium | hard | all

# ---- Model ------------------------------------------------------------------
MODEL="deepseek-ai/deepseek-math-7b-instruct"      # HuggingFace model name: GSAI-ML/LLaDA-8B-Instruct | meta-llama/Meta-Llama-3-8B-Instruct  |  deepseek-ai/deepseek-math-7b-instruct
BACKEND="deepseek"         # auto | llama | deepseek  (auto detects from model name)
DEVICE="cuda:0"          # auto | cuda:0 | cuda:1 | cpu

# ---- Inference --------------------------------------------------------------
MODE="few_shot"           # zero_shot | few_shot
N_FEW_SHOT=5               # number of few-shot examples (ignored for zero_shot)
MAX_NEW_TOKENS=256         # token budget per puzzle
PERPLEXITY=false           # true | false  (adds perplexity computation, LLaMA only)

# ---- Output -----------------------------------------------------------------
OUTPUT_DIR="experiments"       # directory where JSON result files are saved
NOTES="deep seek results on expert"       # free-text notes stored in the result JSON

# ---- Visualisation ----------------------------------------------------------
VISUALIZE=5                # print before/after boards for this many problems (0 = off)
VISUALIZE_FILTER="wrong"   # which to show: all | correct | wrong
 
# =============================================================================
#  Don't edit below this line
# =============================================================================
 
set -euo pipefail
 
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
 
# optional: set PYTORCH_CUDA_ALLOC_CONF to reduce fragmentation OOMs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
 
# build the perplexity flag
PERPLEXITY_FLAG=""
if [ "$PERPLEXITY" = "true" ]; then
    PERPLEXITY_FLAG="--perplexity"
fi
 
echo ""
echo "============================================================"
echo "  NineGrid Evaluation"
echo "  Model      : $MODEL"
echo "  Mode       : $MODE"
echo "  Difficulty : $DIFFICULTY"
echo "  N samples  : $N_SAMPLES"
echo "  Output dir : $OUTPUT_DIR"
echo "============================================================"
echo ""
 
python -m "eval_pkg.run_eval" \
    --visualize        "$VISUALIZE"        \
    --visualize-filter "$VISUALIZE_FILTER" \
    --parquet        "$PARQUET"        \
    --n-samples      "$N_SAMPLES"      \
    --difficulty     "$DIFFICULTY"     \
    --model          "$MODEL"          \
    --backend        "$BACKEND"        \
    --device         "$DEVICE"         \
    --mode           "$MODE"           \
    --n-few-shot     "$N_FEW_SHOT"     \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --output-dir     "$OUTPUT_DIR"     \
    --notes          "$NOTES"          \
    $PERPLEXITY_FLAG
 
echo ""
echo "Done. Results saved to: $OUTPUT_DIR/"