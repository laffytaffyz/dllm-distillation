#!/bin/bash

# Evaluation script for distilled diffusion model
# This script evaluates a model distilled using simple_distill.py

# Model configuration
# Default to checkpoint path, or pass as first arg
# Supports both HuggingFace model directories and .pt checkpoint files
# MODEL_PATH_ARG="${1:-/orcd/data/tpoggio/001/tiffany8/dllm-distillation/checkpoints/latest_checkpoint_64_steps.pt}"
# MODEL_PATH_ARG="${1:-/orcd/data/tpoggio/001/tiffany8/dllm-distillation/models/open-dcoder-0.5B}"
# MODEL_PATH_ARG="${1:-/orcd/data/tpoggio/001/tiffany8/dllm-distillation/models/qwen2-0.5B}"
# MODEL_PATH_ARG="${1:-/orcd/data/tpoggio/001/tiffany8/dllm-distillation/checkpoints/64to32steps/latest_checkpoint.pt}"
MODEL_PATH_ARG="${1:-/orcd/data/tpoggio/001/tiffany8/dllm-distillation/checkpoints/32to16steps/latest_checkpoint_16_steps.pt}"
MAX_NEW_TOKENS=128
STEPS=16  # Student was trained with 16 steps (half of teacher's 32)
TEMPERATURE=0.8
ALG="p2"  # Same algorithm used during distillation
# TOP_P=0.95
# DO_SAMPLE=true
NUM_PROCESSES=1  # Use all 8 GPUs (change if needed)

# Get script directory for finding eval directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# GPU configuration
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export HF_ALLOW_CODE_EVAL=1

# Try to resolve as local path first
if command -v realpath >/dev/null 2>&1; then
    MODEL_PATH="$(realpath "$MODEL_PATH_ARG" 2>/dev/null || echo "$MODEL_PATH_ARG")"
else
    # Fallback: resolve relative to current working directory
    if [ "${MODEL_PATH_ARG#/}" = "$MODEL_PATH_ARG" ]; then
        MODEL_PATH="$(pwd)/$MODEL_PATH_ARG"
        # Normalize path (remove ./ and ../)
        MODEL_PATH="$(cd "$(dirname "$MODEL_PATH")" 2>/dev/null && pwd)/$(basename "$MODEL_PATH")" || MODEL_PATH="$MODEL_PATH_ARG"
    else
        MODEL_PATH="$MODEL_PATH_ARG"
    fi
fi

# Check if local path exists
if [ -d "$MODEL_PATH" ] || [ -f "$MODEL_PATH" ]; then
    # Local path exists - use it
    :
else
    # Local path doesn't exist - assume it's a HuggingFace model ID
    MODEL_PATH="$MODEL_PATH_ARG"
    echo "Local path not found, treating as HuggingFace model ID: $MODEL_PATH"
fi

echo "=========================================="
echo "Evaluating Diffusion Model"
echo "=========================================="
echo "Model Path: $MODEL_PATH"
echo "Steps: $STEPS"
echo "Algorithm: $ALG"
echo "Max New Tokens: $MAX_NEW_TOKENS"
echo "Temperature: $TEMPERATURE"
echo "=========================================="

# Add lm-evaluation-harness to PYTHONPATH (avoids quota issues with pip install)
export PYTHONPATH="$SCRIPT_DIR/../lm-evaluation-harness:$PYTHONPATH"

# Change to eval directory
cd "$SCRIPT_DIR/../eval/eval_completion" || exit 1

# Create results directory
RESULTS_DIR="evals_results/distilled-$(basename "$MODEL_PATH")"
mkdir -p "$RESULTS_DIR"

# HumanEval
echo ""
echo "Running HumanEval evaluation..."
accelerate launch --num_processes $NUM_PROCESSES --num_machines 1 --mixed_precision no --dynamo_backend no eval_iris.py \
    --model custom_coder \
    --model_args "pretrained=$MODEL_PATH,max_new_tokens=$MAX_NEW_TOKENS,steps=$STEPS,temperature=$TEMPERATURE,alg=$ALG,add_bos_token=true" \
    --tasks humaneval \
    --num_fewshot 0 \
    --batch_size 10 \
    --output_path "$RESULTS_DIR/humaneval-ns0" \
    --log_samples \
    --confirm_run_unsafe_code

# HumanEval Plus
echo ""
echo "Running HumanEval Plus evaluation..."
accelerate launch --num_processes $NUM_PROCESSES --num_machines 1 --mixed_precision no --dynamo_backend no eval_iris.py \
    --model custom_coder \
    --model_args "pretrained=$MODEL_PATH,max_new_tokens=$MAX_NEW_TOKENS,steps=$STEPS,temperature=$TEMPERATURE,alg=$ALG,add_bos_token=true" \
    --tasks humaneval_plus \
    --num_fewshot 0 \
    --batch_size 10 \
    --output_path "$RESULTS_DIR/humaneval_plus-ns0" \
    --log_samples \
    --confirm_run_unsafe_code

# MBPP
echo ""
echo "Running MBPP evaluation..."
accelerate launch --num_processes $NUM_PROCESSES --num_machines 1 --mixed_precision no --dynamo_backend no eval_iris.py \
    --model custom_coder \
    --model_args "pretrained=$MODEL_PATH,max_new_tokens=$MAX_NEW_TOKENS,steps=$STEPS,temperature=$TEMPERATURE,alg=$ALG,add_bos_token=true" \
    --tasks mbpp \
    --num_fewshot 0 \
    --batch_size 10 \
    --output_path "$RESULTS_DIR/mbpp-ns0" \
    --log_samples \
    --confirm_run_unsafe_code

# MBPP Plus
echo ""
echo "Running MBPP Plus evaluation..."
accelerate launch --num_processes $NUM_PROCESSES --num_machines 1 --mixed_precision no --dynamo_backend no eval_iris.py \
    --model custom_coder \
    --model_args "pretrained=$MODEL_PATH,max_new_tokens=$MAX_NEW_TOKENS,steps=$STEPS,temperature=$TEMPERATURE,alg=$ALG,add_bos_token=true" \
    --tasks mbpp_plus \
    --num_fewshot 0 \
    --batch_size 10 \
    --output_path "$RESULTS_DIR/mbpp_plus-ns0" \
    --log_samples \
    --confirm_run_unsafe_code

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Summary files:"
echo "  - $RESULTS_DIR/humaneval-ns0/"
echo "  - $RESULTS_DIR/humaneval_plus-ns0/"
echo "  - $RESULTS_DIR/mbpp-ns0/"
echo "  - $RESULTS_DIR/mbpp_plus-ns0/"
echo "=========================================="

