#!/bin/bash

# Evaluation script for distilled diffusion model
# This script evaluates a model distilled using simple_distill.py

# Model configuration
MODEL_PATH_ARG="${1:-./distilled_student_model}"  # Default to distilled_student_model, or pass as first arg
MAX_NEW_TOKENS=128
STEPS=64  # Student was trained with 64 steps (half of teacher's 128)
TEMPERATURE=0.8
ALG="p2"  # Same algorithm used during distillation
NUM_PROCESSES=4

# Get script directory for finding eval directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Convert model path to absolute path before changing directories
# Use realpath if available, otherwise construct absolute path from current directory
if command -v realpath >/dev/null 2>&1; then
    MODEL_PATH="$(realpath "$MODEL_PATH_ARG")"
else
    # Fallback: resolve relative to current working directory
    if [ "${MODEL_PATH_ARG#/}" = "$MODEL_PATH_ARG" ]; then
        MODEL_PATH="$(pwd)/$MODEL_PATH_ARG"
        # Normalize path (remove ./ and ../)
        MODEL_PATH="$(cd "$(dirname "$MODEL_PATH")" && pwd)/$(basename "$MODEL_PATH")"
    else
        MODEL_PATH="$MODEL_PATH_ARG"
    fi
fi

# GPU configuration
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export HF_ALLOW_CODE_EVAL=1

# Check if model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path '$MODEL_PATH' does not exist!"
    echo "Usage: $0 [model_path]"
    echo "Example: $0 ./distilled_student_model"
    exit 1
fi

echo "=========================================="
echo "Evaluating Distilled Model"
echo "=========================================="
echo "Model Path: $MODEL_PATH"
echo "Steps: $STEPS"
echo "Algorithm: $ALG"
echo "Temperature: $TEMPERATURE"
echo "=========================================="

# Change to eval directory
cd "$SCRIPT_DIR/../eval/eval_completion" || exit 1

# Create results directory
RESULTS_DIR="evals_results/distilled-$(basename "$MODEL_PATH")"
mkdir -p "$RESULTS_DIR"

# HumanEval
echo ""
echo "Running HumanEval evaluation..."
accelerate launch --num_processes $NUM_PROCESSES eval.py \
    --model custom_coder \
    --model_args "pretrained=$MODEL_PATH,max_new_tokens=$MAX_NEW_TOKENS,steps=$STEPS,add_bos_token=true,temperature=$TEMPERATURE,top_p=0.95,alg=$ALG" \
    --tasks humaneval \
    --num_fewshot 0 \
    --batch_size 10 \
    --output_path "$RESULTS_DIR/humaneval-ns0" \
    --log_samples \
    --confirm_run_unsafe_code

# HumanEval Plus
echo ""
echo "Running HumanEval Plus evaluation..."
accelerate launch --num_processes $NUM_PROCESSES eval.py \
    --model custom_coder \
    --model_args "pretrained=$MODEL_PATH,max_new_tokens=$MAX_NEW_TOKENS,steps=$STEPS,add_bos_token=true,temperature=$TEMPERATURE,top_p=0.95,alg=$ALG" \
    --tasks humaneval_plus \
    --num_fewshot 0 \
    --batch_size 10 \
    --output_path "$RESULTS_DIR/humaneval_plus-ns0" \
    --log_samples \
    --confirm_run_unsafe_code

# MBPP
echo ""
echo "Running MBPP evaluation..."
accelerate launch --num_processes $NUM_PROCESSES eval.py \
    --model custom_coder \
    --model_args "pretrained=$MODEL_PATH,max_new_tokens=$MAX_NEW_TOKENS,steps=$STEPS,add_bos_token=true,temperature=$TEMPERATURE,top_p=0.95,alg=$ALG" \
    --tasks mbpp \
    --num_fewshot 0 \
    --batch_size 10 \
    --output_path "$RESULTS_DIR/mbpp-ns0" \
    --log_samples \
    --confirm_run_unsafe_code

# MBPP Plus
echo ""
echo "Running MBPP Plus evaluation..."
accelerate launch --num_processes $NUM_PROCESSES eval.py \
    --model custom_coder \
    --model_args "pretrained=$MODEL_PATH,max_new_tokens=$MAX_NEW_TOKENS,steps=$STEPS,add_bos_token=true,temperature=$TEMPERATURE,top_p=0.95,alg=$ALG" \
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

