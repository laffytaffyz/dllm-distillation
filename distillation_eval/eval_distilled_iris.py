#!/usr/bin/env python3
"""
Evaluation script for distilled diffusion model.

Usage:
    python eval_distilled.py [--model_path MODEL_PATH] [--steps STEPS] [--temperature TEMP]
    
Example:
    python eval_distilled.py --model_path ./distilled_student_model --steps 64
"""

import argparse
import os
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Evaluate distilled diffusion model")
    parser.add_argument(
        "model_path",
        nargs="?",
        type=str,
        default="fredzzp/open-dcoder-0.5B",
        help="Path to distilled model directory (default: fredzzp/open-dcoder-0.5B)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=16,
        help="Number of diffusion steps (default: 16, matching student training)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)"
    )
    parser.add_argument(
        "--alg",
        type=str,
        default="p2",
        help="Sampling algorithm (default: p2)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate (default: 128)"
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=8,
        help="Number of processes for accelerate (default: 8)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Batch size for evaluation (default: 10)"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["humaneval_iris", "humaneval_plus_iris", "mbpp_iris", "mbpp_plus_iris"],
        help="Tasks to evaluate (default: all generation tasks; perplexity is computed on prompts automatically)"
    )
    parser.add_argument(
        "--cuda_devices",
        type=str,
        default=None,
        help="CUDA devices to use (e.g., '0,1,2,3'). If not set, uses CUDA_VISIBLE_DEVICES env var"
    )
    
    args = parser.parse_args()
    
    # Get script directory for finding eval_iris.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try to resolve as local path first
    model_path_arg = args.model_path
    model_path = os.path.abspath(model_path_arg)
    
    # Check if local path exists
    if os.path.isdir(model_path) or os.path.isfile(model_path):
        # Local path exists - use it
        model_path = model_path
    else:
        # Local path doesn't exist - assume it's a HuggingFace model ID
        model_path = model_path_arg
        print(f"Local path not found, treating as HuggingFace model ID: {model_path}")
    
    # Set CUDA devices
    if args.cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    
    # Get absolute path to eval_iris.py (uses fixed diffusion perplexity)
    eval_dir = os.path.join(script_dir, "..", "eval", "eval_completion")
    eval_dir = os.path.abspath(eval_dir)  # Resolve .. in path
    eval_script = os.path.join(eval_dir, "eval_iris.py")
    
    if not os.path.exists(eval_script):
        print(f"Error: Could not find eval script at {eval_script}")
        sys.exit(1)
    
    # Create results directory
    model_name = os.path.basename(model_path)
    results_dir = os.path.join(eval_dir, "evals_results", f"distilled-{model_name}")
    os.makedirs(results_dir, exist_ok=True)
    
    print("=" * 50)
    print("Evaluating Distilled Model")
    print("=" * 50)
    print(f"Model Path: {model_path}")
    print(f"Steps: {args.steps}")
    print(f"Algorithm: {args.alg}")
    print(f"Temperature: {args.temperature}")
    print(f"Tasks: {', '.join(args.tasks)}")
    print(f"Results Dir: {results_dir}")
    print("=" * 50)
    print()
    
    # Build model args string
    model_args = (
        f"pretrained={model_path},"
        f"max_new_tokens={args.max_new_tokens},"
        f"steps={args.steps},"
        f"add_bos_token=true,"
        f"temperature={args.temperature},"
        f"alg={args.alg}"
    )
    
    # Run evaluation for each task
    for task in args.tasks:
        print(f"\n{'='*50}")
        print(f"Running {task} evaluation...")
        print(f"{'='*50}")
        
        output_path = os.path.join(results_dir, f"{task}-ns0")
        
        cmd = [
            "accelerate", "launch",
            "--num_processes", str(args.num_processes),
            eval_script,
            "--model", "custom_coder",
            "--model_args", model_args,
            "--tasks", task,
            "--num_fewshot", "0",
            "--batch_size", str(args.batch_size),
            "--gen_kwargs", "num_return_sequences=10",
            "--output_path", output_path,
            "--log_samples",
            "--confirm_run_unsafe_code"
        ]
        
        # Use full dataset for all tasks (no sampling)
        print(f"Command: {' '.join(cmd)}")
        print()
        
        result = subprocess.run(cmd, cwd=eval_dir)
        
        if result.returncode != 0:
            print(f"\nError: Evaluation of {task} failed with return code {result.returncode}")
            sys.exit(1)
    
    print()
    print("=" * 50)
    print("Evaluation Complete!")
    print("=" * 50)
    print(f"Results saved to: {results_dir}")
    print()
    print("Summary files:")
    for task in args.tasks:
        print(f"  - {results_dir}/{task}-ns0/")
    print("=" * 50)

if __name__ == "__main__":
    main()

