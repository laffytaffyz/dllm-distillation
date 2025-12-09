#!/bin/bash
set -euo pipefail

# Perplexity runner that uses the custom diffusion-aware compute_perplexity
# in eval.py directly (not lm-evaluation-harness loglikelihood_rolling).
# This version pulls prompts from lm-evaluation-harness tasks (doc_to_text).
# Defaults to code benchmarks: HumanEval, HumanEval+, MBPP, MBPP+.

# MODEL_PATH="fredzzp/open-dcoder-0.5B"
# STEPS=16


TEMPERATURE=0.8
ALG="p2"
MAX_NEW_TOKENS=128
BATCH_SIZE=1             # keep small; we aggregate loss across batches
TASKS=("humaneval" "humaneval_plus" "mbpp" "mbpp_plus")
MAX_SAMPLES=""           # set to an integer to truncate for quick tests

echo "=========================================="
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Model Path: $MODEL_PATH"
echo "Steps: $STEPS"
echo "=========================================="

# Check if model path exists
if [ ! -d "$MODEL_PATH" ] && [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model path '$MODEL_PATH' does not exist!"
    exit 1
fi

# Determine if this is a .pt checkpoint file and set base_model if needed
if [[ "$MODEL_PATH" == *.pt ]]; then
    # This is a checkpoint file, need to specify base_model
    # Default base model path
    DEFAULT_BASE_MODEL="/orcd/data/tpoggio/001/tiffany8/dllm-distillation/models/open-dcoder-0.5B"
    if [ -d "$DEFAULT_BASE_MODEL" ]; then
        export BASE_MODEL="$DEFAULT_BASE_MODEL"
    else
        # Fallback to HuggingFace model ID
        export BASE_MODEL="fredzzp/open-dcoder-0.5B"
    fi
    echo "Detected .pt checkpoint file, using base_model: $BASE_MODEL"
else
    export BASE_MODEL=""
fi

export PYTHONPATH=/resource/dllm-distillation/dllm_try/Open-dLLM
export HF_ALLOW_CODE_EVAL=1

python - <<'PY'
import math
import os
import torch
from eval.eval_completion.eval_perplexity_klu_tiff import CustomCoder
from lm_eval import tasks as lm_tasks

model_path = os.environ.get("MODEL_PATH", "fredzzp/open-dcoder-0.5B")
steps = int(os.environ.get("STEPS", "64"))
temperature = float(os.environ.get("TEMPERATURE", "0.8"))
alg = os.environ.get("ALG", "p2")
max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "128"))
batch_size = int(os.environ.get("BATCH_SIZE", "1"))
max_samples_env = os.environ.get("MAX_SAMPLES", "")
max_samples = int(max_samples_env) if max_samples_env.strip() else None

tasks_env = os.environ.get("TASKS", "")
if tasks_env.strip():
    task_names = tasks_env.strip().split(",")
else:
    task_names = ["humaneval", "humaneval_plus", "mbpp", "mbpp_plus"]

device = "cuda" if torch.cuda.is_available() else "cpu"

# Get base_model from environment if provided (for .pt files)
base_model = os.environ.get("BASE_MODEL", None)
if base_model and base_model.strip():
    base_model = base_model.strip()
else:
    base_model = None

print("Loading model...")
if base_model:
    print(f"Using base model: {base_model} for checkpoint file")
lm = CustomCoder(
    pretrained=model_path,
    batch_size=batch_size,
    device=device,
    steps=steps,
    temperature=temperature,
    top_k=200,
    alg=alg,
    max_new_tokens=max_new_tokens,
    base_model=base_model,
)

def iter_prompts(task_name):
    task_dict = lm_tasks.get_task_dict([task_name])
    if task_name not in task_dict:
        raise ValueError(f"Task '{task_name}' not found in lm_eval.tasks")
    task = task_dict[task_name]
    if hasattr(task, "validation_docs") and task.validation_docs() is not None:
        docs = task.validation_docs()
    elif hasattr(task, "test_docs"):
        docs = task.test_docs()
    else:
        docs = []
    for doc in docs:
        yield task.doc_to_text(doc)
output = {}
for task_name in task_names:
    print(f"\nRunning perplexity on task: {task_name}")
    texts = list(iter_prompts(task_name))
    if max_samples is not None:
        texts = texts[:max_samples]
    print(f"Total prompts: {len(texts)}")

    total_loss = 0.0
    total_tokens = 0
    print(f"TASK NAME: {task_name} \n")

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{math.ceil(len(texts) / batch_size)} (indices {i} to {min(i + batch_size, len(texts))})...", flush=True)
        res = lm.compute_perplexity(batch, return_stats=True)
        print(f"Batch {i // batch_size} completed, got result: {res is not None}", flush=True)
        if res is None:
            print(f"Batch {i // batch_size} returned None, skipping", flush=True)
            continue
        ppl, loss_sum, n_tokens = res
        total_loss += loss_sum
        total_tokens += n_tokens
        print(f"Batch {i // batch_size}: ppl={ppl:.4f}, tokens={n_tokens}", flush=True)

    if total_tokens == 0:
        print("No valid tokens encountered.")
    else:
        final_ppl = math.exp(total_loss / total_tokens)
        output[f"{task_name}"] = final_ppl
        print(f"Final perplexity for {task_name}: {final_ppl:.4f}")

print(output)
PY

# run_eval_perplexity.sh