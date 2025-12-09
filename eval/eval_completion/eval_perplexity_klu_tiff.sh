#!/bin/bash
set -euo pipefail

# Perplexity runner that uses the custom diffusion-aware compute_perplexity
# in eval.py directly (not lm-evaluation-harness loglikelihood_rolling).
# This version pulls prompts from lm-evaluation-harness tasks (doc_to_text).
# Defaults to code benchmarks: HumanEval, HumanEval+, MBPP, MBPP+.

MODEL_PATH="fredzzp/open-dcoder-0.5B"
STEPS=16
TEMPERATURE=0.8
ALG="p2"
MAX_NEW_TOKENS=128
BATCH_SIZE=1             # keep small; we aggregate loss across batches
TASKS=("humaneval" "humaneval_plus" "mbpp" "mbpp_plus")
MAX_SAMPLES=""           # set to an integer to truncate for quick tests

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

print("Loading model...")
lm = CustomCoder(
    pretrained=model_path,
    batch_size=batch_size,
    device=device,
    steps=steps,
    temperature=temperature,
    top_k=200,
    alg=alg,
    max_new_tokens=max_new_tokens,
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
        res = lm.compute_perplexity(batch, return_stats=True)
        if res is None:
            continue
        ppl, loss_sum, n_tokens = res
        total_loss += loss_sum
        total_tokens += n_tokens
        print(f"Batch {i // batch_size}: ppl={ppl:.4f}, tokens={n_tokens}")

    if total_tokens == 0:
        print("No valid tokens encountered.")
    else:
        final_ppl = math.exp(total_loss / total_tokens)
        output[f"{task_name}"] = final_ppl
        print(f"Final perplexity for {task_name}: {final_ppl:.4f}")

print(output)
PY

# run_eval_perplexity.sh