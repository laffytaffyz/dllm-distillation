# Diffusion Knowledge Distillation

This directory contains tools for knowledge distillation of diffusion-based language models (dLLMs). The pipeline supports:

1. **Loading teacher model from HuggingFace** - Use any pre-trained diffusion LLM as teacher
2. **Deep copying teacher weights to student** - Initialize student with teacher's weights
3. **Collecting logits during diffusion sampling** - Extract teacher logits at each diffusion step
4. **Combined loss function** - Mix cross-entropy with logit matching loss
5. **Flexible data pipeline** - Support both live pipeline and pre-computed logits

## Overview

The distillation process works as follows:

1. **Teacher Model**: A larger, pre-trained diffusion LLM (e.g., `fredzzp/open-dcoder-0.5B` or a 7B model)
2. **Student Model**: A smaller model to be trained (can start from teacher weights via deepcopy)
3. **Diffusion Sampling**: During training, we run diffusion sampling with the teacher model and collect logits at each step
4. **Distillation Loss**: Student learns to match teacher's logits at masked positions during diffusion steps
5. **Combined Training**: `loss = (1-α) * cross_entropy_loss + α * distillation_loss`

## Files

- **`diffusion_logit_collector.py`**: Core utilities for collecting teacher logits during diffusion sampling
- **`train_diffusion_distill.py`**: Enhanced training script that integrates diffusion logit collection
- **`precompute_teacher_logits.py`**: Script to pre-compute and store teacher logits (for faster training)
- **`get_gt_model.py`**: Original example script showing basic logit collection

## Quick Start

### Option 1: Live Pipeline (On-the-fly logit collection)

This mode collects teacher logits during training. Slower but more flexible:

```bash
torchrun --nnodes=1 --nproc-per-node=4 tasks/train_diffusion_distill.py \
  configs/pretrain/qwen2_5_coder_500M.yaml \
  --data.train_path=./data \
  --train.ckpt_manager=dcp \
  --train.micro_batch_size=4 \
  --train.global_batch_size=64 \
  --train.output_dir=./logs/distill_student \
  --distill.teacher_model_path=fredzzp/open-dcoder-0.5B \
  --distill.distillation_alpha=0.5 \
  --distill.num_sampling_steps=200 \
  --distill.collect_every_n_steps=10 \
  --distill.init_student_from_teacher=True \
  --distill.use_stored_logits=False
```

### Option 2: Pre-computed Logits (Faster training)

First, pre-compute teacher logits:

```bash
python distillation/precompute_teacher_logits.py \
  --teacher_model_path=fredzzp/open-dcoder-0.5B \
  --data_path=./data \
  --output_path=./teacher_logits \
  --num_sampling_steps=200 \
  --collect_every_n_steps=10 \
  --batch_size=4 \
  --max_seq_len=2048
```

Then train with stored logits:

```bash
torchrun --nnodes=1 --nproc-per-node=4 tasks/train_diffusion_distill.py \
  configs/pretrain/qwen2_5_coder_500M.yaml \
  --data.train_path=./data \
  --train.output_dir=./logs/distill_student \
  --distill.teacher_model_path=fredzzp/open-dcoder-0.5B \
  --distill.use_stored_logits=True \
  --distill.stored_logits_path=./teacher_logits
```

## Key Parameters

### Diffusion Distillation Arguments

- `teacher_model_path`: HuggingFace model ID or local path to teacher model
- `distillation_alpha`: Weight for distillation loss (0.0 = only cross-entropy, 1.0 = only distillation)
- `temperature`: Temperature for softmax in distillation loss (higher = softer probabilities)
- `init_student_from_teacher`: Whether to deepcopy teacher weights to student
- `num_sampling_steps`: Number of diffusion sampling steps (e.g., 200)
- `collect_every_n_steps`: Collect logits every N steps (1 = all steps, 10 = every 10th step)
- `diffusion_sampling_alg`: Sampling algorithm (`p2`, `entropy`, `origin`, etc.)
- `diffusion_temperature`: Temperature for diffusion sampling
- `diffusion_top_k`: Top-k filtering for diffusion sampling

## How It Works

### 1. Teacher Logit Collection

During diffusion sampling, the teacher model processes sequences with masked tokens. At each step:
- Teacher predicts logits for masked positions
- We collect these logits (optionally every N steps to save memory)
- Logits are stored either in memory (live) or on disk (pre-computed)

### 2. Student Training

For each training batch:
- Student processes the same input sequence
- We compute cross-entropy loss (standard training loss)
- We compute distillation loss by matching student logits to teacher logits at masked positions
- Combined loss: `(1-α) * cross_entropy + α * distillation`

### 3. Diffusion Sampling Algorithms

The pipeline supports multiple sampling algorithms:
- **`p2`**: P2 sampling (recommended, from recent research)
- **`entropy`**: Entropy-based confidence sampling
- **`origin`**: Original discrete diffusion algorithm

## Research Background

This implementation is inspired by:
- **Knowledge Distillation**: Hinton et al., "Distilling the Knowledge in a Neural Network" (2015)
- **Discrete Diffusion**: Hoogeboom et al., "Argmax Flows and Multinomial Diffusion" (2021)
- **Diffusion LLMs**: Recent work on diffusion-based language models (Dream, LLaDA, etc.)

Key insight: Unlike standard autoregressive models, diffusion models generate through multiple denoising steps. By matching logits at each step, the student learns the teacher's denoising behavior.

## Memory Considerations

- **Live Pipeline**: Requires keeping teacher model in memory. Use `collect_every_n_steps > 1` to reduce memory.
- **Pre-computed Logits**: Teacher model only needed during pre-computation. Training is faster but requires disk space.
- **Teacher Freezing**: Always freeze teacher (`freeze_teacher=True`) to save memory and ensure teacher doesn't change.

## Evaluation

After training, evaluate the student model using the standard evaluation scripts:

```bash
cd eval/eval_completion
bash run_eval.sh
```

## Troubleshooting

1. **Out of Memory**: 
   - Reduce `num_sampling_steps` or increase `collect_every_n_steps`
   - Use pre-computed logits instead of live pipeline
   - Reduce batch size

2. **Slow Training**:
   - Use pre-computed logits
   - Increase `collect_every_n_steps`
   - Use smaller teacher model for faster forward passes

3. **Student Not Learning**:
   - Adjust `distillation_alpha` (try 0.3-0.7)
   - Ensure `init_student_from_teacher=True` if architectures match
   - Check that teacher logits are being collected correctly

## Future Improvements

- [ ] Support for averaging logits across multiple diffusion steps
- [ ] Adaptive sampling step selection (focus on important steps)
- [ ] Multi-teacher distillation
- [ ] Progressive distillation (start with fewer steps, increase over time)

