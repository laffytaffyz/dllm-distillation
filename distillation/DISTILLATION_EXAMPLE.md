## 🎓 Knowledge Distillation Training

Knowledge distillation allows you to train a smaller student model using a larger pre-trained teacher model.

### Example: Distillation Training

```bash
export TOKENIZERS_PARALLELISM=false
NNODES=${NNODES:=1}
NPROC_PER_NODE=4
NODE_RANK=${NODE_RANK:=0}
MASTER_ADDR=${MASTER_ADDR:=0.0.0.0}
MASTER_PORT=${MASTER_PORT:=12345}

torchrun --nnodes=$NNODES --nproc-per-node $NPROC_PER_NODE --node-rank $NODE_RANK \
  --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT tasks/train_distill.py \
  configs/pretrain/qwen2_5_coder_500M.yaml \
  --data.train_path=data/data \
  --train.ckpt_manager=dcp \
  --train.micro_batch_size=16 \
  --train.global_batch_size=512 \
  --train.output_dir=logs/Qwen2.5-Coder-0.5B_distilled \
  --train.save_steps=10000 \
  --distill.teacher_model_path=Qwen/Qwen2.5-Coder-7B \
  --distill.distillation_alpha=0.5 \
  --distill.temperature=4.0 \
  --distill.freeze_teacher=true
```

### Distillation Parameters

- `--distill.teacher_model_path`: Path to teacher model (e.g., `Qwen/Qwen2.5-Coder-7B`)
- `--distill.distillation_alpha`: Weight for distillation loss (0.0-1.0, default: 0.5)
  - `0.0` = only student loss (no distillation)
  - `0.5` = equal weight to both losses
  - `1.0` = only distillation loss
- `--distill.temperature`: Temperature for softmax in distillation (default: 4.0, higher = softer probabilities)
- `--distill.freeze_teacher`: Whether to freeze teacher model (default: true, recommended)

### Loss Function

The combined loss is:
```
total_loss = (1 - alpha) * student_loss + alpha * distill_loss
```

Where:
- `student_loss`: Standard cross-entropy loss with ground truth labels
- `distill_loss`: KL divergence between teacher and student probability distributions

