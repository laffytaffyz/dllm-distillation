"""
Step-wise Diffusion Distillation Script for Open-dCoder

Distills a teacher model (128 steps) to a student model (64 steps).
Teacher logits are collected every 2 steps, and student learns to match them.

Usage:
    python simple_distill.py
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# Import the custom model and generation config
try:
    from veomni.models.transformers.qwen2.modeling_qwen2 import Qwen2ForCausalLM
    from veomni.models.transformers.qwen2.generation_utils import MDMGenerationConfig, sample_tokens
except ImportError:
    print("Warning: veomni not found. Trying alternative import...")
    import sys
    sys.path.insert(0, '.')
    from veomni.models.transformers.qwen2.modeling_qwen2 import Qwen2ForCausalLM
    from veomni.models.transformers.qwen2.generation_utils import MDMGenerationConfig, sample_tokens

# Configuration
TEACHER_MODEL = "fredzzp/open-dcoder-0.5B"
STUDENT_MODEL = "fredzzp/open-dcoder-0.5B"  # Can be same or different
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2  # Smaller batch size due to memory requirements
LEARNING_RATE = 5e-5
NUM_EPOCHS = 1
DISTILLATION_ALPHA = 0.5  # Weight for distillation loss
TEMPERATURE = 4.0  # Temperature for softmax
MAX_SEQ_LEN = 256  # Reduced for memory efficiency

# Diffusion parameters
TEACHER_STEPS = 128  # Teacher runs 128 steps
STUDENT_STEPS = 64   # Student runs 64 steps
COLLECT_EVERY_N = 2  # Collect teacher logits every 2 steps
ALG = "p2"  # Sampling algorithm
DIFFUSION_TEMP = 0.7  # Temperature for diffusion sampling
TOP_K = 200

print(f"Loading teacher model: {TEACHER_MODEL}")
teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL, trust_remote_code=True)
teacher_model = Qwen2ForCausalLM.from_pretrained(
    TEACHER_MODEL,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).to(DEVICE).eval()

# Add mask token if needed
if teacher_tokenizer.mask_token is None:
    teacher_tokenizer.add_special_tokens({"mask_token": "<M>"})
    teacher_model.resize_token_embeddings(len(teacher_tokenizer))

print(f"Loading student model: {STUDENT_MODEL}")
student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, trust_remote_code=True)
student_model = Qwen2ForCausalLM.from_pretrained(
    STUDENT_MODEL,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).to(DEVICE).train()

# Add mask token if needed
if student_tokenizer.mask_token is None:
    student_tokenizer.add_special_tokens({"mask_token": "<M>"})
    student_model.resize_token_embeddings(len(student_tokenizer))

# Deep copy teacher weights to student
print("Initializing student from teacher weights...")
try:
    teacher_state_dict = teacher_model.state_dict()
    student_state_dict = student_model.state_dict()
    
    copied = 0
    for name, param in teacher_state_dict.items():
        if name in student_state_dict and param.shape == student_state_dict[name].shape:
            student_state_dict[name].data.copy_(param.data)
            copied += 1
    
    student_model.load_state_dict(student_state_dict, strict=False)
    print(f"Copied {copied} parameters from teacher to student")
except Exception as e:
    print(f"Could not copy weights: {e}")

# Freeze teacher
for param in teacher_model.parameters():
    param.requires_grad = False

# Setup optimizer
optimizer = torch.optim.AdamW(student_model.parameters(), lr=LEARNING_RATE)

# Load dataset
print("Loading dataset...")
try:
    dataset = load_dataset("bigcode/the-stack-dedup", data_dir="data/python", split="train[:500]")
    def tokenize_function(examples):
        return teacher_tokenizer(
            examples["content"],
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding="max_length"
        )
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
except Exception as e:
    print(f"Could not load dataset: {e}")
    print("Creating dummy data...")
    dummy_texts = ["def hello():\n    print('hello')", "def world():\n    print('world')"] * 50
    tokenized_dataset = teacher_tokenizer(
        dummy_texts,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
        return_tensors="pt"
    )


def run_diffusion_with_logit_collection(
    model,
    input_ids,
    num_steps,
    collect_every_n,
    tokenizer,
    device,
    alg=ALG,
    temperature=DIFFUSION_TEMP,
    top_k=TOP_K
):
    """
    Run diffusion sampling and collect logits at specified steps.
    
    Returns:
        collected_logits: List of logit tensors [num_collected, batch, seq_len, vocab_size]
        collected_mask_positions: List of mask position tensors
        collected_states: List of x_t states at collection points
    """
    mask_token_id = tokenizer.mask_token_id
    batch_size, prompt_len = input_ids.shape
    
    # Pad with mask tokens for generation
    max_length = input_ids.shape[1] + 128  # Add space for generation
    pad_len = max_length - prompt_len
    masked_suffix = torch.full(
        (batch_size, pad_len),
        mask_token_id,
        device=device,
        dtype=torch.long
    )
    x_t = torch.cat([input_ids, masked_suffix], dim=1)
    
    # Fixed tokens (prompt) should never be remasked
    fix_mask = (x_t != mask_token_id)
    
    # Attention mask
    attention_mask = (x_t != tokenizer.pad_token_id).long() if tokenizer.pad_token_id is not None else None
    
    # Timesteps from 1.0 down to eps
    eps = 1e-3
    timesteps = torch.linspace(1.0, eps, num_steps + 1, device=device)
    
    # Storage
    collected_logits = []
    collected_mask_positions = []
    collected_states = []
    
    with torch.no_grad():
        for step in range(num_steps):
            mask_index = (x_t == mask_token_id)
            if not mask_index.any():
                break
            
            t = timesteps[step]
            s = timesteps[step + 1] if step < num_steps - 1 else eps
            
            # Forward pass
            outputs = model(input_ids=x_t, attention_mask=attention_mask, is_causal=False)
            logits = outputs.logits
            
            # Shift logits to align with training
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            
            # Collect logits if this is a collection step
            if step % collect_every_n == 0:
                collected_logits.append(logits.clone())
                collected_mask_positions.append(mask_index.clone())
                collected_states.append(x_t.clone())
            
            # Update x_t using p2 algorithm (same as generation_utils.py)
            mask_logits = logits[mask_index]
            
            if alg == "p2":
                kappa_t = (step + 1) / num_steps
                
                # Compute confidence and sampled tokens
                probs = F.softmax(logits / temperature, dim=-1)
                confidence = probs.max(dim=-1).values
                x0_full = logits.argmax(dim=-1)
                
                # Construct confidence matrix
                full_conf = confidence.clone()
                full_conf[fix_mask] = float("inf")
                full_conf = torch.where(
                    torch.isfinite(full_conf), full_conf, torch.full_like(full_conf, float("inf"))
                )
                
                # Calculate how many positions to re-mask
                num_positions = (~fix_mask).sum(dim=1)
                num_to_mask = (num_positions.float() * (1.0 - kappa_t)).floor().to(torch.long)
                num_to_mask = num_to_mask.clamp_min(0)
                num_to_mask = torch.minimum(num_to_mask, num_positions)
                
                # Select lowest-confidence positions for re-masking
                sorted_idx = torch.argsort(full_conf, dim=1, descending=False)
                max_k = int(num_to_mask.max().item())
                
                if max_k > 0:
                    topk_idx = sorted_idx[:, :max_k]
                    row_mask = torch.arange(max_k, device=device).unsqueeze(0) < num_to_mask.unsqueeze(1)
                    
                    to_mask = torch.zeros_like(x_t, dtype=torch.bool)
                    batch_arange = torch.arange(x_t.size(0), device=device).unsqueeze(1).expand_as(topk_idx)
                    valid_batch = batch_arange[row_mask]
                    valid_col = topk_idx[row_mask]
                    to_mask[valid_batch, valid_col] = True
                else:
                    to_mask = torch.zeros_like(x_t, dtype=torch.bool)
                
                # Apply re-masking
                x_t[to_mask] = mask_token_id
                
                # Unmask positions that started as mask and weren't re-masked
                keep_unmask = mask_index & (~to_mask)
                x_t[keep_unmask] = x0_full[keep_unmask]
            
            else:
                raise NotImplementedError(f"Algorithm {alg} not implemented")
    
    return collected_logits, collected_mask_positions, collected_states


# Training loop
print("Starting training...")
mask_token_id = teacher_tokenizer.mask_token_id

# Create dataloader
if isinstance(tokenized_dataset, dict):
    input_ids = tokenized_dataset["input_ids"]
    num_batches = (len(input_ids) + BATCH_SIZE - 1) // BATCH_SIZE
    dataset_type = "dict"
else:
    dataloader = torch.utils.data.DataLoader(
        tokenized_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    dataset_type = "dataloader"

for epoch in range(NUM_EPOCHS):
    total_loss = 0
    total_student_loss = 0
    total_distill_loss = 0
    num_batches_processed = 0
    
    if dataset_type == "dict":
        batch_iter = range(num_batches)
    else:
        batch_iter = dataloader
    
    for batch_idx in tqdm(batch_iter, desc=f"Epoch {epoch+1}"):
        if dataset_type == "dict":
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(input_ids))
            batch_input_ids = input_ids[start_idx:end_idx].to(DEVICE)
        else:
            batch_input_ids = batch_idx["input_ids"].to(DEVICE)
        
        # Prepare input: use first part as prompt, rest will be masked
        prompt_len = batch_input_ids.shape[1] // 2  # Use half as prompt
        prompt_ids = batch_input_ids[:, :prompt_len]
        
        # ========== TEACHER: Run 128 steps, collect every 2 steps ==========
        with torch.no_grad():
            teacher_logits_list, teacher_mask_positions_list, teacher_states_list = run_diffusion_with_logit_collection(
                model=teacher_model,
                input_ids=prompt_ids,
                num_steps=TEACHER_STEPS,
                collect_every_n=COLLECT_EVERY_N,
                tokenizer=teacher_tokenizer,
                device=DEVICE,
                alg=ALG,
                temperature=DIFFUSION_TEMP,
                top_k=TOP_K
            )
        
        # We should have 64 teacher logits (128 steps / 2)
        assert len(teacher_logits_list) == STUDENT_STEPS, f"Expected {STUDENT_STEPS} teacher logits, got {len(teacher_logits_list)}"
        
        # ========== STUDENT: Run 64 steps, collect at each step ==========
        student_logits_list = []
        student_mask_positions_list = []
        student_states_list = []
        
        # Initialize student's x_t from teacher's first state (same starting point)
        x_t_student = teacher_states_list[0].clone()
        fix_mask = (x_t_student != mask_token_id)
        attention_mask = (x_t_student != student_tokenizer.pad_token_id).long() if student_tokenizer.pad_token_id is not None else None
        
        eps = 1e-3
        timesteps_student = torch.linspace(1.0, eps, STUDENT_STEPS + 1, device=DEVICE)
        
        for step in range(STUDENT_STEPS):
            mask_index = (x_t_student == mask_token_id)
            if not mask_index.any():
                break
            
            t = timesteps_student[step]
            s = timesteps_student[step + 1] if step < STUDENT_STEPS - 1 else eps
            
            # Forward pass (WITH gradients for student)
            outputs = student_model(input_ids=x_t_student, attention_mask=attention_mask, is_causal=False)
            logits = outputs.logits
            
            # Shift logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            
            # Collect logits at each step
            student_logits_list.append(logits)
            student_mask_positions_list.append(mask_index)
            student_states_list.append(x_t_student.clone())
            
            # Update x_t using same algorithm as teacher
            if ALG == "p2":
                kappa_t = (step + 1) / STUDENT_STEPS
                
                probs = F.softmax(logits / DIFFUSION_TEMP, dim=-1)
                confidence = probs.max(dim=-1).values
                x0_full = logits.argmax(dim=-1)
                
                full_conf = confidence.clone()
                full_conf[fix_mask] = float("inf")
                full_conf = torch.where(
                    torch.isfinite(full_conf), full_conf, torch.full_like(full_conf, float("inf"))
                )
                
                num_positions = (~fix_mask).sum(dim=1)
                num_to_mask = (num_positions.float() * (1.0 - kappa_t)).floor().to(torch.long)
                num_to_mask = num_to_mask.clamp_min(0)
                num_to_mask = torch.minimum(num_to_mask, num_positions)
                
                sorted_idx = torch.argsort(full_conf, dim=1, descending=False)
                max_k = int(num_to_mask.max().item())
                
                if max_k > 0:
                    topk_idx = sorted_idx[:, :max_k]
                    row_mask = torch.arange(max_k, device=DEVICE).unsqueeze(0) < num_to_mask.unsqueeze(1)
                    
                    to_mask = torch.zeros_like(x_t_student, dtype=torch.bool)
                    batch_arange = torch.arange(x_t_student.size(0), device=DEVICE).unsqueeze(1).expand_as(topk_idx)
                    valid_batch = batch_arange[row_mask]
                    valid_col = topk_idx[row_mask]
                    to_mask[valid_batch, valid_col] = True
                else:
                    to_mask = torch.zeros_like(x_t_student, dtype=torch.bool)
                
                x_t_student = x_t_student.clone()  # Need clone for gradient tracking
                x_t_student[to_mask] = mask_token_id
                keep_unmask = mask_index & (~to_mask)
                x_t_student[keep_unmask] = x0_full[keep_unmask]
        
        # ========== COMPUTE DISTILLATION LOSS ==========
        # Match student logit at step i to teacher logit at step 2*i
        distill_losses = []
        for i in range(min(len(student_logits_list), len(teacher_logits_list))):
            student_logits = student_logits_list[i]
            teacher_logits = teacher_logits_list[i]  # Teacher logit at step 2*i
            
            student_mask = student_mask_positions_list[i]
            teacher_mask = teacher_mask_positions_list[i]
            
            # Only compute loss at masked positions
            # Extract logits at masked positions
            student_mask_logits = student_logits[student_mask]  # [num_masked, vocab_size]
            teacher_mask_logits = teacher_logits[teacher_mask]  # [num_masked, vocab_size]
            
            if student_mask_logits.numel() > 0 and teacher_mask_logits.numel() > 0:
                # Ensure same number of masked positions (pad if needed)
                min_masked = min(student_mask_logits.shape[0], teacher_mask_logits.shape[0])
                if min_masked > 0:
                    student_log_probs = F.log_softmax(student_mask_logits[:min_masked] / TEMPERATURE, dim=-1)
                    teacher_probs = F.softmax(teacher_mask_logits[:min_masked] / TEMPERATURE, dim=-1)
                    
                    step_distill_loss = F.kl_div(
                        student_log_probs,
                        teacher_probs,
                        reduction='mean',
                        log_target=False
                    ) * (TEMPERATURE ** 2)
                    
                    distill_losses.append(step_distill_loss)
        
        if distill_losses:
            distill_loss = torch.stack(distill_losses).mean()
        else:
            distill_loss = torch.tensor(0.0, device=DEVICE)
        
        # Standard student loss (cross-entropy on final prediction)
        # Use the last student state to compute standard loss
        if len(student_states_list) > 0:
            final_student_outputs = student_model(
                input_ids=student_states_list[-1],
                attention_mask=attention_mask,
                is_causal=False
            )
            # Create labels from original input
            labels = batch_input_ids.clone()
            labels[:, :prompt_len] = -100  # Ignore prompt
            student_loss = F.cross_entropy(
                final_student_outputs.logits.view(-1, final_student_outputs.logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
        else:
            student_loss = torch.tensor(0.0, device=DEVICE)
        
        # Combined loss
        combined_loss = (1 - DISTILLATION_ALPHA) * student_loss + DISTILLATION_ALPHA * distill_loss
        
        # Backward pass
        optimizer.zero_grad()
        combined_loss.backward()
        optimizer.step()
        
        total_loss += combined_loss.item()
        total_student_loss += student_loss.item()
        total_distill_loss += distill_loss.item()
        num_batches_processed += 1
    
    print(f"\nEpoch {epoch+1} Summary:")
    print(f"  Total Loss: {total_loss / num_batches_processed:.4f}")
    print(f"  Student Loss: {total_student_loss / num_batches_processed:.4f}")
    print(f"  Distill Loss: {total_distill_loss / num_batches_processed:.4f}")

# Save student model
print("\nSaving student model...")
student_model.save_pretrained("./distilled_student_model")
student_tokenizer.save_pretrained("./distilled_student_model")
print("Done! Student model saved to ./distilled_student_model")
