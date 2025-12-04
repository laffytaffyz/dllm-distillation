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
# Add parent directory (repo root) to path so we can import veomni
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)  # Go up one level from distillation/ to repo root
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from veomni.models.transformers.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from veomni.models.transformers.qwen2.generation_utils import MDMGenerationConfig, sample_tokens

# INIT TEACHER AND STUDENTS --------------------
# Configuration
TEACHER_MODEL = "fredzzp/open-dcoder-0.5B"
STUDENT_MODEL = "fredzzp/open-dcoder-0.5B"  # Can be same or different
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1  # Very small batch size due to memory requirements (2 models + diffusion steps)
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


# DATASET LOADING --------------------
# Load dataset - using FineCode (same as Open-dLLM repo)
print("Loading FineCode dataset (fredzzp/fine_code)...")
print("Dataset size: ~95.7 GB (43.2M examples, 192 files)")
print("To download locally, run:")
print("  python3 scripts/download_hf_data.py --repo_id fredzzp/fine_code --local_dir ./data --max_workers 4")
print()

DATASET_NAME = "fredzzp/fine_code"
USE_STREAMING = True  # Set to False to use local download (requires ~96GB free space)
LOCAL_DATA_DIR = "./data/fine_code"  # Path to downloaded data

try:
    if USE_STREAMING:
        # Streaming mode - loads on-the-fly, no download
        print("Using streaming mode (no local download)...")
        dataset = load_dataset(DATASET_NAME, streaming=True, split="train")
        # Take first N samples for training
        dataset_samples = []
        for i, sample in enumerate(dataset):
            if i >= 1000:  # Limit for demo - remove this for full training
                break
            dataset_samples.append(sample)
        
        # Extract text - try common keys
        texts = []
        for sample in dataset_samples:
            text = sample.get("text") or sample.get("content") or sample.get("code") or ""
            if text:
                texts.append(text)
    else:
        # Load from local directory if downloaded
        print(f"Loading from local directory: {LOCAL_DATA_DIR}")
        try:
            # Try loading from local parquet files (FineCode format)
            import glob
            parquet_files = glob.glob(f"{LOCAL_DATA_DIR}/data/train-*.parquet")
            if parquet_files:
                print(f"Found {len(parquet_files)} parquet files locally")
                # Load first few files for demo (remove limit for full training)
                dataset = load_dataset("parquet", data_files=parquet_files[:10], split="train[:1000]")
            else:
                # Try JSONL format
                jsonl_files = glob.glob(f"{LOCAL_DATA_DIR}/data/*.jsonl")
                if jsonl_files:
                    dataset = load_dataset("json", data_files=jsonl_files[:10], split="train[:1000]")
                else:
                    raise FileNotFoundError(f"No data files found in {LOCAL_DATA_DIR}")
        except Exception as e:
            print(f"Could not load from local directory: {e}")
            print("Falling back to HuggingFace download...")
            dataset = load_dataset(DATASET_NAME, split="train[:1000]")
        
        # Extract text - try common keys
        text_key = "text" if "text" in dataset.column_names else (
            "content" if "content" in dataset.column_names else dataset.column_names[0]
        )
        texts = dataset[text_key]
    
    # Filter out empty texts
    texts = [t for t in texts if t and len(t.strip()) > 0]
    
    if len(texts) == 0:
        raise ValueError("No valid text samples found in dataset")
    
    # Tokenize
    print(f"Tokenizing {len(texts)} samples...")
    tokenized_output = teacher_tokenizer(
        texts,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
        return_tensors="pt"
    )
    # Extract input_ids tensor - this is what we need for training
    tokenized_dataset = {"input_ids": tokenized_output["input_ids"]}
    print(f"Successfully loaded and tokenized {len(texts)} samples")
    
except Exception as e:
    print(f"Could not load FineCode dataset: {e}")
    print("\nTo download FineCode dataset:")
    print("  python3 scripts/download_hf_data.py --repo_id fredzzp/fine_code --local_dir ./data")
    print("\nOr use streaming mode (set USE_STREAMING=True in script)")
    print("\nCreating dummy data for demonstration...")
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
            # Move to CPU to save GPU memory (will move back when needed)
            if step % collect_every_n == 0:
                collected_logits.append(logits.detach().cpu())
                collected_mask_positions.append(mask_index.cpu())
                collected_states.append(x_t.cpu())
            
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
# Extract input_ids tensor from tokenized output
if isinstance(tokenized_dataset, dict):
    input_ids = tokenized_dataset["input_ids"]
else:
    # Handle BatchEncoding or other tokenizer output types
    input_ids = tokenized_dataset["input_ids"] if hasattr(tokenized_dataset, "__getitem__") else tokenized_dataset.input_ids

# Ensure input_ids is a tensor
if not isinstance(input_ids, torch.Tensor):
    input_ids = torch.tensor(input_ids)

num_batches = (len(input_ids) + BATCH_SIZE - 1) // BATCH_SIZE
dataset_type = "dict"

for epoch in range(NUM_EPOCHS):
    total_loss = 0
    total_student_loss = 0
    total_distill_loss = 0
    num_batches_processed = 0
    
    batch_iter = range(num_batches)
    
    for batch_idx in tqdm(batch_iter, desc=f"Epoch {epoch+1}"):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(input_ids))
        batch_input_ids = input_ids[start_idx:end_idx].to(DEVICE)
        
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
        
        # Clear GPU cache after teacher forward pass
        torch.cuda.empty_cache()
        
        # ========== STUDENT: Run 64 steps, collect at each step ==========
        student_logits_list = []
        student_mask_positions_list = []
        student_states_list = []
        
        # Initialize student's x_t from teacher's first state (same starting point)
        # Move back to GPU
        x_t_student = teacher_states_list[0].to(DEVICE)
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
            # Move teacher logits back to GPU for loss computation
            teacher_logits = teacher_logits_list[i].to(DEVICE)  # Teacher logit at step 2*i
            
            student_mask = student_mask_positions_list[i]
            teacher_mask = teacher_mask_positions_list[i].to(DEVICE)
            
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
            # Move final state back to GPU
            final_student_outputs = student_model(
                input_ids=student_states_list[-1].to(DEVICE),
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
        
        # Store loss values before deleting tensors
        loss_val = combined_loss.item()
        student_loss_val = student_loss.item()
        distill_loss_val = distill_loss.item()
        
        # Clear GPU cache and delete intermediate tensors to free memory
        del teacher_logits_list, teacher_mask_positions_list, teacher_states_list
        del student_logits_list, student_mask_positions_list, student_states_list
        del combined_loss, student_loss, distill_loss
        torch.cuda.empty_cache()
        
        total_loss += loss_val
        total_student_loss += student_loss_val
        total_distill_loss += distill_loss_val
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
