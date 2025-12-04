"""
Simple Knowledge Distillation Script for Open-dCoder

A minimal, self-contained script for distilling open-dcoder models.
Just loads models from HuggingFace and trains with distillation loss.

Usage:
    python simple_distill.py
"""

import copy
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

# Configuration
TEACHER_MODEL = "fredzzp/open-dcoder-0.5B"  # Teacher model (can be larger)
STUDENT_MODEL = "fredzzp/open-dcoder-0.5B"  # Student model (can be smaller or same)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
LEARNING_RATE = 5e-5
NUM_EPOCHS = 1
DISTILLATION_ALPHA = 0.5  # Weight for distillation loss
TEMPERATURE = 4.0  # Temperature for softmax
MAX_SEQ_LEN = 512

print(f"Loading teacher model: {TEACHER_MODEL}")
teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL, trust_remote_code=True)
teacher_model = AutoModelForCausalLM.from_pretrained(
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
student_model = AutoModelForCausalLM.from_pretrained(
    STUDENT_MODEL,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).to(DEVICE).train()

# Add mask token if needed
if student_tokenizer.mask_token is None:
    student_tokenizer.add_special_tokens({"mask_token": "<M>"})
    student_model.resize_token_embeddings(len(student_tokenizer))

# Deep copy teacher weights to student (optional - only if architectures match)
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
    print(f"Could not copy weights (architectures may differ): {e}")

# Freeze teacher
for param in teacher_model.parameters():
    param.requires_grad = False

# Setup optimizer
optimizer = torch.optim.AdamW(student_model.parameters(), lr=LEARNING_RATE)

# Load dataset (using a simple code dataset)
print("Loading dataset...")
try:
    # Try to load a code dataset
    dataset = load_dataset("bigcode/the-stack-dedup", data_dir="data/python", split="train[:1000]")
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
    print("Creating dummy data for demonstration...")
    # Create dummy data
    dummy_texts = ["def hello():", "def world():", "def test():", "def example():"]
    tokenized_dataset = teacher_tokenizer(
        dummy_texts * 100,  # Repeat to have some data
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
        return_tensors="pt"
    )

# Training loop
print("Starting training...")
mask_token_id = teacher_tokenizer.mask_token_id

for epoch in range(NUM_EPOCHS):
    total_loss = 0
    total_student_loss = 0
    total_distill_loss = 0
    
    # Create batches
    if isinstance(tokenized_dataset, dict):
        # If it's already tokenized as dict
        input_ids = tokenized_dataset["input_ids"]
        num_batches = (len(input_ids) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for i in tqdm(range(num_batches), desc=f"Epoch {epoch+1}"):
            start_idx = i * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, len(input_ids))
            batch_input_ids = input_ids[start_idx:end_idx].to(DEVICE)
            
            # Create labels (shifted for next token prediction)
            labels = batch_input_ids.clone()
            labels[:, :-1] = labels[:, 1:].clone()
            labels[:, -1] = -100  # Ignore last token
            
            # Create masked version for teacher (randomly mask some tokens)
            masked_input = batch_input_ids.clone()
            mask_ratio = 0.15
            mask_indices = torch.rand(masked_input.shape, device=DEVICE) < mask_ratio
            masked_input[mask_indices] = mask_token_id
            
            # Student forward pass
            student_outputs = student_model(input_ids=masked_input, labels=labels)
            student_loss = student_outputs.loss
            student_logits = student_outputs.logits
            
            # Teacher forward pass (no gradients)
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_ids=masked_input,
                    is_causal=False  # Bidirectional for diffusion models
                )
                teacher_logits = teacher_outputs.logits
                # Shift logits to align with training
                teacher_logits = torch.cat([teacher_logits[:, :1], teacher_logits[:, :-1]], dim=1)
            
            # Distillation loss (KL divergence)
            student_log_probs = F.log_softmax(student_logits / TEMPERATURE, dim=-1)
            teacher_probs = F.softmax(teacher_logits / TEMPERATURE, dim=-1)
            
            distill_loss = F.kl_div(
                student_log_probs.view(-1, student_log_probs.size(-1)),
                teacher_probs.view(-1, teacher_probs.size(-1)),
                reduction='batchmean'
            ) * (TEMPERATURE ** 2)
            
            # Combined loss
            combined_loss = (1 - DISTILLATION_ALPHA) * student_loss + DISTILLATION_ALPHA * distill_loss
            
            # Backward pass
            optimizer.zero_grad()
            combined_loss.backward()
            optimizer.step()
            
            total_loss += combined_loss.item()
            total_student_loss += student_loss.item()
            total_distill_loss += distill_loss.item()
    
    else:
        # If it's a HuggingFace dataset
        dataloader = torch.utils.data.DataLoader(
            tokenized_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(DEVICE)
            
            # Create labels
            labels = input_ids.clone()
            labels[:, :-1] = labels[:, 1:].clone()
            labels[:, -1] = -100
            
            # Create masked version
            masked_input = input_ids.clone()
            mask_ratio = 0.15
            mask_indices = torch.rand(masked_input.shape, device=DEVICE) < mask_ratio
            masked_input[mask_indices] = mask_token_id
            
            # Student forward
            student_outputs = student_model(input_ids=masked_input, labels=labels)
            student_loss = student_outputs.loss
            student_logits = student_outputs.logits
            
            # Teacher forward
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_ids=masked_input,
                    is_causal=False
                )
                teacher_logits = teacher_outputs.logits
                teacher_logits = torch.cat([teacher_logits[:, :1], teacher_logits[:, :-1]], dim=1)
            
            # Distillation loss
            student_log_probs = F.log_softmax(student_logits / TEMPERATURE, dim=-1)
            teacher_probs = F.softmax(teacher_logits / TEMPERATURE, dim=-1)
            
            distill_loss = F.kl_div(
                student_log_probs.view(-1, student_log_probs.size(-1)),
                teacher_probs.view(-1, teacher_probs.size(-1)),
                reduction='batchmean'
            ) * (TEMPERATURE ** 2)
            
            # Combined loss
            combined_loss = (1 - DISTILLATION_ALPHA) * student_loss + DISTILLATION_ALPHA * distill_loss
            
            # Backward
            optimizer.zero_grad()
            combined_loss.backward()
            optimizer.step()
            
            total_loss += combined_loss.item()
            total_student_loss += student_loss.item()
            total_distill_loss += distill_loss.item()
    
    print(f"\nEpoch {epoch+1} Summary:")
    print(f"  Total Loss: {total_loss / (num_batches if isinstance(tokenized_dataset, dict) else len(dataloader)):.4f}")
    print(f"  Student Loss: {total_student_loss / (num_batches if isinstance(tokenized_dataset, dict) else len(dataloader)):.4f}")
    print(f"  Distill Loss: {total_distill_loss / (num_batches if isinstance(tokenized_dataset, dict) else len(dataloader)):.4f}")

# Save student model
print("\nSaving student model...")
student_model.save_pretrained("./distilled_student_model")
student_tokenizer.save_pretrained("./distilled_student_model")
print("Done! Student model saved to ./distilled_student_model")

