"""
Step-wise Diffusion Distillation Script for Open-dCoder

Distills a teacher model (64 steps) to a student model (32 steps).
Teacher logits are collected every 2 steps, and student learns to match them.

This script continues distillation from a checkpoint (64→32 step distillation).
Usage:
    python simple_distill_local_parallel.py
    or
    accelerate launch simple_distill_local_parallel.py
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader, IterableDataset
import os
import json

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

# Initialize Accelerate
accelerator = Accelerator()
if accelerator.is_main_process:
    print(f"[ accelerate ] Using {accelerator.num_processes} GPUs")
    print(f"[ accelerate ] Mixed precision: {accelerator.mixed_precision}")

# Initialize veomni parallel state to match Accelerate's distributed setup
# This prevents errors when veomni models check parallel state
# Note: Accelerate may not initialize torch.distributed until prepare() is called,
# so we'll initialize veomni parallel state after prepare() instead

# INIT TEACHER AND STUDENTS --------------------
# Configuration
# Model paths: Can be HuggingFace model ID (e.g., "fredzzp/open-dcoder-0.5B") or local path
# Set to None to use HuggingFace model ID, or provide local path for faster loading
TEACHER_MODEL_PATH = '/orcd/data/tpoggio/001/tiffany8/dllm-distillation/models/open-dcoder-0.5B'  # e.g., "/path/to/local/open-dcoder-0.5B" or None for HuggingFace
STUDENT_MODEL_PATH = '/orcd/data/tpoggio/001/tiffany8/dllm-distillation/models/open-dcoder-0.5B'  # e.g., "/path/to/local/open-dcoder-0.5B" or None for HuggingFace
TEACHER_MODEL = "fredzzp/open-dcoder-0.5B"  # HuggingFace model ID (used if TEACHER_MODEL_PATH is None)
STUDENT_MODEL = "fredzzp/open-dcoder-0.5B"  # HuggingFace model ID (used if STUDENT_MODEL_PATH is None)
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2  # Accumulate gradients over N mini-batches (effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * num_gpus)

# Use accelerator device
DEVICE = accelerator.device 
LEARNING_RATE = 5e-5
NUM_EPOCHS = 1
DISTILLATION_ALPHA = 0.5  # Weight for distillation loss
TEMPERATURE = 4.0  # Temperature for softmax
MAX_SEQ_LEN = 256  # Reduced for memory efficiency

# Diffusion parameters
TEACHER_STEPS = 64  # Teacher runs 64 steps
STUDENT_STEPS = 32   # Student runs 32 steps
COLLECT_EVERY_N = 2  # Collect teacher logits every 2 steps
ALG = "p2"  # Sampling algorithm
DIFFUSION_TEMP = 0.7  # Temperature for diffusion sampling
TOP_K = 200

# Dataset parameters
# Set MAX_SAMPLES to match the number of samples in 1600 batches (same as previous distillation stage)
MAX_SAMPLES = 1600 * BATCH_SIZE  # 1600 batches * batch_size

# Checkpoint parameters
CHECKPOINT_DIR = "./checkpoints/64to32steps"
SAVE_CHECKPOINT_EVERY_N_BATCHES = 200  # Save checkpoint every N batches (0 to disable)
SAVE_CHECKPOINT_EVERY_EPOCH = True  # Save checkpoint at end of each epoch
RESUME_FROM_CHECKPOINT = "./checkpoints/latest_checkpoint_64_steps.pt"  # Path to checkpoint to resume from (64→32 distillation checkpoint)

# Rollout printing parameters
PRINT_ROLLOUTS_EVERY_N_BATCHES = 100  # Print teacher vs student rollouts every N batches (0 to disable)
PRINT_ROLLOUTS_AT_STEPS = [0, 32, 63]  # Which student steps to print (0-indexed, max is STUDENT_STEPS-1)

# Use local path if provided, otherwise use HuggingFace model ID
teacher_model_path = TEACHER_MODEL_PATH if TEACHER_MODEL_PATH is not None else TEACHER_MODEL
if accelerator.is_main_process:
    print(f"Loading teacher model: {teacher_model_path} {'(local)' if TEACHER_MODEL_PATH else '(HuggingFace)'}")
teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path, trust_remote_code=True)
teacher_model = Qwen2ForCausalLM.from_pretrained(
    teacher_model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).to(DEVICE).eval()

# Add mask token if needed
if teacher_tokenizer.mask_token is None:
    teacher_tokenizer.add_special_tokens({"mask_token": "<M>"})
    teacher_model.resize_token_embeddings(len(teacher_tokenizer))

# Use local path if provided, otherwise use HuggingFace model ID
student_model_path = STUDENT_MODEL_PATH if STUDENT_MODEL_PATH is not None else STUDENT_MODEL
if accelerator.is_main_process:
    print(f"Loading student model: {student_model_path} {'(local)' if STUDENT_MODEL_PATH else '(HuggingFace)'}")
student_tokenizer = AutoTokenizer.from_pretrained(student_model_path, trust_remote_code=True)
student_model = Qwen2ForCausalLM.from_pretrained(
    student_model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).to(DEVICE).train()

# Add mask token if needed
if student_tokenizer.mask_token is None:
    student_tokenizer.add_special_tokens({"mask_token": "<M>"})
    student_model.resize_token_embeddings(len(student_tokenizer))

# Deep copy teacher weights to student
if accelerator.is_main_process:
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
    if accelerator.is_main_process:
        print(f"Copied {copied} parameters from teacher to student")
except Exception as e:
    if accelerator.is_main_process:
        print(f"Could not copy weights: {e}")

# Freeze teacher
for param in teacher_model.parameters():
    param.requires_grad = False

# Setup optimizer
optimizer = torch.optim.AdamW(student_model.parameters(), lr=LEARNING_RATE)


# Create IterableDataset class for streaming mode
class StreamingTokenizedDataset(IterableDataset):
    """Dataset that tokenizes on-the-fly for streaming mode"""
    def __init__(self, dataset, tokenizer, max_length, max_samples=None, skip_samples=0):
        self.dataset = dataset  # This is the streaming dataset iterator
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_samples = max_samples  # Limit number of samples
        self.skip_samples = skip_samples  # Number of samples to skip before yielding
    
    def __iter__(self):
        """Iterate through streaming dataset and tokenize on-the-fly"""
        samples_skipped = 0
        samples_yielded = 0
        
        for sample in self.dataset:
            # Skip samples if we haven't reached skip_samples yet
            if samples_skipped < self.skip_samples:
                samples_skipped += 1
                continue
            
            # Check if we've reached the limit
            if self.max_samples is not None and samples_yielded >= self.max_samples:
                break
            
            # Extract text from sample
            text = sample.get("text") or sample.get("content") or sample.get("code") or ""
            if not text or len(text.strip()) == 0:
                # Skip empty samples
                continue
            
            # Tokenize on-the-fly
            tokenized = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            yield {"input_ids": tokenized["input_ids"].squeeze(0)}
            samples_yielded += 1


# Note: TextDataset removed - now using streaming for both HuggingFace and local files
# Streaming is faster and uses less memory for large datasets


# Checkpoint loading function (defined early so it can be used before dataset creation)
def load_checkpoint(checkpoint_path, student_model, optimizer, device, accelerator):
    """Load a training checkpoint"""
    if accelerator.is_main_process:
        print(f"Loading checkpoint from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state (will be wrapped by accelerator.prepare later if needed)
    unwrapped_model = accelerator.unwrap_model(student_model) if hasattr(accelerator, 'unwrap_model') else student_model
    unwrapped_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint['epoch']
    start_batch = checkpoint.get('batch_idx', 0)
    
    if accelerator.is_main_process:
        print(f"Resumed from epoch {start_epoch}, batch {start_batch}")
        print(f"  Total Loss: {checkpoint.get('total_loss', 'N/A'):.4f}")
        print(f"  Student Loss: {checkpoint.get('student_loss', 'N/A'):.4f}")
        print(f"  Distill Loss: {checkpoint.get('distill_loss', 'N/A'):.4f}")
    
    return start_epoch, start_batch


# DATASET LOADING --------------------
# Load dataset - using FineCode (same as Open-dLLM repo)
print("Loading FineCode dataset (fredzzp/fine_code)...")
print("Dataset size: ~95.7 GB (43.2M examples, 192 files)")

DATASET_NAME = "fredzzp/fine_code"
USE_STREAMING = False  # Set to False to use local download (requires ~96GB free space)
LOCAL_DATA_DIR = "/orcd/data/tpoggio/001/tiffany8/dllm-distillation/data"  # Path to downloaded data

# Skip samples will be calculated after checkpoint is loaded (if resuming)
# This allows us to skip batches we've already seen
SKIP_SAMPLES = 0  # Will be updated based on checkpoint

# Load checkpoint early to determine SKIP_SAMPLES before creating dataset
# This ensures we skip the correct number of samples when resuming
start_epoch = 0
start_batch_idx = 0
if RESUME_FROM_CHECKPOINT is not None:
    if RESUME_FROM_CHECKPOINT == "latest":
        # Auto-resume from latest checkpoint if it exists
        latest_checkpoint = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pt")
        if os.path.exists(latest_checkpoint):
            if accelerator.is_main_process:
                print(f"Auto-resuming from latest checkpoint: {latest_checkpoint}")
            start_epoch, start_batch_idx = load_checkpoint(latest_checkpoint, student_model, optimizer, DEVICE, accelerator)
            
            # For self-distillation continuation: copy checkpoint weights from student to teacher
            if accelerator.is_main_process:
                print("Copying checkpoint weights from student to teacher for self-distillation...")
            try:
                student_state_dict = student_model.state_dict()
                teacher_state_dict = teacher_model.state_dict()
                
                copied = 0
                for name, param in student_state_dict.items():
                    if name in teacher_state_dict and param.shape == teacher_state_dict[name].shape:
                        teacher_state_dict[name].data.copy_(param.data)
                        copied += 1
                
                teacher_model.load_state_dict(teacher_state_dict, strict=False)
                if accelerator.is_main_process:
                    print(f"Copied {copied} parameters from checkpoint (student) to teacher")
            except Exception as e:
                if accelerator.is_main_process:
                    print(f"Warning: Could not copy checkpoint weights to teacher: {e}")
            
            SKIP_SAMPLES = start_batch_idx * BATCH_SIZE
            if accelerator.is_main_process:
                print(f"Skipping first {SKIP_SAMPLES:,} samples (equivalent to {start_batch_idx} batches with batch_size={BATCH_SIZE})")
        elif accelerator.is_main_process:
            print("No latest checkpoint found, starting from scratch")
    elif os.path.exists(RESUME_FROM_CHECKPOINT):
        start_epoch, start_batch_idx = load_checkpoint(RESUME_FROM_CHECKPOINT, student_model, optimizer, DEVICE, accelerator)
        
        # For self-distillation continuation: copy checkpoint weights from student to teacher
        # This ensures teacher uses the checkpoint model (64-step distilled model)
        if accelerator.is_main_process:
            print("Copying checkpoint weights from student to teacher for self-distillation...")
        try:
            student_state_dict = student_model.state_dict()
            teacher_state_dict = teacher_model.state_dict()
            
            copied = 0
            for name, param in student_state_dict.items():
                if name in teacher_state_dict and param.shape == teacher_state_dict[name].shape:
                    teacher_state_dict[name].data.copy_(param.data)
                    copied += 1
            
            teacher_model.load_state_dict(teacher_state_dict, strict=False)
            if accelerator.is_main_process:
                print(f"Copied {copied} parameters from checkpoint (student) to teacher")
        except Exception as e:
            if accelerator.is_main_process:
                print(f"Warning: Could not copy checkpoint weights to teacher: {e}")
        
        # Calculate number of samples to skip based on batches already seen
        # This assumes each batch uses BATCH_SIZE samples (accounting for gradient accumulation separately)
        # Note: In distributed training, each process sees a subset of data, but skip_samples should account for total samples seen
        SKIP_SAMPLES = start_batch_idx * BATCH_SIZE
        if accelerator.is_main_process:
            print(f"Skipping first {SKIP_SAMPLES:,} samples (equivalent to {start_batch_idx} batches with batch_size={BATCH_SIZE})")
    elif accelerator.is_main_process:
        print(f"Warning: Checkpoint path {RESUME_FROM_CHECKPOINT} does not exist, starting from scratch")

try:
    if USE_STREAMING:
        # Streaming mode - loads on-the-fly, no download
        if accelerator.is_main_process:
            print("Using streaming mode (no local download)...")
            print("Note: Streaming mode tokenizes on-the-fly during training")
        dataset = load_dataset(DATASET_NAME, streaming=True, split="train")
        
        # Create streaming dataset that tokenizes on-the-fly
        tokenized_dataset = StreamingTokenizedDataset(
            dataset,
            teacher_tokenizer,
            MAX_SEQ_LEN,
            max_samples=MAX_SAMPLES,
            skip_samples=SKIP_SAMPLES
        )
        if accelerator.is_main_process:
            if MAX_SAMPLES is not None:
                print(f"Streaming dataset ready (will tokenize on-the-fly, limited to {MAX_SAMPLES:,} samples)")
            else:
                print("Streaming dataset ready (will tokenize on-the-fly, using all available samples)")
    else:
        # Load from local directory if downloaded - use streaming for efficiency
        if accelerator.is_main_process:
            print(f"Loading from local directory: {LOCAL_DATA_DIR}")
            print("Note: Using streaming mode for local files (tokenizes on-the-fly, lower memory usage)")
        # Try loading from local parquet files (FineCode format)
        import glob
        parquet_files = glob.glob(f"{LOCAL_DATA_DIR}/data/train-*.parquet")
        if parquet_files:
            if accelerator.is_main_process:
                print(f"Found {len(parquet_files)} parquet files locally")
            # Use streaming mode for local parquet files - faster and lower memory
            dataset = load_dataset("parquet", data_files=parquet_files, split="train", streaming=True)
        else:
            # Try JSONL format
            jsonl_files = glob.glob(f"{LOCAL_DATA_DIR}/data/*.jsonl")
            if jsonl_files:
                # Use streaming mode for local JSONL files
                dataset = load_dataset("json", data_files=jsonl_files, split="train", streaming=True)
            else:
                raise FileNotFoundError(
                    f"No data files found in {LOCAL_DATA_DIR}\n"
                    f"Expected parquet files at {LOCAL_DATA_DIR}/data/train-*.parquet\n"
                    f"or JSONL files at {LOCAL_DATA_DIR}/data/*.jsonl\n"
                    f"To download the dataset, run:\n"
                    f"  python3 scripts/download_hf_data.py --repo_id fredzzp/fine_code --local_dir ./data"
                )
        
        # Create streaming dataset that tokenizes on-the-fly (same as HuggingFace streaming)
        tokenized_dataset = StreamingTokenizedDataset(
            dataset,
            teacher_tokenizer,
            MAX_SEQ_LEN,
            max_samples=MAX_SAMPLES,
            skip_samples=SKIP_SAMPLES
        )
        if accelerator.is_main_process:
            if MAX_SAMPLES is not None:
                print(f"Local streaming dataset ready (will tokenize on-the-fly, limited to {MAX_SAMPLES:,} samples)")
            else:
                print("Local streaming dataset ready (will tokenize on-the-fly, using all available samples)")
    
except Exception as e:
    if accelerator.is_main_process:
        print(f"Could not load FineCode dataset: {e}")
        print("\nTo download FineCode dataset:")
        print("  python3 scripts/download_hf_data.py --repo_id fredzzp/fine_code --local_dir ./data")
        print("\nOr use streaming mode (set USE_STREAMING=True in script)")
    raise  # Re-raise the exception instead of falling back to dummy data



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


# Checkpoint saving and loading functions
def save_checkpoint(student_model, optimizer, epoch, batch_idx, total_loss, student_loss, distill_loss, 
                   checkpoint_dir, tokenizer, accelerator):
    """Save a training checkpoint (only on main process)"""
    if not accelerator.is_main_process:
        return
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Unwrap model before saving
    unwrapped_model = accelerator.unwrap_model(student_model)
    
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}_batch_{batch_idx}.pt")
    checkpoint_info_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}_batch_{batch_idx}.json")
    
    # Save model and optimizer state
    torch.save({
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': unwrapped_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'total_loss': total_loss,
        'student_loss': student_loss,
        'distill_loss': distill_loss,
    }, checkpoint_path)
    
    # Save metadata as JSON for easy inspection
    with open(checkpoint_info_path, 'w') as f:
        json.dump({
            'epoch': epoch,
            'batch_idx': batch_idx,
            'total_loss': float(total_loss),
            'student_loss': float(student_loss),
            'distill_loss': float(distill_loss),
        }, f, indent=2)
    
    # Also save latest checkpoint
    latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
    latest_info_path = os.path.join(checkpoint_dir, "latest_checkpoint.json")
    torch.save({
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': unwrapped_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'total_loss': total_loss,
        'student_loss': student_loss,
        'distill_loss': distill_loss,
    }, latest_path)
    
    with open(latest_info_path, 'w') as f:
        json.dump({
            'epoch': epoch,
            'batch_idx': batch_idx,
            'total_loss': float(total_loss),
            'student_loss': float(student_loss),
            'distill_loss': float(distill_loss),
        }, f, indent=2)
    
    print(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


# Create DataLoader (before prepare - will be prepared by accelerator)
# tokenized_dataset is always StreamingTokenizedDataset (for both HuggingFace and local files)
# Both modes now use streaming for efficiency - tokenizes on-the-fly, lower memory usage
dataloader = DataLoader(
    tokenized_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,  # Can't shuffle streaming datasets (they're iterators)
    pin_memory=True,
    num_workers=0  # Set to 0 to avoid multiprocessing issues
)
# elif RESUME_FROM_CHECKPOINT is None:
#     # Try to resume from latest checkpoint if it exists
#     latest_checkpoint = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pt")
#     if os.path.exists(latest_checkpoint) and accelerator.is_main_process:
#         print(f"Found latest checkpoint at {latest_checkpoint}")
#         response = input("Resume from latest checkpoint? (y/n): ").strip().lower()
#         if response == 'y':
#             start_epoch, start_batch_idx = load_checkpoint(latest_checkpoint, student_model, optimizer, DEVICE, accelerator)
#     # Synchronize across all processes
#     accelerator.wait_for_everyone()

# Prepare models, optimizer, and dataloader with Accelerate
# Note: teacher_model is not prepared since it's frozen and only used for inference
student_model, optimizer, dataloader = accelerator.prepare(
    student_model, optimizer, dataloader
)

# Initialize veomni parallel state AFTER Accelerate prepares (when torch.distributed is initialized)
# This prevents errors when veomni models check parallel state
import torch.distributed as dist
from veomni.distributed.parallel_state import init_parallel_state

if dist.is_initialized():
    world_size = dist.get_world_size()
    if accelerator.is_main_process:
        print(f"[ veomni ] Initializing parallel state for {world_size} processes (data parallel only)")
    # All processes must call init_parallel_state
    init_parallel_state(
        dp_size=world_size,  # Match Accelerate's world size
        tp_size=1,           # No tensor parallelism
        ep_size=1,           # No expert parallelism
        pp_size=1,           # No pipeline parallelism
        cp_size=1,           # No context parallelism
        ulysses_size=1,      # No sequence parallelism
        dp_mode="ddp",       # Data parallel mode (matches Accelerate)
        device_type="cuda",
        include_sp_in_fsdp=True,
    )

# Training loop
if accelerator.is_main_process:
    print("Starting training...")
mask_token_id = teacher_tokenizer.mask_token_id

for epoch in range(start_epoch, NUM_EPOCHS):
    total_loss = 0
    total_student_loss = 0
    total_distill_loss = 0
    num_batches_processed = 0
    
    # Use tqdm only on main process
    if accelerator.is_main_process:
        batch_iter = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    else:
        batch_iter = dataloader
    
    # Skip batches if resuming from checkpoint
    # Note: We've already skipped samples in StreamingTokenizedDataset, so we just need to track batch_idx correctly
    if epoch == start_epoch and start_batch_idx > 0 and accelerator.is_main_process:
        print(f"Resuming training from batch {start_batch_idx} (skipped {SKIP_SAMPLES:,} samples in dataset)")
    
    # Initialize gradient accumulation counter
    accumulation_step = 0
    optimizer.zero_grad()  # Initialize gradients at start of epoch
    
    for batch_idx, batch in enumerate(batch_iter):
        # Adjust batch_idx if resuming (though streaming makes this approximate)
        if epoch == start_epoch:
            actual_batch_idx = start_batch_idx + batch_idx
        else:
            actual_batch_idx = batch_idx
        
        # Both HuggingFace and local files now use streaming - already tokenized in StreamingTokenizedDataset
        batch_input_ids = batch["input_ids"].to(DEVICE)
        
        # Prepare input: use first part as prompt, rest will be masked
        prompt_len = batch_input_ids.shape[1] // 2  # Use half as prompt
        prompt_ids = batch_input_ids[:, :prompt_len]
        
        # ========== TEACHER: Run 64 steps, collect every 2 steps ==========
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
        
        # We should have 32 teacher logits (64 steps / 2)
        assert len(teacher_logits_list) == STUDENT_STEPS, f"Expected {STUDENT_STEPS} teacher logits, got {len(teacher_logits_list)}"
        
        # Get teacher final output for comparison
        teacher_final_state = teacher_states_list[-1] if len(teacher_states_list) > 0 else None
        
        # Clear GPU cache after teacher forward pass
        torch.cuda.empty_cache()
        
        # ========== STUDENT: Run 32 steps, collect at each step ==========
        student_logits_list = []
        student_mask_positions_list = []
        student_states_list = []
        
        # Initialize student's x_t from teacher's first state (same starting point)
        # Move back to GPU (only when needed)
        x_t_student = teacher_states_list[0].to(DEVICE)
        fix_mask = (x_t_student != mask_token_id)
        attention_mask = (x_t_student != student_tokenizer.pad_token_id).long() if student_tokenizer.pad_token_id is not None else None
        
        # Note: teacher_states_list is kept on CPU for rollout printing later
        # It will be deleted after rollout printing is complete
        
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
            
            # Collect logits at each step (move to CPU to save GPU memory)
            student_logits_list.append(logits.detach().cpu())
            student_mask_positions_list.append(mask_index.cpu())
            student_states_list.append(x_t_student.cpu().clone())
            
            # Update x_t using same algorithm as teacher
            if ALG == "p2":
                kappa_t = (step + 1) / STUDENT_STEPS
                
                # Compute softmax in chunks to reduce memory usage
                # Process logits in smaller chunks to avoid OOM
                logits_scaled = logits / DIFFUSION_TEMP
                # Use in-place operations where possible
                probs = F.softmax(logits_scaled, dim=-1)
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
            
            # Print rollouts at specified steps (only on main process)
            if PRINT_ROLLOUTS_EVERY_N_BATCHES > 0 and actual_batch_idx % PRINT_ROLLOUTS_EVERY_N_BATCHES == 0 and accelerator.is_main_process:
                if step in PRINT_ROLLOUTS_AT_STEPS:
                    # Get corresponding teacher step (step * 2 since teacher collects every 2 steps)
                    teacher_step_idx = step
                    if teacher_step_idx < len(teacher_states_list):
                        teacher_state_at_step = teacher_states_list[teacher_step_idx]
                        student_state_at_step = x_t_student
                        
                        # Decode and print
                        teacher_tokens = teacher_state_at_step[0].cpu()  # First sample in batch
                        student_tokens = student_state_at_step[0].cpu()
                        
                        # Remove padding tokens for cleaner output
                        teacher_tokens = teacher_tokens[teacher_tokens != teacher_tokenizer.pad_token_id]
                        student_tokens = student_tokens[student_tokens != student_tokenizer.pad_token_id]
                        
                        teacher_text = teacher_tokenizer.decode(teacher_tokens, skip_special_tokens=False)
                        student_text = student_tokenizer.decode(student_tokens, skip_special_tokens=False)
                        
                        print(f"\n{'='*80}")
                        print(f"Batch {actual_batch_idx}, Student Step {step}/{STUDENT_STEPS-1} (Teacher Step {step*2}/{TEACHER_STEPS-1})")
                        print(f"{'='*80}")
                        print(f"TEACHER (Step {step*2}):")
                        print(f"{teacher_text[:500]}...")  # Limit to 500 chars
                        print(f"\nSTUDENT (Step {step}):")
                        print(f"{student_text[:500]}...")
                        print(f"{'='*80}\n")
        
        # Get student final output for comparison
        student_final_state = student_states_list[-1] if len(student_states_list) > 0 else None
        
        # Now we can safely delete teacher_states_list (rollout printing is done)
        # It was already on CPU, so this just frees CPU memory
        del teacher_states_list
        
        # ========== COMPUTE DISTILLATION LOSS ==========
        # Match student logit at step i to teacher logit at step 2*i
        distill_losses = []
        for i in range(min(len(student_logits_list), len(teacher_logits_list))):
            # Move logits to GPU only when needed for loss computation
            student_logits = student_logits_list[i].to(DEVICE)
            teacher_logits = teacher_logits_list[i].to(DEVICE)  # Teacher logit at step 2*i
            
            student_mask = student_mask_positions_list[i].to(DEVICE)
            teacher_mask = teacher_mask_positions_list[i].to(DEVICE)
            
            # Only compute loss at masked positions
            # Extract logits at masked positions
            student_mask_logits = student_logits[student_mask]  # [num_masked, vocab_size]
            teacher_mask_logits = teacher_logits[teacher_mask]  # [num_masked, vocab_size]
            
            if student_mask_logits.numel() > 0 and teacher_mask_logits.numel() > 0:
                # Ensure same number of masked positions (pad if needed)
                min_masked = min(student_mask_logits.shape[0], teacher_mask_logits.shape[0])
                
                # Move logits back to CPU after loss computation to free GPU memory
                # (We'll move them back when needed for loss computation)
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
                    
                    # Clear intermediate tensors from GPU
                    del student_log_probs, teacher_probs, step_distill_loss
            
            # Clear logits from GPU after processing
            del student_logits, teacher_logits, student_mask_logits, teacher_mask_logits, student_mask, teacher_mask
        
        # Clear cache once at end of loss computation loop (after all tensors are deleted)
        torch.cuda.empty_cache()
        
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
        
        # Scale loss by gradient accumulation steps (Accelerate handles this automatically)
        # This ensures the effective gradient is the same as if we used a larger batch size
        scaled_loss = combined_loss / GRADIENT_ACCUMULATION_STEPS
        
        # Backward pass with Accelerate (accumulates gradients)
        accelerator.backward(scaled_loss)
        
        # Increment accumulation counter
        accumulation_step += 1
        
        # Only update weights and zero gradients after accumulating enough steps
        if accumulation_step % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
            accumulation_step = 0
        
        # Store loss values before deleting tensors
        loss_val = combined_loss.item()
        student_loss_val = student_loss.item()
        distill_loss_val = distill_loss.item()
        
        # Clear GPU cache and delete intermediate tensors to free memory
        # Note: teacher_states_list was already deleted earlier (after rollout printing)
        del teacher_logits_list, teacher_mask_positions_list
        del student_logits_list, student_mask_positions_list, student_states_list
        del combined_loss, student_loss, distill_loss
        torch.cuda.empty_cache()
        
        total_loss += loss_val
        total_student_loss += student_loss_val
        total_distill_loss += distill_loss_val
        num_batches_processed += 1
        
        # Print final rollouts comparison (only on main process)
        if PRINT_ROLLOUTS_EVERY_N_BATCHES > 0 and actual_batch_idx % PRINT_ROLLOUTS_EVERY_N_BATCHES == 0 and accelerator.is_main_process:
            if teacher_final_state is not None and student_final_state is not None:
                # Decode final outputs
                teacher_final_tokens = teacher_final_state[0].cpu()
                student_final_tokens = student_final_state[0].cpu()
                
                # Remove padding
                teacher_final_tokens = teacher_final_tokens[teacher_final_tokens != teacher_tokenizer.pad_token_id]
                student_final_tokens = student_final_tokens[student_final_tokens != student_tokenizer.pad_token_id]
                
                teacher_final_text = teacher_tokenizer.decode(teacher_final_tokens, skip_special_tokens=False)
                student_final_text = student_tokenizer.decode(student_final_tokens, skip_special_tokens=False)
                
                print(f"\n{'='*80}")
                print(f"FINAL OUTPUTS - Batch {actual_batch_idx}")
                print(f"{'='*80}")
                print(f"TEACHER FINAL (after {TEACHER_STEPS} steps):")
                print(f"{teacher_final_text}")
                print(f"\nSTUDENT FINAL (after {STUDENT_STEPS} steps):")
                print(f"{student_final_text}")
                print(f"\nLoss: Total={loss_val:.4f}, Student={student_loss_val:.4f}, Distill={distill_loss_val:.4f}")
                print(f"{'='*80}\n")
        
        # Handle remaining accumulated gradients at end of batch loop (before checkpointing)
        if accumulation_step > 0:
            optimizer.step()
            optimizer.zero_grad()
            accumulation_step = 0
        
        # Save checkpoint periodically (only on main process)
        if SAVE_CHECKPOINT_EVERY_N_BATCHES > 0 and actual_batch_idx > 0 and actual_batch_idx % SAVE_CHECKPOINT_EVERY_N_BATCHES == 0:
            avg_loss = total_loss / num_batches_processed
            avg_student_loss = total_student_loss / num_batches_processed
            avg_distill_loss = total_distill_loss / num_batches_processed
            save_checkpoint(
                student_model, optimizer, epoch, actual_batch_idx,
                avg_loss, avg_student_loss, avg_distill_loss,
                CHECKPOINT_DIR, student_tokenizer, accelerator
            )
    
    # Save checkpoint at end of epoch (only on main process)
    if SAVE_CHECKPOINT_EVERY_EPOCH and num_batches_processed > 0:
        avg_loss = total_loss / num_batches_processed
        avg_student_loss = total_student_loss / num_batches_processed
        avg_distill_loss = total_distill_loss / num_batches_processed
        save_checkpoint(
            student_model, optimizer, epoch, num_batches_processed - 1,
            avg_loss, avg_student_loss, avg_distill_loss,
            CHECKPOINT_DIR, student_tokenizer, accelerator
        )
    
    if accelerator.is_main_process:
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Total Loss: {total_loss / num_batches_processed:.4f}")
        print(f"  Student Loss: {total_student_loss / num_batches_processed:.4f}")
        print(f"  Distill Loss: {total_distill_loss / num_batches_processed:.4f}")

# Save student model (only on main process)
if accelerator.is_main_process:
    print("\nSaving student model...")
    # Unwrap model before saving
    unwrapped_model = accelerator.unwrap_model(student_model)
    unwrapped_model.save_pretrained("./distilled_student_model")
    student_tokenizer.save_pretrained("./distilled_student_model")
    print("Done! Student model saved to ./distilled_student_model")

# Wait for all processes to finish
accelerator.wait_for_everyone()
