"""
Pre-compute Teacher Logits for Diffusion Distillation

This script pre-computes teacher model logits during diffusion sampling
and stores them to disk. This is useful when teacher forward passes are
slow and you want to iterate quickly during student training.

Usage:
    python precompute_teacher_logits.py \
        --teacher_model_path fredzzp/open-dcoder-0.5B \
        --data_path ./data \
        --output_path ./teacher_logits \
        --num_sampling_steps 200 \
        --collect_every_n_steps 10
"""

import argparse
import json
import os
import pickle
from functools import partial
from pathlib import Path
from typing import Dict, Any

import torch
from tqdm import tqdm

from veomni.data import build_iterative_dataset, build_mapping_dataset
from veomni.data.data_transform import process_pretrain_example
from veomni.models import build_foundation_model, build_tokenizer
from distillation.diffusion_logit_collector import (
    DiffusionLogitCollector,
    DiffusionLogitCollectionConfig,
)


def main():
    parser = argparse.ArgumentParser(description="Pre-compute teacher logits for distillation")
    parser.add_argument("--teacher_model_path", type=str, required=True,
                       help="Path to teacher model (HuggingFace ID or local path)")
    parser.add_argument("--teacher_config_path", type=str, default=None,
                       help="Path to teacher config (if different from model_path)")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to training data")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Output directory for stored logits")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                       help="Tokenizer path (defaults to teacher_model_path)")
    parser.add_argument("--num_sampling_steps", type=int, default=200,
                       help="Number of diffusion sampling steps")
    parser.add_argument("--collect_every_n_steps", type=int, default=10,
                       help="Collect logits every N steps")
    parser.add_argument("--diffusion_alg", type=str, default="p2",
                       help="Diffusion sampling algorithm")
    parser.add_argument("--diffusion_temperature", type=float, default=0.7,
                       help="Temperature for diffusion sampling")
    parser.add_argument("--diffusion_top_k", type=int, default=200,
                       help="Top-k for diffusion sampling")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for processing")
    parser.add_argument("--max_seq_len", type=int, default=2048,
                       help="Maximum sequence length")
    parser.add_argument("--max_examples", type=int, default=None,
                       help="Maximum number of examples to process (None = all)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(args.output_path, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Load tokenizer
    tokenizer_path = args.tokenizer_path or args.teacher_model_path
    tokenizer = build_tokenizer(tokenizer_path)
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "<M>"})
    
    # Load teacher model
    print(f"Loading teacher model from {args.teacher_model_path}...")
    teacher_config_path = args.teacher_config_path or args.teacher_model_path
    teacher_model = build_foundation_model(
        config_path=teacher_config_path,
        weights_path=args.teacher_model_path,
        torch_dtype="bfloat16",
        init_device=args.device,
    )
    teacher_model = teacher_model.to(args.device).eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    # Set up logit collector
    collection_config = DiffusionLogitCollectionConfig(
        num_sampling_steps=args.num_sampling_steps,
        temperature=args.diffusion_temperature,
        top_k=args.diffusion_top_k,
        alg=args.diffusion_alg,
        collect_every_n_steps=args.collect_every_n_steps,
        collect_mask_positions_only=True,
        store_on_cpu=True,
    )
    collector = DiffusionLogitCollector(
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        config=collection_config,
        device=args.device
    )
    
    # Load dataset
    print(f"Loading dataset from {args.data_path}...")
    transform = partial(
        process_pretrain_example,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        text_keys=["text"],
    )
    
    # Try mapping dataset first, fall back to iterable
    try:
        dataset = build_mapping_dataset(args.data_path, transform=transform)
        is_iterable = False
    except:
        dataset = build_iterative_dataset(args.data_path, transform=transform, seed=42)
        is_iterable = True
    
    # Process dataset
    print("Processing dataset and collecting teacher logits...")
    collected_data = []
    
    batch = []
    batch_indices = []
    
    for idx, example in enumerate(tqdm(dataset, desc="Collecting logits")):
        if args.max_examples and idx >= args.max_examples:
            break
        
        input_ids = example["input_ids"]
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids)
        
        batch.append(input_ids)
        batch_indices.append(idx)
        
        if len(batch) >= args.batch_size:
            # Process batch
            batch_tensor = torch.stack(batch).to(args.device)
            
            with torch.no_grad():
                logit_data = collector.collect_logits_for_batch(
                    input_ids=batch_tensor,
                    return_intermediate_states=False
                )
            
            # Store results
            for i, orig_idx in enumerate(batch_indices):
                example_data = {
                    "example_idx": orig_idx,
                    "input_ids": batch[i].cpu(),
                    "logits": [logits[i].cpu() if logits.dim() > 1 else logits.cpu() 
                              for logits in logit_data["logits"]],
                    "mask_positions": [mask[i].cpu() if mask.dim() > 1 else mask.cpu()
                                      for mask in logit_data["mask_positions"]],
                    "timesteps": logit_data["timesteps"],
                }
                collected_data.append(example_data)
            
            batch = []
            batch_indices = []
    
    # Process remaining batch
    if batch:
        batch_tensor = torch.stack(batch).to(args.device)
        with torch.no_grad():
            logit_data = collector.collect_logits_for_batch(
                input_ids=batch_tensor,
                return_intermediate_states=False
            )
        
        for i, orig_idx in enumerate(batch_indices):
            example_data = {
                "example_idx": orig_idx,
                "input_ids": batch[i].cpu(),
                "logits": [logits[i].cpu() if logits.dim() > 1 else logits.cpu() 
                          for logits in logit_data["logits"]],
                "mask_positions": [mask[i].cpu() if mask.dim() > 1 else mask.cpu()
                                  for mask in logit_data["mask_positions"]],
                "timesteps": logit_data["timesteps"],
            }
            collected_data.append(example_data)
    
    # Save collected data
    output_file = os.path.join(args.output_path, "teacher_logits.pkl")
    print(f"Saving {len(collected_data)} examples to {output_file}...")
    with open(output_file, "wb") as f:
        pickle.dump(collected_data, f)
    
    print(f"Done! Collected logits for {len(collected_data)} examples.")
    print(f"Output saved to {args.output_path}")


if __name__ == "__main__":
    main()

