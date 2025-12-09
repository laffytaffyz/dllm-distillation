#!/usr/bin/env python3
"""
Load Qwen 0.5 from HuggingFace, do autoregressive generation, and calculate perplexity.

This script:
1. Loads Qwen 0.5 autoregressive model from HuggingFace
2. Generates text using standard autoregressive generation (model.generate())
3. Calculates perplexity for the generated text
4. Supports datasets: human eval, human eval plus, mbpp, mbpp plus
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import torch
import torch.nn.functional as F
from tqdm import tqdm
import transformers
from accelerate import Accelerator

def load_prompts_from_dataset(dataset_name: str) -> List[Dict]:
    """Load prompts from a dataset."""
    prompts = []
    
    if dataset_name in ["humaneval", "humaneval_plus"]:
        try:
            from datasets import load_dataset
            dataset = load_dataset("openai/humaneval", split="test")
            for item in dataset:
                prompts.append({
                    'task_id': item.get('task_id', 'unknown'),
                    'prompt': item.get('prompt', ''),
                    'canonical_solution': item.get('canonical_solution', ''),
                })
        except ImportError:
            print("Warning: datasets library not found. Install with: pip install datasets")
            return []
    elif dataset_name in ["mbpp", "mbpp_plus"]:
        try:
            from datasets import load_dataset
            dataset = load_dataset("mbpp", split="test")
            for item in dataset:
                prompts.append({
                    'task_id': item.get('task_id', 'unknown'),
                    'prompt': item.get('prompt', ''),
                    'code': item.get('code', ''),
                })
        except ImportError:
            print("Warning: datasets library not found. Install with: pip install datasets")
            return []
    else:
        print(f"Warning: Unknown dataset {dataset_name}")
        return []
    
    return prompts

def generate_text(
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    prompts: List[str],
    device: torch.device,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.95,
    do_sample: bool = True,
    batch_size: int = 8,
) -> List[str]:
    """Generate text using standard autoregressive generation."""
    generated_texts = []
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating text"):
            batch_prompts = prompts[i:i + batch_size]
            
            # Tokenize prompts
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            # Generate
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id,
            )
            
            # Decode generated text (skip the prompt)
            prompt_lengths = input_ids.shape[1]
            for j, gen_ids in enumerate(generated_ids):
                # Extract only the newly generated tokens
                new_tokens = gen_ids[prompt_lengths:]
                generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                generated_texts.append(generated_text)
    
    return generated_texts

def calculate_perplexity(
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    texts: List[str],
    device: torch.device,
    batch_size: int = 8,
    max_length: int = 2048
) -> List[float]:
    """Calculate perplexity for a list of texts."""
    perplexities = []
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Calculating perplexity"):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize texts
            encodings = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            
            # Calculate logits
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_attention = attention_mask[..., 1:].contiguous()
            
            # Calculate log probabilities
            log_probs = F.log_softmax(shift_logits, dim=-1)
            
            # Get log probability of actual tokens
            batch_size_actual = shift_logits.size(0)
            vocab_size = shift_logits.size(-1)
            
            # Reshape for gathering
            log_probs_flat = log_probs.view(-1, vocab_size)
            labels_flat = shift_labels.view(-1)
            attention_flat = shift_attention.view(-1)
            
            # Get log probability of each token
            token_log_probs = log_probs_flat.gather(1, labels_flat.unsqueeze(1)).squeeze(1)
            
            # Mask out padding tokens
            masked_log_probs = token_log_probs * attention_flat.float()
            
            # Reshape back to batch
            masked_log_probs = masked_log_probs.view(batch_size_actual, -1)
            attention_flat = attention_flat.view(batch_size_actual, -1)
            
            # Calculate perplexity for each text in batch
            for j in range(batch_size_actual):
                valid_tokens = attention_flat[j].sum().item()
                if valid_tokens > 0:
                    nll = -masked_log_probs[j].sum().item() / valid_tokens
                    ppl = torch.exp(torch.tensor(nll)).item()
                    perplexities.append(ppl)
                else:
                    perplexities.append(float('inf'))
    
    return perplexities

def main():
    parser = argparse.ArgumentParser(
        description="Load Qwen 0.5 from HuggingFace, generate text autoregressively, and calculate perplexity"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="Path to Qwen 0.5 model on HuggingFace (default: qwen0.5b)"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["humaneval", "humaneval_plus", "mbpp", "mbpp_plus"],
        help="Datasets to evaluate (default: humaneval humaneval_plus mbpp mbpp_plus)"
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="JSON file with prompts (overrides datasets). Format: [{'task_id': str, 'prompt': str}, ...]"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate (default: 128)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Temperature for generation (default: 0.8)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling (default: 0.95)"
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        default=True,
        help="Use sampling for generation (default: True)"
    )
    parser.add_argument(
        "--no_sample",
        action="store_true",
        help="Disable sampling (use greedy decoding)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for generation and perplexity calculation (default: 8)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length (default: 2048)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file to save results (default: print to stdout)"
    )
    parser.add_argument(
        "--save_generations",
        type=str,
        default=None,
        help="File to save generated texts (optional)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)"
    )
    
    args = parser.parse_args()
    
    # Handle do_sample flag
    if args.no_sample:
        args.do_sample = False
    
    # Initialize accelerator
    accelerator = Accelerator()
    device = accelerator.device if torch.cuda.is_available() else torch.device(args.device)
    
    if accelerator.is_main_process:
        print(f"Loading model from HuggingFace: {args.model_path}")
        print(f"Using device: {device}")
        print(f"Generation parameters: max_new_tokens={args.max_new_tokens}, temperature={args.temperature}, top_p={args.top_p}, do_sample={args.do_sample}")
    
    # Load model and tokenizer from HuggingFace
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
        device_map="auto" if device.type == "cuda" else None
    )
    
    if device.type == "cpu":
        model = model.to(device)
    
    model.eval()
    
    # Load prompts
    all_prompts_data = []
    
    if args.prompts_file:
        # Load from file
        with open(args.prompts_file, 'r') as f:
            all_prompts_data = json.load(f)
        if accelerator.is_main_process:
            print(f"Loaded {len(all_prompts_data)} prompts from {args.prompts_file}")
    else:
        # Load from datasets
        for dataset in args.datasets:
            prompts_data = load_prompts_from_dataset(dataset)
            all_prompts_data.extend(prompts_data)
            if accelerator.is_main_process:
                print(f"Loaded {len(prompts_data)} prompts from {dataset}")
    
    if not all_prompts_data:
        print("Error: No prompts loaded!")
        return
    
    # Extract prompt strings
    prompts = [item['prompt'] for item in all_prompts_data]
    
    if accelerator.is_main_process:
        print(f"\nTotal prompts: {len(prompts)}")
        print("Generating text...")
    
    # Generate text using standard autoregressive generation
    generated_texts = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        batch_size=args.batch_size,
    )
    
    if accelerator.is_main_process:
        print(f"Generated {len(generated_texts)} texts")
        print("Calculating perplexity...")
    
    # Calculate perplexity
    perplexities = calculate_perplexity(
        model=model,
        tokenizer=tokenizer,
        texts=generated_texts,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    
    # Prepare results
    results = {
        'model_path': args.model_path,
        'generation_params': {
            'max_new_tokens': args.max_new_tokens,
            'temperature': args.temperature,
            'top_p': args.top_p,
            'do_sample': args.do_sample,
        },
        'num_samples': len(generated_texts),
        'mean_perplexity': sum(perplexities) / len(perplexities) if perplexities else float('inf'),
        'median_perplexity': sorted(perplexities)[len(perplexities) // 2] if perplexities else float('inf'),
        'min_perplexity': min(perplexities) if perplexities else float('inf'),
        'max_perplexity': max(perplexities) if perplexities else float('inf'),
        'samples': [
            {
                'task_id': item.get('task_id', f'sample_{i}'),
                'prompt': item.get('prompt', ''),
                'generated_text': gen_text,
                'perplexity': ppl
            }
            for i, (item, gen_text, ppl) in enumerate(zip(all_prompts_data, generated_texts, perplexities))
        ]
    }
    
    # Save or print results
    if accelerator.is_main_process:
        if args.save_generations:
            with open(args.save_generations, 'w') as f:
                for sample in results['samples']:
                    f.write(json.dumps(sample) + '\n')
            print(f"\nGenerated texts saved to: {args.save_generations}")
        
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {args.output_file}")
        else:
            print("\n" + "=" * 80)
            print("PERPLEXITY RESULTS")
            print("=" * 80)
            print(f"Model: {args.model_path}")
            print(f"Generation params: max_new_tokens={args.max_new_tokens}, temperature={args.temperature}, top_p={args.top_p}, do_sample={args.do_sample}")
            print(f"\nNumber of samples: {results['num_samples']}")
            print(f"Mean perplexity: {results['mean_perplexity']:.4f}")
            print(f"Median perplexity: {results['median_perplexity']:.4f}")
            print(f"Min perplexity: {results['min_perplexity']:.4f}")
            print(f"Max perplexity: {results['max_perplexity']:.4f}")

if __name__ == "__main__":
    main()
