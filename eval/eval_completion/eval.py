import logging
import os
import json
import glob
import re
import time
from typing import List, Optional, Union

import torch
import transformers
from accelerate import Accelerator
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import get_dtype
from lm_eval.__main__ import cli_evaluate
from lm_eval import tasks
from tqdm import tqdm

# Import your custom model and generation config
from veomni.models.transformers.qwen2.modeling_qwen2 import (
    Qwen2ForCausalLM
)
from veomni.models.transformers.qwen2.generation_utils import (
    MDMGenerationConfig
)

# Import eval utilities for perplexity calculation
from eval_utils import (
    calculate_log_likelihood_from_logits,
    get_rolling_token_windows
)

eval_logger = logging.getLogger("eval_logger")

# Global counter for printing sample prompts/responses
_sample_print_count = 0
_MAX_SAMPLES_TO_PRINT = 3


@register_model("custom_coder")
class CustomCoder(LM):
    def __init__(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        batch_size: Optional[Union[int, str]] = 1,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        # Generation parameters for diffusion_generate
        max_new_tokens: Optional[int] = 128,
        steps: Optional[int] = 100,
        temperature: Optional[float] = 0.5,
        top_k: Optional[int] = 200,
        alg: Optional[str] = 'p2',
        alg_temp: Optional[float] = 0.5,
        trust_remote_code: Optional[bool] = True,
        # Checkpoint loading (if pretrained is a .pt file)
        base_model: Optional[str] = None,  # Base model path for loading checkpoints
        # Other lm-harness params
        max_length: Optional[int] = 2048,
        **kwargs,
    ) -> None:
        super().__init__()

        # Initialize accelerator for multi-GPU support
        self.accelerator = Accelerator()
        self._device = self.accelerator.device
        self.batch_size_per_gpu = int(batch_size)
        self.max_length = max_length
        self.trust_remote_code = trust_remote_code
        self.base_model = base_model

        # Initialize parallel state for custom model
        self._init_parallel_state()

        # Store generation parameters
        self.max_new_tokens = max_new_tokens
        self.steps = steps
        self.temperature = temperature
        self.top_k = top_k
        self.alg = alg
        self.alg_temp = alg_temp

        # Load the custom model and tokenizer
        self._create_model_and_tokenizer(pretrained, dtype)

    def _init_parallel_state(self):
        """Initialize parallel state for the custom model."""
        from veomni.distributed.parallel_state import init_parallel_state
        import torch.distributed as dist

        # If only 1 process, skip distributed init
        if self.accelerator.num_processes == 1:
            eval_logger.info("Single GPU detected. Skipping distributed init.")
            return

        # Multi-GPU: ensure process group is initialized
        if not dist.is_initialized():
            # import pdb; pdb.set_trace()  # Commented out debug statement
            dist.init_process_group(
                backend="nccl",   # use "gloo" if CPU-only
                init_method="env://"
            )

        world_size = self.accelerator.num_processes
        init_parallel_state(
            dp_size=world_size,
            tp_size=1,
            ep_size=1,
            pp_size=1,
            cp_size=1,
            ulysses_size=1,
            dp_mode="ddp",
            device_type="cuda",
            include_sp_in_fsdp=True,
        )

    def _create_model_and_tokenizer(self, pretrained, dtype):
        """Loads the Qwen2ForCausalLM model and its tokenizer.
        
        Supports both HuggingFace model directories and PyTorch checkpoint files (.pt).
        If pretrained is a .pt file, loads the base model first, then applies the checkpoint.
        """
        # Check if pretrained is a checkpoint file (.pt)
        is_checkpoint = isinstance(pretrained, str) and pretrained.endswith('.pt') and os.path.isfile(pretrained)
        
        if is_checkpoint:
            # Load from checkpoint file
            eval_logger.info(f"Loading checkpoint: {pretrained}")
            
            # Determine base model path
            if self.base_model:
                base_model_path = self.base_model
            else:
                # Default base model path (same as used in training)
                base_model_path = '/orcd/data/tpoggio/001/tiffany8/dllm-distillation/models/open-dcoder-0.5B'
                if not os.path.exists(base_model_path):
                    # Fallback to HuggingFace model ID
                    base_model_path = "fredzzp/open-dcoder-0.5B"
            
            eval_logger.info(f"Loading tokenizer from: {base_model_path}")
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                base_model_path,
                trust_remote_code=self.trust_remote_code,
            )
            
            eval_logger.info(f"Loading base model from: {base_model_path}")
            self.model = Qwen2ForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=get_dtype(dtype),
                trust_remote_code=self.trust_remote_code,
            )
            
            # Load checkpoint state dict
            checkpoint = torch.load(pretrained, map_location="cpu")
            
            # Handle both 'model_state_dict' (from training checkpoints) and direct state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Load state dict directly (model is not wrapped yet at this point)
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                eval_logger.warning(f"Missing keys when loading checkpoint: {len(missing_keys)} keys")
            if unexpected_keys:
                eval_logger.warning(f"Unexpected keys when loading checkpoint: {len(unexpected_keys)} keys")
            
            eval_logger.info("Checkpoint loaded successfully")
        else:
            # Load from HuggingFace model directory or model ID
            eval_logger.info(f"Loading tokenizer from: {pretrained}")
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                pretrained,
                trust_remote_code=self.trust_remote_code,
            )

            eval_logger.info(f"Loading model from: {pretrained}")
            self.model = Qwen2ForCausalLM.from_pretrained(
                pretrained,
                torch_dtype=get_dtype(dtype),
                trust_remote_code=self.trust_remote_code,
            )

        # Set the mask token if not already set. This is crucial for
        # generation.
        if self.tokenizer.mask_token is None:
            self.tokenizer.add_special_tokens({'mask_token': '[MASK]'})
            self.model.resize_token_embeddings(len(self.tokenizer))
            eval_logger.info("Added new [MASK] token.")

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            eval_logger.info("Set pad_token to eos_token")

        self.tokenizer.padding_side = "left"

        # Prepare model for distributed training/inference
        self.model = self.accelerator.prepare(self.model)
        self.model.eval()

    def _generate_batch(
        self, prompts: List[str], gen_kwargs: dict = None
    ) -> List[str]:
        """Generates text for a batch of prompts using the same method as training."""
        global _sample_print_count
        import torch.nn.functional as F

        # Tokenize the batch of prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length - self.max_new_tokens,
        ).to(self.device)

        prompt_ids = inputs.input_ids
        batch_size, prompt_len = prompt_ids.shape
        mask_token_id = self.tokenizer.mask_token_id
        device = self.device
        
        # Extract specific parameters from yaml if provided
        if gen_kwargs is None:
            gen_kwargs = {}
        num_return_sequences = gen_kwargs.get('num_return_sequences', 1)
        
        # Use the same generation method as training
        # Pad with mask tokens for generation (same as training)
        max_length = prompt_ids.shape[1] + self.max_new_tokens
        pad_len = max_length - prompt_len
        masked_suffix = torch.full(
            (batch_size, pad_len),
            mask_token_id,
            device=device,
            dtype=torch.long
        )
        x_t = torch.cat([prompt_ids, masked_suffix], dim=1)
        
        # Fixed tokens (prompt) should never be remasked
        fix_mask = (x_t != mask_token_id)
        
        # Attention mask
        attention_mask = (x_t != self.tokenizer.pad_token_id).long() if self.tokenizer.pad_token_id is not None else None
        
        # Timesteps from 1.0 down to eps (same as training)
        eps = 1e-3
        timesteps = torch.linspace(1.0, eps, self.steps + 1, device=device)
        
        # Access the underlying model when wrapped in DDP
        model = (
            self.model.module
            if hasattr(self.model, 'module')
            else self.model
        )
        
        # Debug: Print input info
        if _sample_print_count < _MAX_SAMPLES_TO_PRINT and self.accelerator.is_main_process:
            eval_logger.info(f"DEBUG: Using training-style generation")
            eval_logger.info(f"DEBUG: Input shape: {prompt_ids.shape}")
            eval_logger.info(f"DEBUG: After padding with masks: {x_t.shape}")
            eval_logger.info(f"DEBUG: Steps: {self.steps}, temperature: {self.temperature}, alg: {self.alg}")
        
        with torch.no_grad():
            for step in range(self.steps):
                mask_index = (x_t == mask_token_id)
                if not mask_index.any():
                    break
                
                t = timesteps[step]
                s = timesteps[step + 1] if step < self.steps - 1 else eps
                
                # Forward pass with is_causal=False (same as training)
                outputs = model(input_ids=x_t, attention_mask=attention_mask, is_causal=False)
                logits = outputs.logits
                
                # Shift logits to align with training (same as training)
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
                
                # Update x_t using p2 algorithm (same as training)
                if self.alg == "p2":
                    kappa_t = (step + 1) / self.steps
                    
                    # Compute confidence and sampled tokens
                    probs = F.softmax(logits / self.temperature, dim=-1)
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
                    raise NotImplementedError(f"Algorithm {self.alg} not implemented")
        
        # Debug: Print output info
        if _sample_print_count < _MAX_SAMPLES_TO_PRINT and self.accelerator.is_main_process:
            eval_logger.info(f"DEBUG: Final sequence shape: {x_t.shape}")
            eval_logger.info(f"DEBUG: Generated part (first 50 tokens): {x_t[0, prompt_len:prompt_len+50].tolist()}")
        
        # Decode the generated sequences, skipping the prompt
        generated_texts = self.tokenizer.batch_decode(
            x_t[:, prompt_len:],
            skip_special_tokens=True
        )
        
        # Print sample prompts and responses for debugging (only first few)
        if _sample_print_count < _MAX_SAMPLES_TO_PRINT and self.accelerator.is_main_process:
            for i in range(min(len(prompts), _MAX_SAMPLES_TO_PRINT - _sample_print_count)):
                idx = _sample_print_count + i
                if idx < len(prompts):
                    eval_logger.info("=" * 80)
                    eval_logger.info(f"SAMPLE PROMPT/RESPONSE #{idx + 1}")
                    eval_logger.info("=" * 80)
                    eval_logger.info(f"PROMPT ({len(prompts[idx])} chars):")
                    eval_logger.info(f"{prompts[idx][:500]}{'...' if len(prompts[idx]) > 500 else ''}")
                    eval_logger.info("-" * 80)
                    eval_logger.info(f"RESPONSE ({len(generated_texts[i])} chars):")
                    eval_logger.info(f"{generated_texts[i][:500]}{'...' if len(generated_texts[i]) > 500 else ''}")
                    eval_logger.info("=" * 80)
            _sample_print_count += min(len(prompts), _MAX_SAMPLES_TO_PRINT - _sample_print_count)

        return generated_texts

    @torch.no_grad()
    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        """
        The main generation method called by lm-harness.
        It processes requests in batches and uses our _generate_batch method.
        """
        res = []
        
        # Track inference time for first few samples
        inference_times = []
        num_inference_samples = min(10, len(requests))
        measure_inference = self.accelerator.is_main_process and num_inference_samples > 0

        # Only show progress bar on main process
        disable_tqdm = disable_tqdm or not self.accelerator.is_main_process
        pbar = tqdm(
            total=len(requests),
            disable=disable_tqdm,
            desc="Running generate_until requests",
        )

        for i in range(0, len(requests), self.batch_size):
            batch_requests = requests[i:i + self.batch_size]
            contexts = [req.args[0] for req in batch_requests]

            # Extract yaml generation kwargs from first request
            first_req_args = batch_requests[0].args
            gen_kwargs = first_req_args[1] if len(first_req_args) > 1 else {}

            # Measure inference time for first few samples
            if measure_inference and i < num_inference_samples:
                # Warmup
                if i == 0:
                    _ = self._generate_batch([contexts[0]], gen_kwargs)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                
                # Measure time
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.perf_counter()
                
                batch_responses = self._generate_batch(contexts, gen_kwargs)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                
                # Average time per sample in batch
                batch_time = (end_time - start_time) / len(contexts)
                inference_times.extend([batch_time] * len(contexts))
            else:
                # Generate responses for the batch
                batch_responses = self._generate_batch(contexts, gen_kwargs)

            # Process 'until' stopping criteria
            for resp, req in zip(batch_responses, batch_requests):
                stop_sequences = req.args[1].get('until', [])
                if stop_sequences:
                    for stop_seq in stop_sequences:
                        if stop_seq in resp:
                            resp = resp.split(stop_seq)[0]
                res.append(resp)

            pbar.update(len(batch_requests))

        pbar.close()
        
        # Print inference time summary
        if measure_inference and inference_times:
            avg_time = sum(inference_times) / len(inference_times)
            eval_logger.info(f"Average inference time: {avg_time:.4f} seconds per sample (measured on {len(inference_times)} samples)")
            print(f"\nAverage inference time: {avg_time:.4f} seconds per sample (measured on {len(inference_times)} samples)")
        
        # Calculate perplexity on prompts (for generation tasks)
        if self.accelerator.is_main_process and len(requests) > 0:
            # Compute perplexity on prompts using autoregressive loglikelihood
            prompt_perplexities = []
            total_prompt_tokens = 0
            
            # Access the underlying model when wrapped in DDP
            model = (
                self.model.module
                if hasattr(self.model, 'module')
                else self.model
            )
            
            # Sample a subset of prompts for perplexity (to avoid being too slow)
            num_ppl_samples = min(50, len(requests))
            sample_indices = torch.linspace(0, len(requests) - 1, num_ppl_samples, device='cpu').long().tolist()
            
            with torch.no_grad():
                for idx in sample_indices:
                    context = requests[idx].args[0]
                    
                    # Tokenize prompt
                    token_list = self.tokenizer.encode(context, add_special_tokens=False)
                    if len(token_list) < 2:  # Need at least 2 tokens for perplexity
                        continue
                    
                    # Use rolling windows to compute log-likelihood
                    prefix_token = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else self.tokenizer.bos_token_id
                    windows = get_rolling_token_windows(
                        token_list, prefix_token, self.max_length, context_len=1
                    )
                    
                    total_log_likelihood = 0.0
                    total_tokens = 0
                    
                    for context_enc, continuation_enc in windows:
                        if len(continuation_enc) == 0:
                            continue
                        
                        # Combine context and continuation
                        input_ids = context_enc + continuation_enc
                        
                        # Create input tensor (remove last token for prediction)
                        input_tensor = torch.tensor([input_ids[:-1]], device=self.device)
                        target_tensor = torch.tensor([continuation_enc], device=self.device)
                        
                        # Forward pass (standard HuggingFace models are autoregressive by default)
                        outputs = model(input_ids=input_tensor)
                        logits = outputs.logits
                        
                        # For autoregressive models, logits[i] predicts token[i+1]
                        # Get logits for continuation tokens
                        context_len = len(context_enc)
                        # logits shape: [batch, seq_len, vocab_size]
                        # We want logits for positions that predict continuation tokens
                        cont_logits = logits[:, context_len-1:context_len-1+len(continuation_enc), :]
                        
                        # Calculate log-likelihood
                        log_likelihood, num_tokens = calculate_log_likelihood_from_logits(
                            cont_logits, target_tensor
                        )
                        
                        total_log_likelihood += log_likelihood
                        total_tokens += num_tokens
                    
                    if total_tokens > 0:
                        # Compute perplexity: PPL = exp(-log_likelihood / num_tokens)
                        nll = -total_log_likelihood / total_tokens
                        ppl = torch.exp(torch.tensor(nll)).item()
                        prompt_perplexities.append(ppl)
                        total_prompt_tokens += total_tokens
            
            if len(prompt_perplexities) > 0:
                avg_ppl = sum(prompt_perplexities) / len(prompt_perplexities)
                eval_logger.info(f"Average prompt perplexity: {avg_ppl:.4f} (computed on {len(prompt_perplexities)}/{len(requests)} prompts, {total_prompt_tokens} tokens)")
                print(f"Average prompt perplexity: {avg_ppl:.4f} (computed on {len(prompt_perplexities)}/{len(requests)} prompts, {total_prompt_tokens} tokens)")
        
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[tuple[float, bool]]:
        """
        Compute log-likelihood of generating a continuation from a context.
        
        For diffusion models, we use autoregressive mode to compute log-likelihood.
        """
        res = []
        
        # Access the underlying model when wrapped in DDP
        model = (
            self.model.module
            if hasattr(self.model, 'module')
            else self.model
        )
        
        for request in tqdm(requests, disable=not self.accelerator.is_main_process, desc="Computing loglikelihood"):
            context, continuation = request.args
            
            # Tokenize context and continuation
            if context == "":
                # Use prefix token (EOS/BOS) as context
                context_enc = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else []
                continuation_enc = self.tokenizer.encode(continuation, add_special_tokens=False)
            else:
                # Encode full sequence and split
                full_enc = self.tokenizer.encode(context + continuation, add_special_tokens=False)
                context_enc = self.tokenizer.encode(context, add_special_tokens=False)
                continuation_enc = full_enc[len(context_enc):]
            
            if len(continuation_enc) == 0:
                res.append((float('-inf'), False))
                continue
            
            # Combine context and continuation
            input_ids = context_enc + continuation_enc
            
            # Truncate if too long
            if len(input_ids) > self.max_length + 1:
                input_ids = input_ids[-(self.max_length + 1):]
                context_enc = input_ids[:-len(continuation_enc)]
                continuation_enc = input_ids[-len(continuation_enc):]
            
            # Create input tensor (remove last token for prediction)
            input_tensor = torch.tensor([input_ids[:-1]], device=self.device)
            target_tensor = torch.tensor([continuation_enc], device=self.device)
            
            # Forward pass with is_causal=True for autoregressive mode
            with torch.no_grad():
                outputs = model(input_ids=input_tensor, is_causal=True)
                logits = outputs.logits
                
                # Shift logits to align with targets (same as training)
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
                
                # Get logits for continuation tokens only
                # We need to get the logits at positions corresponding to continuation
                context_len = len(context_enc)
                cont_logits = logits[:, context_len-1:context_len-1+len(continuation_enc), :]
                
                # Calculate log-likelihood
                log_likelihood, num_tokens = calculate_log_likelihood_from_logits(
                    cont_logits, target_tensor
                )
                
                # Check if greedy (argmax matches continuation)
                greedy_tokens = cont_logits.argmax(dim=-1)
                is_greedy = (greedy_tokens.squeeze(0) == target_tensor.squeeze(0)).all().item()
                
                res.append((log_likelihood, is_greedy))
        
        return res

    def loglikelihood_rolling(
        self, requests: List[Instance]
    ) -> List[float]:
        """
        Stub implementation - not used for HumanEval/MBPP tasks.
        Required by LM base class but not called for generation tasks.
        """
        raise NotImplementedError(
            "loglikelihood_rolling is not implemented. "
            "This model is only used for generation tasks (HumanEval, MBPP) which use generate_until."
        )

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self.accelerator.process_index

    @property
    def world_size(self):
        return self.accelerator.num_processes

    def _measure_inference_time(self, prompts: List[str], num_warmup: int = 2) -> float:
        """
        Measure average inference time (excluding tokenization/decoding) on given prompts.
        
        Args:
            prompts: List of prompt strings to measure on
            num_warmup: Number of warmup runs before timing
        
        Returns:
            Average inference time in seconds
        """
        if not self.accelerator.is_main_process:
            return 0.0
        
        # Warmup runs
        for i in range(num_warmup):
            prompt = prompts[min(i, len(prompts) - 1)]
            inputs = self.tokenizer(
                [prompt],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length - self.max_new_tokens,
            ).to(self.device)
            
            generation_config = MDMGenerationConfig(
                mask_token_id=self.tokenizer.mask_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.max_new_tokens,
                steps=self.steps,
                temperature=self.temperature,
                top_k=self.top_k,
                alg=self.alg,
                alg_temp=self.alg_temp,
                num_return_sequences=1,
                return_dict_in_generate=True,
                output_history=False
            )
            
            model = (
                self.model.module
                if hasattr(self.model, 'module')
                else self.model
            )
            
            with torch.no_grad():
                _ = model.diffusion_generate(
                    inputs=inputs.input_ids,
                    generation_config=generation_config,
                )
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # Actual timing runs
        inference_times = []
        for prompt in prompts:
            # Tokenize (not timed)
            inputs = self.tokenizer(
                [prompt],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length - self.max_new_tokens,
            ).to(self.device)
            
            generation_config = MDMGenerationConfig(
                mask_token_id=self.tokenizer.mask_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.max_new_tokens,
                steps=self.steps,
                temperature=self.temperature,
                top_k=self.top_k,
                alg=self.alg,
                alg_temp=self.alg_temp,
                num_return_sequences=1,
                return_dict_in_generate=True,
                output_history=False
            )
            
            model = (
                self.model.module
                if hasattr(self.model, 'module')
                else self.model
            )
            
            # Measure only inference time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            with torch.no_grad():
                _ = model.diffusion_generate(
                    inputs=inputs.input_ids,
                    generation_config=generation_config,
                )
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            inference_times.append(end_time - start_time)
        
        return sum(inference_times) / len(inference_times) if inference_times else 0.0


@register_model("autoregressive_qwen")
class AutoregressiveQwen(LM):
    """
    Standard autoregressive Qwen model for comparison with diffusion models.
    Uses standard transformers.AutoModelForCausalLM and model.generate().
    """
    def __init__(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        batch_size: Optional[Union[int, str]] = 1,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        # Generation parameters for standard generate
        max_new_tokens: Optional[int] = 128,
        temperature: Optional[float] = 0.8,
        top_p: Optional[float] = 0.95,
        do_sample: Optional[bool] = True,
        trust_remote_code: Optional[bool] = True,
        # Checkpoint loading (if pretrained is a .pt file)
        base_model: Optional[str] = None,  # Base model path for loading checkpoints
        # Other lm-harness params
        max_length: Optional[int] = 2048,
        **kwargs,
    ) -> None:
        super().__init__()

        # Initialize accelerator for multi-GPU support
        self.accelerator = Accelerator()
        self._device = self.accelerator.device
        self.batch_size_per_gpu = int(batch_size)
        self.max_length = max_length
        self.trust_remote_code = trust_remote_code

        # Store generation parameters
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.base_model = base_model

        # Load the model and tokenizer
        self._create_model_and_tokenizer(pretrained, dtype)
        
        # Store globally for inference time measurement
        global _global_model_instance
        if self.accelerator.is_main_process:
            _global_model_instance = self

    def _create_model_and_tokenizer(self, pretrained, dtype):
        """Loads the standard AutoModelForCausalLM model and its tokenizer.
        
        Supports both HuggingFace model directories and PyTorch checkpoint files (.pt).
        If pretrained is a .pt file, loads the base model first, then applies the checkpoint.
        """
        # Check if pretrained is a checkpoint file (.pt)
        is_checkpoint = isinstance(pretrained, str) and pretrained.endswith('.pt') and os.path.isfile(pretrained)
        
        if is_checkpoint:
            # Load from checkpoint file
            eval_logger.info(f"Loading checkpoint: {pretrained}")
            
            # Determine base model path
            if self.base_model:
                base_model_path = self.base_model
            else:
                # Default base model path (same as used in training)
                base_model_path = '/orcd/data/tpoggio/001/tiffany8/dllm-distillation/models/open-dcoder-0.5B'
                if not os.path.exists(base_model_path):
                    # Fallback to HuggingFace model ID
                    base_model_path = "fredzzp/open-dcoder-0.5B"
            
            eval_logger.info(f"Loading tokenizer from: {base_model_path}")
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                base_model_path,
                trust_remote_code=self.trust_remote_code,
            )
            
            eval_logger.info(f"Loading base model from: {base_model_path}")
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=get_dtype(dtype),
                trust_remote_code=self.trust_remote_code,
            )
            
            checkpoint = torch.load(pretrained, map_location="cpu")
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Load checkpoint weights
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                eval_logger.warning(f"Missing keys when loading checkpoint: {len(missing_keys)} keys")
            if unexpected_keys:
                eval_logger.warning(f"Unexpected keys when loading checkpoint: {len(unexpected_keys)} keys")
            
            eval_logger.info("Checkpoint loaded successfully")
        else:
            # Load from model directory or HuggingFace model ID
            eval_logger.info(f"Loading tokenizer from: {pretrained}")
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                pretrained,
                trust_remote_code=self.trust_remote_code,
            )

            eval_logger.info(f"Loading autoregressive model from: {pretrained}")
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                pretrained,
                torch_dtype=get_dtype(dtype),
                trust_remote_code=self.trust_remote_code,
            )

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            eval_logger.info("Set pad_token to eos_token")

        self.tokenizer.padding_side = "left"

        # Prepare model for distributed training/inference
        self.model = self.accelerator.prepare(self.model)
        self.model.eval()

    def _generate_batch(
        self, prompts: List[str], gen_kwargs: dict = None
    ) -> List[str]:
        """Generates text for a batch of prompts using standard generate()."""

        # Tokenize the batch of prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length - self.max_new_tokens,
        ).to(self.device)

        prompt_lengths = inputs.input_ids.shape[1]

        # Extract specific parameters from yaml if provided
        if gen_kwargs is None:
            gen_kwargs = {}

        # Prepare generation kwargs
        generate_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # Override with yaml parameters if provided
        generate_kwargs.update(gen_kwargs)
        
        # Remove 'until' from kwargs - it's not a valid model.generate() parameter
        # 'until' is handled separately in generate_until() as a stopping criterion
        generate_kwargs.pop('until', None)

        with torch.no_grad():
            # Access the underlying model when wrapped in DDP
            model = (
                self.model.module
                if hasattr(self.model, 'module')
                else self.model
            )
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **generate_kwargs
            )

        # Decode the generated sequences, skipping the prompt
        generated_texts = self.tokenizer.batch_decode(
            outputs[:, prompt_lengths:],
            skip_special_tokens=True
        )

        return generated_texts

    @torch.no_grad()
    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        """
        The main generation method called by lm-harness.
        It processes requests in batches and uses our _generate_batch method.
        """
        res = []
        
        # Track inference time for first few samples
        inference_times = []
        num_inference_samples = min(10, len(requests))
        measure_inference = self.accelerator.is_main_process and num_inference_samples > 0

        # Only show progress bar on main process
        disable_tqdm = disable_tqdm or not self.accelerator.is_main_process
        pbar = tqdm(
            total=len(requests),
            disable=disable_tqdm,
            desc="Running generate_until requests",
        )

        for i in range(0, len(requests), self.batch_size):
            batch_requests = requests[i:i + self.batch_size]
            contexts = [req.args[0] for req in batch_requests]

            # Extract yaml generation kwargs from first request
            first_req_args = batch_requests[0].args
            gen_kwargs = first_req_args[1] if len(first_req_args) > 1 else {}

            # Measure inference time for first few samples
            if measure_inference and i < num_inference_samples:
                # Warmup
                if i == 0:
                    _ = self._generate_batch([contexts[0]], gen_kwargs)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                
                # Measure time
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.perf_counter()
                
                batch_responses = self._generate_batch(contexts, gen_kwargs)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                
                # Average time per sample in batch
                batch_time = (end_time - start_time) / len(contexts)
                inference_times.extend([batch_time] * len(contexts))
            else:
                # Generate responses for the batch
                batch_responses = self._generate_batch(contexts, gen_kwargs)

            # Process 'until' stopping criteria
            for resp, req in zip(batch_responses, batch_requests):
                stop_sequences = req.args[1].get('until', [])
                if stop_sequences:
                    for stop_seq in stop_sequences:
                        if stop_seq in resp:
                            resp = resp.split(stop_seq)[0]
                res.append(resp)

            pbar.update(len(batch_requests))

        pbar.close()
        
        # Print inference time summary
        if measure_inference and inference_times:
            avg_time = sum(inference_times) / len(inference_times)
            eval_logger.info(f"Average inference time: {avg_time:.4f} seconds per sample (measured on {len(inference_times)} samples)")
            print(f"\nAverage inference time: {avg_time:.4f} seconds per sample (measured on {len(inference_times)} samples)")
        
        # Calculate perplexity on prompts (for generation tasks)
        if self.accelerator.is_main_process and len(requests) > 0:
            # Compute perplexity on prompts using autoregressive loglikelihood
            prompt_perplexities = []
            total_prompt_tokens = 0
            
            # Access the underlying model when wrapped in DDP
            model = (
                self.model.module
                if hasattr(self.model, 'module')
                else self.model
            )
            
            # Sample a subset of prompts for perplexity (to avoid being too slow)
            num_ppl_samples = min(50, len(requests))
            sample_indices = torch.linspace(0, len(requests) - 1, num_ppl_samples, device='cpu').long().tolist()
            
            with torch.no_grad():
                for idx in sample_indices:
                    context = requests[idx].args[0]
                    
                    # Tokenize prompt
                    token_list = self.tokenizer.encode(context, add_special_tokens=False)
                    if len(token_list) < 2:  # Need at least 2 tokens for perplexity
                        continue
                    
                    # Use rolling windows to compute log-likelihood
                    prefix_token = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else self.tokenizer.bos_token_id
                    windows = get_rolling_token_windows(
                        token_list, prefix_token, self.max_length, context_len=1
                    )
                    
                    total_log_likelihood = 0.0
                    total_tokens = 0
                    
                    for context_enc, continuation_enc in windows:
                        if len(continuation_enc) == 0:
                            continue
                        
                        # Combine context and continuation
                        input_ids = context_enc + continuation_enc
                        
                        # Create input tensor (remove last token for prediction)
                        input_tensor = torch.tensor([input_ids[:-1]], device=self.device)
                        target_tensor = torch.tensor([continuation_enc], device=self.device)
                        
                        # Forward pass (standard HuggingFace models are autoregressive by default)
                        outputs = model(input_ids=input_tensor)
                        logits = outputs.logits
                        
                        # For autoregressive models, logits[i] predicts token[i+1]
                        # Get logits for continuation tokens
                        context_len = len(context_enc)
                        # logits shape: [batch, seq_len, vocab_size]
                        # We want logits for positions that predict continuation tokens
                        cont_logits = logits[:, context_len-1:context_len-1+len(continuation_enc), :]
                        
                        # Calculate log-likelihood
                        log_likelihood, num_tokens = calculate_log_likelihood_from_logits(
                            cont_logits, target_tensor
                        )
                        
                        total_log_likelihood += log_likelihood
                        total_tokens += num_tokens
                    
                    if total_tokens > 0:
                        # Compute perplexity: PPL = exp(-log_likelihood / num_tokens)
                        nll = -total_log_likelihood / total_tokens
                        ppl = torch.exp(torch.tensor(nll)).item()
                        prompt_perplexities.append(ppl)
                        total_prompt_tokens += total_tokens
            
            if len(prompt_perplexities) > 0:
                avg_ppl = sum(prompt_perplexities) / len(prompt_perplexities)
                eval_logger.info(f"Average prompt perplexity: {avg_ppl:.4f} (computed on {len(prompt_perplexities)}/{len(requests)} prompts, {total_prompt_tokens} tokens)")
                print(f"Average prompt perplexity: {avg_ppl:.4f} (computed on {len(prompt_perplexities)}/{len(requests)} prompts, {total_prompt_tokens} tokens)")
        
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[tuple[float, bool]]:
        """
        Compute log-likelihood of generating a continuation from a context.
        
        For diffusion models, we use autoregressive mode to compute log-likelihood.
        """
        res = []
        
        # Access the underlying model when wrapped in DDP
        model = (
            self.model.module
            if hasattr(self.model, 'module')
            else self.model
        )
        
        for request in tqdm(requests, disable=not self.accelerator.is_main_process, desc="Computing loglikelihood"):
            context, continuation = request.args
            
            # Tokenize context and continuation
            if context == "":
                # Use prefix token (EOS/BOS) as context
                context_enc = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else []
                continuation_enc = self.tokenizer.encode(continuation, add_special_tokens=False)
            else:
                # Encode full sequence and split
                full_enc = self.tokenizer.encode(context + continuation, add_special_tokens=False)
                context_enc = self.tokenizer.encode(context, add_special_tokens=False)
                continuation_enc = full_enc[len(context_enc):]
            
            if len(continuation_enc) == 0:
                res.append((float('-inf'), False))
                continue
            
            # Combine context and continuation
            input_ids = context_enc + continuation_enc
            
            # Truncate if too long
            if len(input_ids) > self.max_length + 1:
                input_ids = input_ids[-(self.max_length + 1):]
                context_enc = input_ids[:-len(continuation_enc)]
                continuation_enc = input_ids[-len(continuation_enc):]
            
            # Create input tensor (remove last token for prediction)
            input_tensor = torch.tensor([input_ids[:-1]], device=self.device)
            target_tensor = torch.tensor([continuation_enc], device=self.device)
            
            # Forward pass with is_causal=True for autoregressive mode
            with torch.no_grad():
                outputs = model(input_ids=input_tensor, is_causal=True)
                logits = outputs.logits
                
                # Shift logits to align with targets (same as training)
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
                
                # Get logits for continuation tokens only
                # We need to get the logits at positions corresponding to continuation
                context_len = len(context_enc)
                cont_logits = logits[:, context_len-1:context_len-1+len(continuation_enc), :]
                
                # Calculate log-likelihood
                log_likelihood, num_tokens = calculate_log_likelihood_from_logits(
                    cont_logits, target_tensor
                )
                
                # Check if greedy (argmax matches continuation)
                greedy_tokens = cont_logits.argmax(dim=-1)
                is_greedy = (greedy_tokens.squeeze(0) == target_tensor.squeeze(0)).all().item()
                
                res.append((log_likelihood, is_greedy))
        
        return res

    def loglikelihood_rolling(
        self, requests: List[Instance]
    ) -> List[float]:
        """
        Stub implementation - not used for HumanEval/MBPP tasks.
        Required by LM base class but not called for generation tasks.
        """
        raise NotImplementedError(
            "loglikelihood_rolling is not implemented. "
            "This model is only used for generation tasks (HumanEval, MBPP) which use generate_until."
        )

    def _measure_inference_time(self, prompts: List[str], num_warmup: int = 2) -> float:
        """
        Measure average inference time (excluding tokenization/decoding) on given prompts.
        
        Args:
            prompts: List of prompt strings to measure on
            num_warmup: Number of warmup runs before timing
        
        Returns:
            Average inference time in seconds
        """
        if not self.accelerator.is_main_process:
            return 0.0
        
        # Warmup runs
        for i in range(num_warmup):
            prompt = prompts[min(i, len(prompts) - 1)]
            inputs = self.tokenizer(
                [prompt],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length - self.max_new_tokens,
            ).to(self.device)
            
            generate_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "do_sample": self.do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            model = (
                self.model.module
                if hasattr(self.model, 'module')
                else self.model
            )
            
            with torch.no_grad():
                _ = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    **generate_kwargs
                )
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # Actual timing runs
        inference_times = []
        for prompt in prompts:
            # Tokenize (not timed)
            inputs = self.tokenizer(
                [prompt],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length - self.max_new_tokens,
            ).to(self.device)
            
            generate_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "do_sample": self.do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            model = (
                self.model.module
                if hasattr(self.model, 'module')
                else self.model
            )
            
            # Measure only inference time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            with torch.no_grad():
                _ = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    **generate_kwargs
                )
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            inference_times.append(end_time - start_time)
        
        return sum(inference_times) / len(inference_times) if inference_times else 0.0

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self.accelerator.process_index

    @property
    def world_size(self):
        return self.accelerator.num_processes


# Removed ensure_perplexity_in_results and measure_and_inject_inference_time
# Metrics are now calculated directly during evaluation:
# - Inference time: measured in generate_until() method
# - Perplexity: calculated on prompts in generate_until() method (for generation tasks)


if __name__ == "__main__":
    # This allows us to run the evaluation directly from the command line
    # using lm-harness's built-in argument parser.
    import sys
    from eval_utils import upload_results_after_eval

    # Set default model to custom_coder if not specified
    if '--model' not in sys.argv:
        # Insert --model custom_coder after the script name
        sys.argv.insert(1, 'custom_coder')
        sys.argv.insert(1, '--model')
        eval_logger.info("No --model specified, defaulting to 'custom_coder'")

    # Remove wandb_project_name from sys.argv before calling cli_evaluate
    # since lm-harness doesn't recognize this argument
    wandb_project_name = None
    if '--wandb_project_name' in sys.argv:
        idx = sys.argv.index('--wandb_project_name')
        if idx + 1 < len(sys.argv):
            wandb_project_name = sys.argv[idx + 1]
            # Remove both the flag and its value
            sys.argv = sys.argv[:idx] + sys.argv[idx + 2:]
    
    # Get output_path before cli_evaluate modifies sys.argv
    output_path = None
    for i, arg in enumerate(sys.argv):
        if arg == "--output_path" and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]
            break
    
    # Check if any perplexity tasks are being evaluated and add default limit=100 if not specified
    # Perplexity tasks use loglikelihood_rolling output type - check OUTPUT_TYPE, not keywords
    has_limit = '--limit' in sys.argv
    if not has_limit:
        # Import tasks to check OUTPUT_TYPE
        from lm_eval import tasks
        
        tasks_idx = None
        for i, arg in enumerate(sys.argv):
            if arg == "--tasks" and i + 1 < len(sys.argv):
                tasks_idx = i + 1
                break
        
        if tasks_idx is not None:
            tasks_str = sys.argv[tasks_idx]
            tasks_list = [t.strip() for t in tasks_str.split(',')]
            
            # Check if any task uses loglikelihood_rolling output type
            has_perplexity_task = False
            for task_name in tasks_list:
                try:
                    # Initialize tasks to check their OUTPUT_TYPE
                    tasks.initialize_tasks()
                    if task_name in tasks.TASK_REGISTRY:
                        task = tasks.TASK_REGISTRY[task_name]()
                        # Check if task uses loglikelihood_rolling (perplexity)
                        if hasattr(task, 'OUTPUT_TYPE') and task.OUTPUT_TYPE == "loglikelihood_rolling":
                            has_perplexity_task = True
                            eval_logger.info(f"Perplexity task detected: '{task_name}' uses loglikelihood_rolling")
                            break
                except Exception as e:
                    # If task can't be loaded, skip it
                    eval_logger.debug(f"Could not check task '{task_name}': {e}")
                    continue
            
            if has_perplexity_task:
                # Insert --limit 100 before --tasks
                sys.argv.insert(tasks_idx, '100')
                sys.argv.insert(tasks_idx, '--limit')
                eval_logger.info("Adding default --limit 100 for perplexity evaluation (faster)")
    
    cli_evaluate()
    
    # Metrics (inference time and perplexity) are now calculated directly during evaluation
    # No need to parse results files - they're printed during the evaluation process
    
    # Only upload to wandb if wandb_project_name was provided
    if wandb_project_name:
        upload_results_after_eval(wandb_project_name)
    else:
        print("Wandb project name not provided - skipping wandb logging")
