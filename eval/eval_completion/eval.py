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

eval_logger = logging.getLogger("eval_logger")

# Global variable to store model instance for inference time measurement
_global_model_instance = None


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
        
        # Store globally for inference time measurement
        global _global_model_instance
        if self.accelerator.is_main_process:
            _global_model_instance = self

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
            eval_logger.info(f"Detected checkpoint file: {pretrained}")
            
            # Determine base model path
            if self.base_model:
                base_model_path = self.base_model
            else:
                # Default base model path (same as used in training)
                base_model_path = '/orcd/data/tpoggio/001/tiffany8/dllm-distillation/models/open-dcoder-0.5B'
                if not os.path.exists(base_model_path):
                    # Fallback to HuggingFace model ID
                    base_model_path = "fredzzp/open-dcoder-0.5B"
            
            eval_logger.info(f"Loading base model from: {base_model_path}")
            eval_logger.info(f"Loading tokenizer from base model: {base_model_path}")
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                base_model_path,
                trust_remote_code=self.trust_remote_code,
            )
            
            eval_logger.info(f"Loading base model: {base_model_path}")
            self.model = Qwen2ForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=get_dtype(dtype),
                trust_remote_code=self.trust_remote_code,
            )
            
            # Load checkpoint state dict
            eval_logger.info(f"Loading checkpoint weights from: {pretrained}")
            checkpoint = torch.load(pretrained, map_location="cpu")
            
            # Handle both 'model_state_dict' (from training checkpoints) and direct state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                eval_logger.info(f"Checkpoint info: epoch={checkpoint.get('epoch', 'N/A')}, "
                               f"batch={checkpoint.get('batch_idx', 'N/A')}, "
                               f"loss={checkpoint.get('total_loss', 'N/A'):.4f}")
            else:
                state_dict = checkpoint
            
            # Load state dict directly (model is not wrapped yet at this point)
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                eval_logger.warning(f"Missing keys when loading checkpoint: {len(missing_keys)} keys")
            if unexpected_keys:
                eval_logger.warning(f"Unexpected keys when loading checkpoint: {len(unexpected_keys)} keys")
            
            eval_logger.info("✓ Checkpoint loaded successfully")
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
        """Generates text for a batch of prompts using diffusion_generate."""

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

        # Use specific yaml parameters: num_return_sequences
        # Other parameters still come from eval.sh (model_args)
        # Note: do_sample is automatically set to True by MDMGenerationConfig
        num_return_sequences = gen_kwargs.get('num_return_sequences', 1)

        # Create a generation configuration object
        generation_config = MDMGenerationConfig(
            mask_token_id=self.tokenizer.mask_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            # Parameters from eval.sh (model_args) - unchanged
            max_new_tokens=self.max_new_tokens,
            steps=self.steps,
            temperature=self.temperature,
            top_k=self.top_k,
            alg=self.alg,
            alg_temp=self.alg_temp,
            # Parameters from yaml - override model_args
            num_return_sequences=num_return_sequences,
            return_dict_in_generate=True,
            output_history=False
        )

        with torch.no_grad():
            # Access the underlying model when wrapped in DDP
            model = (
                self.model.module
                if hasattr(self.model, 'module')
                else self.model
            )
            outputs = model.diffusion_generate(
                inputs=inputs.input_ids,
                generation_config=generation_config,
            )

        # Decode the generated sequences, skipping the prompt
        generated_sequences = outputs.sequences
        
        # Reshape to group by original prompts
        # Shape: [batch_size * num_return_sequences, seq_len]
        batch_size = len(prompts)
        num_seqs = num_return_sequences
        
        # Decode all sequences
        all_generated_texts = self.tokenizer.batch_decode(
            generated_sequences[:, prompt_lengths:],
            skip_special_tokens=True
        )
        
        # For now, just return the first sequence for each prompt
        # Since we changed to repeats=10, each call should generate 1 sequence
        generated_texts = []
        for i in range(batch_size):
            start_idx = i * num_seqs
            generated_texts.append(all_generated_texts[start_idx])

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
        return res

    # The loglikelihood methods are not required for generation-based tasks
    # like HumanEval. We can leave them as not implemented to simplify the
    # initial effort.
    def loglikelihood(self, requests):
        raise NotImplementedError(
            "Loglikelihood not implemented for this model."
        )

    def loglikelihood_rolling(
        self, requests: List[Instance]
    ) -> List[float]:
        raise NotImplementedError(
            "Loglikelihood rolling not implemented for this model."
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

        # Load the model and tokenizer
        self._create_model_and_tokenizer(pretrained, dtype)
        
        # Store globally for inference time measurement
        global _global_model_instance
        if self.accelerator.is_main_process:
            _global_model_instance = self

    def _create_model_and_tokenizer(self, pretrained, dtype):
        """Loads the standard AutoModelForCausalLM model and its tokenizer."""
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
        return res

    def loglikelihood(self, requests):
        raise NotImplementedError(
            "Loglikelihood not implemented for this model."
        )

    def loglikelihood_rolling(
        self, requests: List[Instance]
    ) -> List[float]:
        raise NotImplementedError(
            "Loglikelihood rolling not implemented for this model."
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


def measure_and_inject_inference_time(output_path: Optional[str] = None):
    """
    Measure inference time on 10 fixed samples and inject into results JSON.
    """
    global _global_model_instance
    
    if _global_model_instance is None:
        eval_logger.warning("No model instance available for inference time measurement")
        return
    
    if not _global_model_instance.accelerator.is_main_process:
        return
    
    # Find the latest results file
    if output_path is None:
        # Try to get from command line args
        import sys
        for i, arg in enumerate(sys.argv):
            if arg == "--output_path" and i + 1 < len(sys.argv):
                output_path = sys.argv[i + 1]
                break
    
    if output_path is None or not os.path.exists(output_path):
        eval_logger.warning(f"Output path not found: {output_path}, skipping inference time measurement")
        return
    
    # Find latest results file
    results_pattern = os.path.join(output_path, "*", "results_*.json")
    results_files = glob.glob(results_pattern)
    
    if not results_files:
        eval_logger.warning("No results files found, skipping inference time measurement")
        return
    
    # Get the latest file based on timestamp
    def extract_timestamp(filepath):
        match = re.search(
            r'(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+)',
            filepath
        )
        return match.group(1) if match else ""
    
    latest_results = max(results_files, key=extract_timestamp)
    
    # Load results
    with open(latest_results, 'r') as f:
        results_data = json.load(f)
    
    # Get task names from results
    task_names = list(results_data.get("results", {}).keys())
    if not task_names:
        eval_logger.warning("No tasks found in results, skipping inference time measurement")
        return
    
    # Measure inference time for each task
    configs = results_data.get("configs", {})
    results_updated = False
    
    for task_name in task_names:
        # Try to get actual task name from configs if available
        if task_name in configs:
            actual_task_name = configs[task_name].get("task", task_name)
        else:
            actual_task_name = task_name
        
        # Load task and get 10 fixed samples
        try:
            # Try the actual task name first, then fall back to task_name
            if actual_task_name in tasks.TASK_REGISTRY:
                task = tasks.TASK_REGISTRY[actual_task_name]()
            elif task_name in tasks.TASK_REGISTRY:
                task = tasks.TASK_REGISTRY[task_name]()
            else:
                eval_logger.warning(f"Task '{task_name}' or '{actual_task_name}' not found in TASK_REGISTRY, skipping")
                continue
            
            eval_docs = task.eval_docs
            
            # Get first 10 samples
            num_samples = min(10, len(eval_docs))
            sample_docs = list(eval_docs)[:num_samples]
            
            # Convert docs to prompts using task's doc_to_text
            prompts = [task.doc_to_text(doc) for doc in sample_docs]
            
            eval_logger.info(f"Measuring inference time for task '{task_name}' on {len(prompts)} fixed samples...")
            
            # Measure inference time
            avg_time = _global_model_instance._measure_inference_time(prompts)
            
            eval_logger.info(f"Average inference time for '{task_name}': {avg_time:.4f} seconds")
            
            # Inject into results
            if task_name not in results_data["results"]:
                results_data["results"][task_name] = {}
            
            results_data["results"][task_name]["avg_inference_time_seconds"] = avg_time
            results_updated = True
            
        except Exception as e:
            eval_logger.warning(f"Failed to measure inference time for task '{task_name}': {e}")
            continue
    
    # Save updated results if any task was measured
    if results_updated:
        with open(latest_results, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        eval_logger.info(f"Inference time added to results: {latest_results}")


if __name__ == "__main__":
    # This allows us to run the evaluation directly from the command line
    # using lm-harness's built-in argument parser.
    import sys
    from eval_utils import upload_results_after_eval

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
    
    cli_evaluate()
    
    # Measure and inject inference time into results
    measure_and_inject_inference_time(output_path)
    
    # Only upload to wandb if wandb_project_name was provided
    if wandb_project_name:
        upload_results_after_eval(wandb_project_name)
    else:
        print("Wandb project name not provided - skipping wandb logging")
