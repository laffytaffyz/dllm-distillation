import logging
from typing import List, Optional, Union, Tuple

import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import get_dtype
from lm_eval.__main__ import cli_evaluate
from tqdm import tqdm

'''
To run: bash /eval/eval_completion/eval_perplexity_klu_tiff.sh
'''

# Import your custom model and generation config
from veomni.models.transformers.qwen2.modeling_qwen2 import (
    Qwen2ForCausalLM
)
from veomni.models.transformers.qwen2.generation_utils import (
    MDMGenerationConfig,
    sample_tokens,
)

eval_logger = logging.getLogger("eval_logger")

def log_to_file(file, content):
    with open(file, "a") as f:
        f.write(content)
    return

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
        """Loads the Qwen2ForCausalLM model and its tokenizer."""
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
    def compute_perplexity(
        self,
        texts: List[str],
        return_stats: bool = False
    ) -> Optional[Union[float, Tuple[float, float, int]]]:
        """
        Compute diffusion-style perplexity.

        We follow the diffusion schedule (p2) and only accumulate loss on the
        tokens that are *currently masked* at each step. Tokens that are never
        masked (e.g., padding) are ignored. Tokens are filled with the ground
        truth after each step (teacher forcing) so the schedule matches the
        number of active/unmasked positions per step.
        """
        if not texts:
            return None

        mask_token_id = self.tokenizer.mask_token_id
        if mask_token_id is None:
            eval_logger.error("Mask token id is not set; cannot compute diffusion perplexity.")
            return None

        encodings = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        target_ids = encodings.input_ids.to(self.device)
        attn_mask = encodings.attention_mask.to(self.device) # tiff: attention mask = just padding

        # Initialize sequence fully masked (padding stays padding so it is fixed)
        x = torch.full_like(target_ids, fill_value=mask_token_id)
        if self.tokenizer.pad_token_id is not None:
            x = x.masked_fill(attn_mask == 0, self.tokenizer.pad_token_id)

        # Fixed positions are those that are not mask tokens (padding, etc.)
        fix_mask = x != mask_token_id

        gen_attention_mask = (x != self.tokenizer.pad_token_id).long() if self.tokenizer.pad_token_id is not None else None

        steps = self.steps
        temperature = self.temperature
        top_p = None  # not used in current config, keep for completeness
        top_k = self.top_k
        alg = self.alg
        eps = 1e-3
        timesteps = torch.linspace(1, eps, steps + 1, device=self.device)

        total_loss = torch.tensor(0.0, device=self.device)
        valid_tokens = torch.tensor(0, device=self.device)

        # Underlying model (handle DDP)
        model = self.model.module if hasattr(self.model, "module") else self.model

        for i in range(steps):
            log_to_file("output_perplexity.log", f"step {i}:")
            mask_index = (x == mask_token_id) & (attn_mask == 1)
            if not mask_index.any():
                break

            outputs = model(input_ids=x, attention_mask=gen_attention_mask, is_causal=False)
            logits = outputs.logits 

            # Align with training: shift logits right by one
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1) # tiff: why are we shifting logits

            # # tiff: this accidentaly account for tokens that will be masked
            # # tiff: (update) moved below to use the correct masking!
            # mask_logits = logits[mask_index]
            # mask_targets = target_ids[mask_index] 
            #
            #  step_loss = F.cross_entropy(
            #     mask_logits,
            #     mask_targets,
            #     reduction="sum",
            # )
            #  total_loss += step_loss
            #  log_to_file("output_perplexity.log", f"step {i} loss: {step_loss}")
            #  valid_tokens += mask_targets.numel() # tiff: this overcounts because adding in 0 entries and also accounts for tokens that will be masked

            # Update schedule: mimic p2 remasking logic using confidence,
            # but fill with ground-truth tokens (teacher forcing). # tiff: maybe add in total_loss and valid_token calculation here
            if alg == "p2":
                kappa_t = (i + 1) / steps
                conf_full, _ = sample_tokens(
                    logits, temperature=temperature, top_p=top_p, top_k=top_k, alg=alg
                )

                full_conf = conf_full.clone()
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
                    row_mask = torch.arange(max_k, device=x.device).unsqueeze(0) < num_to_mask.unsqueeze(1)

                    to_mask = torch.zeros_like(x, dtype=torch.bool)
                    batch_arange = torch.arange(x.size(0), device=x.device).unsqueeze(1).expand_as(topk_idx)
                    valid_batch = batch_arange[row_mask]
                    valid_col = topk_idx[row_mask]
                    to_mask[valid_batch, valid_col] = True
                else:
                    to_mask = torch.zeros_like(x, dtype=torch.bool)

                # tiff: (NEW) this is mask for tokens that are now newly unmasked and we want to keep unmasked as we remask everything else
                keep_unmask = mask_index & (~to_mask) 

                unmask_logits = logits[keep_unmask]
                unmask_targets = target_ids[keep_unmask] 
                
                step_loss = F.cross_entropy(
                    unmask_logits,
                    unmask_targets,
                    reduction="sum",
                )
                total_loss += step_loss
                log_to_file("output_perplexity.log", f"step {i} loss: {step_loss}")
                valid_tokens += keep_unmask.sum().item()
                # print(f"total_loss: {total_loss}, valid_tokens: {valid_tokens}, keep_unmask: {keep_unmask}")
                # tiff: (NEW)

                x[keep_unmask] = target_ids[keep_unmask]
                # keep padding fixed
                x = x.masked_fill(attn_mask == 0, self.tokenizer.pad_token_id)
                fix_mask = x != mask_token_id
            else:
                eval_logger.warning(f"Perplexity schedule for alg={alg} not implemented; stopping early.")
                break

        # Reduce across processes
        total_loss = self.accelerator.reduce(total_loss, reduction="sum")
        valid_tokens = self.accelerator.reduce(valid_tokens, reduction="sum")

        # print(f"total_loss: {total_loss}, valid_tokens: {valid_tokens}, target_ids.shape[1]: {target_ids.shape[1]}")
        # print(f"target_ids: {target_ids}")
        assert valid_tokens <= target_ids.shape[1]

        log_to_file(f"output_perplexity_{self.steps}.log", f"total_loss: {total_loss}, normalized_loss:{total_loss // valid_tokens}")

        if valid_tokens.item() == 0:
            return None

        ppl = torch.exp(total_loss / valid_tokens)
        if self.accelerator.is_main_process:
            if return_stats:
                return ppl.item(), total_loss.item(), int(valid_tokens.item())
            return ppl.item()
        return None

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
    
    cli_evaluate()
    
    # Only upload to wandb if wandb_project_name was provided
    if wandb_project_name:
        upload_results_after_eval(wandb_project_name)
    else:
        print("Wandb project name not provided - skipping wandb logging")
