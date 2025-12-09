import logging
import os
import json
import glob
import re
import time
import math
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

from veomni.models.transformers.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from veomni.models.transformers.qwen2.generation_utils import MDMGenerationConfig
from eval_utils import calculate_log_likelihood_from_logits, get_rolling_token_windows

eval_logger = logging.getLogger("eval_logger")
_sample_print_count = 0
_MAX_SAMPLES_TO_PRINT = 3


@register_model("custom_coder")
class CustomCoder(LM):
    """
    Diffusion-style coder with:
      - generate_until() supporting n>1 candidates per prompt (for pass@k)
      - diffusion perplexity defined as: NLL over just-unmasked tokens, summed over all steps, divided by L
    """
    def __init__(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        batch_size: Optional[Union[int, str]] = 1,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        max_new_tokens: Optional[int] = 128,
        steps: Optional[int] = 100,
        temperature: Optional[float] = 0.5,
        top_k: Optional[int] = 200,
        alg: Optional[str] = "p2",
        alg_temp: Optional[float] = 0.5,
        trust_remote_code: Optional[bool] = True,
        base_model: Optional[str] = None,
        max_length: Optional[int] = 2048,
        **kwargs,
    ) -> None:
        super().__init__()
        self.accelerator = Accelerator()
        self._device = self.accelerator.device
        self.batch_size_per_gpu = int(batch_size)
        self.max_length = int(max_length)
        self.trust_remote_code = bool(trust_remote_code)
        self.base_model = base_model

        # diffusion sampling params
        self.max_new_tokens = int(max_new_tokens)
        self.steps = int(steps)
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.alg = str(alg)
        self.alg_temp = float(alg_temp)

        self._init_parallel_state()
        self._create_model_and_tokenizer(pretrained, dtype)

    def _force_string(self, x):
        if isinstance(x, str):
            return x
        if isinstance(x, bytes):
            return x.decode('utf-8', errors='ignore')
        if isinstance(x, (list, tuple)):
            # fully recursive flatten
            return "".join(self._force_string(y) for y in x)
        return str(x)

    # -------------------------
    # Dist / parallel utilities
    # -------------------------
    def _init_parallel_state(self):
        from veomni.distributed.parallel_state import init_parallel_state
        import torch.distributed as dist

        if self.accelerator.num_processes == 1:
            return

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")

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

    # -------------------------
    # Model / tokenizer loading
    # -------------------------
    def _create_model_and_tokenizer(self, pretrained, dtype):
        is_checkpoint = isinstance(pretrained, str) and pretrained.endswith(".pt") and os.path.isfile(pretrained)

        if is_checkpoint:
            eval_logger.info(f"Loading diffusion checkpoint: {pretrained}")
            base_model_path = self.base_model or "/orcd/data/tpoggio/001/tiffany8/dllm-distillation/models/open-dcoder-0.5B"
            if not os.path.exists(base_model_path):
                base_model_path = "fredzzp/open-dcoder-0.5B"

            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                base_model_path, trust_remote_code=self.trust_remote_code
            )
            self.model = Qwen2ForCausalLM.from_pretrained(
                base_model_path, torch_dtype=get_dtype(dtype), trust_remote_code=self.trust_remote_code
            )

            checkpoint = torch.load(pretrained, map_location="cpu")
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                eval_logger.warning(f"[diffusion] Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                eval_logger.warning(f"[diffusion] Unexpected keys: {len(unexpected_keys)}")
        else:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                pretrained, trust_remote_code=self.trust_remote_code
            )
            self.model = Qwen2ForCausalLM.from_pretrained(
                pretrained, torch_dtype=get_dtype(dtype), trust_remote_code=self.trust_remote_code
            )

        # Ensure mask/pad tokens exist and are consistent
        if self.tokenizer.mask_token is None:
            self.tokenizer.add_special_tokens({"mask_token": "[MASK]"})
            self.model.resize_token_embeddings(len(self.tokenizer))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "left"
        self.model = self.accelerator.prepare(self.model)
        self._device = self.accelerator.device
        self.model.eval()

    # -------------------------
    # Diffusion generation
    # -------------------------
    def _generate_batch_single(self, prompts: List[str], gen_kwargs: dict | None = None) -> List[str]:
        """
        One sample per prompt. Returns a flat list[str] of size len(prompts),
        each string is the *completion only* (prompt trimmed).
        """
        global _sample_print_count
        gen_kwargs = gen_kwargs or {}

        # reserve space for completions so the prompt fits in max_length
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length - self.max_new_tokens,
        ).to(self.device)
        prompt_len = inputs.input_ids.shape[1]

        model = self.model.module if hasattr(self.model, "module") else self.model

        with torch.no_grad():
            out = model.diffusion_generate(
                inputs=inputs.input_ids,
                generation_config=MDMGenerationConfig(
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
                    output_history=False,
                ),
            )

        if hasattr(out, "sequences"):
            generated_ids = out.sequences
        elif isinstance(out, torch.Tensor):
            generated_ids = out
        else:
            generated_ids = out[0] if isinstance(out, (tuple, list)) else out

        # decode *completion only* (no prompt leakage!)
        decoded = self.tokenizer.batch_decode(
            generated_ids[:, prompt_len:], skip_special_tokens=True
        )

        # Ensure each decoded output is a string
        completions = []

        completions = [self._force_string(c) for c in decoded]

        # sample logging
        if _sample_print_count < _MAX_SAMPLES_TO_PRINT and self.accelerator.is_main_process:
            slots = min(len(prompts), _MAX_SAMPLES_TO_PRINT - _sample_print_count)
            for j in range(slots):
                eval_logger.info("=" * 80)
                eval_logger.info(f"SAMPLE PROMPT/RESPONSE #{_sample_print_count + j + 1}")
                eval_logger.info(f"PROMPT: {prompts[j][:500]}{'...' if len(prompts[j]) > 500 else ''}")
                eval_logger.info(f"RESPONSE: {completions[j][:500]}{'...' if len(completions[j]) > 500 else ''}")
            _sample_print_count += slots

        return completions

    def _generate_batch(self, prompts: List[str], gen_kwargs: dict | None = None) -> List[List[str]]:
        """
        Returns a grouped structure [[s1..sn], [s1..sn], ...] of length len(prompts),
        each inner list has n candidates (for pass@k).
        """
        gen_kwargs = dict(gen_kwargs or {})
        n = int(gen_kwargs.get("num_return_sequences", gen_kwargs.get("n_samples", 1)))
        # do not leak lm-harness specific args
        gen_kwargs.pop("until", None)

        # accumulate per-prompt lists
        grouped = [[] for _ in range(len(prompts))]

        base_seed = gen_kwargs.get("seed", None)
        for j in range(n):
            if base_seed is not None:
                torch.manual_seed(int(base_seed) + j)

            single_kwargs = {k: v for k, v in gen_kwargs.items() if k not in ("num_return_sequences", "n_samples")}
            outs = self._generate_batch_single(prompts, single_kwargs)

            # append j-th candidate for each prompt
            for i, o in enumerate(outs):
                grouped[i].append(o)

        return grouped

    # -------------------------
    # Diffusion perplexity
    # -------------------------
    def _compute_diffusion_perplexity(self, text: str) -> Optional[float]:
        """
        Progressive-denoising perplexity for discrete DLMs.

        Spec: NLL is computed *only* on tokens newly unmasked at each step, summed over steps, then divided by L.
              PPL = exp( NLL )

        Implementation details:
          - Reveal schedule chosen by *confidence from log-probs* (temperature-free)
          - Teacher-force revealed content with ground-truth x0
          - Last step reveals all remaining tokens to ensure each position is scored exactly once
        """
        import math
        import torch.nn.functional as F

        model = self.model.module if hasattr(self.model, "module") else self.model
        device = self.device

        # Tokenize ground truth properly (encode() doesn't support return_tensors)
        enc = self.tokenizer(
            text, add_special_tokens=False, return_tensors="pt"
        )
        x0 = enc.input_ids.to(device)  # [1, L]
        B, L = x0.shape
        if L == 0:
            return float("nan")

        mask_id = self.tokenizer.mask_token_id
        if mask_id is None:
            raise ValueError("Tokenizer must define mask_token_id before computing diffusion perplexity.")

        # Start fully masked
        x_t = torch.full_like(x0, mask_id)
        masked = torch.ones_like(x0, dtype=torch.bool)  # True if currently masked

        nll_terms = []

        with torch.no_grad():
            for step in range(self.steps):
                # Forward (disable causal masking; diffusion uses full context)
                try:
                    out = model(input_ids=x_t, is_causal=False)
                except TypeError:
                    out = model(input_ids=x_t)
                logits = out.logits  # [B, L, V]
                log_probs = F.log_softmax(logits, dim=-1)

                # Confidence for schedule selection (no temp; use log-prob for stability)
                conf, _ = log_probs.max(dim=-1)  # [B, L]

                # Target number of unmasked positions by step
                # Use ceil & clamp, and make last step reveal all remaining
                kappa = (step + 1) / self.steps
                target_unmasked = torch.clamp(
                    torch.ceil(torch.tensor(kappa * L, device=device)), min=1, max=L
                ).long()  # scalar (apply per-b with same target)

                current_unmasked = (~masked).sum(dim=1)  # [B]
                # On the final step, force full reveal
                if step == self.steps - 1:
                    target_unmasked = torch.full_like(target_unmasked, L)

                num_new = (target_unmasked - current_unmasked).clamp(min=0)  # [B]
                if num_new.max().item() == 0:
                    continue

                newly = torch.zeros_like(masked)  # [B, L] bool

                # select new positions independently per sample
                for b in range(B):
                    k = num_new[b].item()
                    if k <= 0:
                        continue
                    conf_masked = conf[b].clone()
                    conf_masked[~masked[b]] = -float("inf")  # exclude already revealed
                    topk = conf_masked.topk(k, dim=-1).indices
                    newly[b, topk] = True

                # NLL over newly revealed positions
                # log p(x0_i | x_t, t) at the ground-truth ids
                log_p_all = log_probs.gather(2, x0.unsqueeze(-1)).squeeze(-1)  # [B, L]
                nll_step = -log_p_all[newly]
                if nll_step.numel() > 0:
                    nll_terms.append(nll_step)

                # Teacher-force: insert ground truth at revealed positions
                x_t = x_t.clone()
                masked = masked.clone()
                x_t[newly] = x0[newly]
                masked[newly] = False

        if not nll_terms:
            return float("nan")

        total_nll = torch.cat(nll_terms, dim=0).sum()  # scalar
        NLL = total_nll / L  # divided by sequence length L (spec)
        PPL = math.exp(NLL.item())
        return PPL

    # -------------------------
    # lm-harness interfaces
    # -------------------------
    @torch.no_grad()
    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False) -> List[List[str]]:
        """
        Return shape must be: List[List[str]]
        - outer length = len(requests)
        - each inner list = candidates (strings) for that single request
        """
        results: List[List[str]] = []
        inference_times = []
        measure_inference = self.accelerator.is_main_process and len(requests) > 0

        disable_tqdm = disable_tqdm or not self.accelerator.is_main_process
        pbar = tqdm(total=len(requests), disable=disable_tqdm, desc="Running generate_until (diffusion)")

        for i in range(0, len(requests), self.batch_size):
            batch_requests = requests[i : i + self.batch_size]
            contexts = [req.args[0] for req in batch_requests]

            # harness may pass per-request gen kwargs in args[1]
            first_req_args = batch_requests[0].args
            gen_kwargs = first_req_args[1] if len(first_req_args) > 1 else {}
            # don't leak lm-harness-specific args downwards
            gen_kwargs = dict(gen_kwargs)
            gen_kwargs.pop("until", None)

            # warmup for timing
            if measure_inference and i == 0:
                _ = self._generate_batch(contexts[:1], gen_kwargs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            # grouped: List[List[str]] (per prompt, n candidates)
            grouped = self._generate_batch(contexts, gen_kwargs)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            # post-process 'until' per *string*, keep exactly List[str] per request
            for cand_list, req in zip(grouped, batch_requests):
                stop_sequences = self._get_until(req)
                processed: List[str] = []
                for s in cand_list:
                    s_str = self._force_string(s)

                    s_clean = s_str
                    for stop in stop_sequences:
                        if stop and stop in s_clean:
                            s_clean = s_clean.split(stop)[0]

                    processed.append(s_clean)
                results.append(processed)

            # timing per prompt (not per candidate)
            batch_time = (end_time - start_time) / max(1, len(batch_requests))
            inference_times.extend([batch_time] * len(batch_requests))
            pbar.update(len(batch_requests))

        pbar.close()

        if measure_inference and inference_times:
            avg_time = sum(inference_times) / len(inference_times)
            eval_logger.info(f"[diffusion] Average inference time: {avg_time:.4f} seconds per prompt (group size n preserved)")
            print(f"\n[diffusion] Average inference time: {avg_time:.4f} seconds per prompt (group size n preserved)")


        # FINAL CLEANUP – required for HumanEval
        cleaned_results = []
        for cand_list in results:
            cleaned_sub = []
            for c in cand_list:
                cleaned_sub.append(self._force_string(c))  # recursively flatten and stringify
            cleaned_results.append(cleaned_sub)

        final = []
        for cand_list in cleaned_results:
            cleaned = []
            for c in cand_list:
                s = self._force_string(c)
                if not isinstance(s, str):
                    s = str(s)
                cleaned.append(s)
            final.append(cleaned)

        
        # Strong assertion
        for out in final:
            for x in out:
                if not isinstance(x, str):
                    raise ValueError(f"BAD OUTPUT TYPE at final return: {type(x)} → {x}")

        return final


    def loglikelihood(self, requests: List[Instance]) -> List[tuple[float, bool]]:
        raise NotImplementedError("Progressive denoising models do not support AR loglikelihood.")

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        raise NotImplementedError("Progressive denoising models do not support AR loglikelihood.")

    # -------------------------
    # helpers / properties
    # -------------------------
    def _get_until(self, req):
        if len(req.args) > 1 and isinstance(req.args[1], dict):
            return req.args[1].get("until", [])
        return []

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
        Optional: average wall-clock time per prompt for diffusion_generate (excludes tokenization/decoding).
        """
        if not self.accelerator.is_main_process:
            return 0.0

        # Warmup
        for i in range(min(num_warmup, len(prompts))):
            _ = self._generate_batch(prompts[i:i+1], {})
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        # Timed runs
        times = []
        for p in prompts:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = self._generate_batch([p], {})
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)
        return sum(times) / len(times) if times else 0.0


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
    
    cli_evaluate()
    
    # Only upload to wandb if wandb_project_name was provided
    if wandb_project_name:
        upload_results_after_eval(wandb_project_name)
    else:
        print("Wandb project name not provided - skipping wandb logging")