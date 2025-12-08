import os
import time
import json
import logging
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.api.instance import Instance
from lm_eval.models.utils import get_dtype

from veomni.models.transformers.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from veomni.models.transformers.qwen2.generation_utils import MDMGenerationConfig

eval_logger = logging.getLogger("eval_logger")
_sample_print_count = 0
_MAX_PRINT = 3


# ============================================================
#               PROGRESSIVE-DENOISING LM (P2)
# ============================================================

@register_model("custom_coder")
class CustomCoder(LM):

    # --------------------------------------------------------
    #   Initialization
    # --------------------------------------------------------
    def __init__(
        self,
        pretrained: str,
        batch_size: int = 1,
        device: str = "cuda",
        dtype: Union[str, torch.dtype] = "auto",
        max_new_tokens: int = 128,
        steps: int = 100,
        temperature: float = 0.5,
        top_k: int = 200,
        alg: str = "p2",
        alg_temp: float = 0.5,
        trust_remote_code: bool = True,
        base_model: Optional[str] = None,
        max_length: int = 2048,
        **kwargs,
    ):
        super().__init__()

        self.accelerator = Accelerator()
        self._device = self.accelerator.device
        self.batch_size_per_gpu = batch_size
        self.max_length = max_length
        self.trust_remote_code = trust_remote_code
        self.base_model = base_model

        # diffusion generation hyperparams
        self.max_new_tokens = max_new_tokens
        self.steps = steps
        self.temperature = temperature
        self.top_k = top_k
        self.alg = alg
        self.alg_temp = alg_temp

        # distributed init
        self._init_parallel_state()

        # load model/tokenizer
        self._create_model_and_tokenizer(pretrained, dtype)

    # --------------------------------------------------------
    #   Distributed init
    # --------------------------------------------------------
    def _init_parallel_state(self):
        from veomni.distributed.parallel_state import init_parallel_state
        import torch.distributed as dist

        if self.accelerator.num_processes == 1:
            return
        
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")

        init_parallel_state(
            dp_size=self.accelerator.num_processes,
            tp_size=1, ep_size=1, pp_size=1, cp_size=1,
            dp_mode="ddp", device_type="cuda", include_sp_in_fsdp=True
        )

    # --------------------------------------------------------
    #   Load model/tokenizer
    # --------------------------------------------------------
    def _create_model_and_tokenizer(self, pretrained, dtype):
        is_ckpt = isinstance(pretrained, str) and pretrained.endswith(".pt") and os.path.isfile(pretrained)

        # checkpoint case
        if is_ckpt:
            if self.base_model is None:
                base = "/orcd/data/tpoggio/001/tiffany8/dllm-distillation/models/open-dcoder-0.5B"
                if not os.path.exists(base):
                    base = "fredzzp/open-dcoder-0.5B"
                self.base_model = base

            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.base_model, trust_remote_code=self.trust_remote_code
            )
            self.model = Qwen2ForCausalLM.from_pretrained(
                self.base_model, torch_dtype=get_dtype(dtype), trust_remote_code=self.trust_remote_code
            )

            ckpt = torch.load(pretrained, map_location="cpu")
            sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            self.model.load_state_dict(sd, strict=False)

        else:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                pretrained, trust_remote_code=self.trust_remote_code
            )
            self.model = Qwen2ForCausalLM.from_pretrained(
                pretrained, torch_dtype=get_dtype(dtype), trust_remote_code=self.trust_remote_code
            )

        # ensure mask + pad tokens
        if self.tokenizer.mask_token is None:
            self.tokenizer.add_special_tokens({"mask_token": "[MASK]"})
            self.model.resize_token_embeddings(len(self.tokenizer))

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "left"

        # DDP / FSDP wrapping
        self.model = self.accelerator.prepare(self.model)
        self.model.eval()

    # ============================================================
    #   GENERATION
    # ============================================================

    def _generate_batch_single(self, prompts: List[str], gen_kwargs=None):
        """Correct generation: ALWAYS use model.diffusion_generate."""
        global _sample_print_count

        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=self.max_length - self.max_new_tokens
        ).to(self.device)

        gen_cfg = MDMGenerationConfig(
            mask_token_id=self.tokenizer.mask_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            steps=self.steps,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            alg=self.alg,
            alg_temp=self.alg_temp,
            num_return_sequences=1,
            return_dict_in_generate=True,
        )

        model = self.model.module if hasattr(self.model, "module") else self.model

        with torch.no_grad():
            out = model.diffusion_generate(inputs=inputs.input_ids, generation_config=gen_cfg)

        # remove prompt
        prompt_len = inputs.input_ids.shape[1]
        gen_ids = out.sequences[:, prompt_len:]

        texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        # debug prints
        if _sample_print_count < _MAX_PRINT and self.accelerator.is_main_process:
            for i in range(min(len(prompts), _MAX_PRINT - _sample_print_count)):
                eval_logger.info("======== SAMPLE ========")
                eval_logger.info("PROMPT:")
                eval_logger.info(prompts[i][:400])
                eval_logger.info("RESPONSE:")
                eval_logger.info(texts[i][:400])
                eval_logger.info("========================")
            _sample_print_count += len(prompts)

        return texts

    def _generate_batch(self, prompts: List[str], gen_kwargs=None):
        if gen_kwargs is None:
            gen_kwargs = {}

        n = int(gen_kwargs.get("num_return_sequences", gen_kwargs.get("n_samples", 1)))
        grouped = [[] for _ in prompts]

        base_seed = gen_kwargs.get("seed", None)

        for j in range(n):
            if base_seed is not None:
                torch.manual_seed(base_seed + j)

            outs = self._generate_batch_single(prompts, gen_kwargs)

            for i, o in enumerate(outs):
                # ensure string
                if isinstance(o, list):
                    o = " ".join(map(str, o))
                grouped[i].append(o)

        return grouped


    # ============================================================
    #   PROGRESSIVE-DENOISING PERPLEXITY
    # ============================================================

    def _compute_progressive_ppl(self, text: str):
        """
        Computes EXACT progressive-denoising perplexity:
            NLL = (1/L) * Σ_t Σ_{i in F_t} -log p(x0_i | x_t, t)
        """
        ids = self.tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").to(self.device)
        if ids.size(1) < 2:
            return None

        x0 = ids.clone()
        B, L = x0.shape
        mask_id = self.tokenizer.mask_token_id
        x_t = torch.full_like(x0, mask_id)

        model = self.model.module if hasattr(self.model, "module") else self.model

        # timestep schedule
        eps = 1e-3
        ts = torch.linspace(1.0, eps, self.steps + 1, device=self.device)

        total_nll = 0.0
        total_cnt = 0

        # initially all masked, so prev_mask = True everywhere
        prev_mask = (x_t == mask_id)

        with torch.no_grad():
            for step in range(self.steps):
                mask_now = (x_t == mask_id)

                out = model(input_ids=x_t, is_causal=False)
                logits = out.logits

                # eval uses temp=1
                logp = F.log_softmax(logits, dim=-1)

                # DIRECT GATHER
                lp_true = logp.gather(2, x0.unsqueeze(-1)).squeeze(-1)

                # newly unmasked this step
                Ft = (~mask_now) & prev_mask

                if Ft.any():
                    nll = -lp_true[Ft].sum()
                    total_nll += nll.item()
                    total_cnt += Ft.sum().item()

                # ---- P2 UPDATE ----
                confidence, x0_pred = logp.exp().max(dim=-1)

                kappa = (step + 1) / self.steps
                num_keep = int(L * kappa)

                # select top confidence to unmask
                _, idx = confidence.sort(dim=1, descending=True)

                new_mask = torch.ones_like(x_t, dtype=torch.bool)
                new_mask.scatter_(1, idx[:, :num_keep], False)

                x_t = torch.where(new_mask, mask_id, x0_pred)
                prev_mask = new_mask.clone()

        if total_cnt == 0:
            return None

        NLL = total_nll / total_cnt
        return float(torch.exp(torch.tensor(NLL)))

    # ============================================================
    #   LM HARNESS: generate_until
    # ============================================================

    @torch.no_grad()
    def generate_until(self, requests: List[Instance], disable_tqdm=False):
        from tqdm import tqdm

        final_results = []
        pbar = tqdm(total=len(requests), disable=disable_tqdm or not self.accelerator.is_main_process)

        for i in range(0, len(requests), self.batch_size):
            batch = requests[i:i+self.batch_size]
            prompts = [r.args[0] for r in batch]

            gen_kwargs = batch[0].args[1] if len(batch[0].args) > 1 else {}
            grouped_outputs = self._generate_batch(prompts, gen_kwargs)
            # grouped_outputs = List[List[str]]  ← correct shape

            # stop-sequence processing
            for cand_list, req in zip(grouped_outputs, batch):
                until = req.args[1].get("until", [])
                cleaned = []
                for t in cand_list:
                    if isinstance(t, list):
                        t = " ".join(map(str, t))
                    for s in until:
                        if s and s in t:
                            t = t.split(s)[0]
                    cleaned.append(t)

                final_results.append(cleaned)

            pbar.update(len(batch))

        pbar.close()

        # compute perplexity on prompts
        if self.accelerator.is_main_process:
            ppl_list = []
            for req in requests:
                ppl = self._compute_progressive_ppl(req.args[0])
                if ppl is not None:
                    ppl_list.append(ppl)

            if ppl_list:
                avg = sum(ppl_list) / len(ppl_list)
                eval_logger.info(f"[Diffusion PPL] avg={avg:.4f} over {len(ppl_list)} prompts")
                print(f"[Diffusion PPL] avg={avg:.4f} over {len(ppl_list)} prompts")

        return final_results

    # diffusion models do NOT support AR likelihood
    def loglikelihood(self, requests):
        raise NotImplementedError("Diffusion models do NOT support autoregressive loglikelihood.")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("Diffusion models do NOT support autoregressive loglikelihood.")

    # necessary LM properties
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
