"""
Diffusion Logit Collector for Knowledge Distillation

This module collects teacher model logits during diffusion sampling steps.
Supports both live pipeline (on-the-fly) and stored logits (pre-computed) modes.
"""

import torch
from typing import Dict, Optional, Any
from dataclasses import dataclass

from veomni.models.transformers.qwen2.generation_utils import (
    MDMGenerationConfig,
    sample_tokens,
)


@dataclass
class DiffusionLogitCollectionConfig:
    """Configuration for collecting logits during diffusion sampling"""
    num_sampling_steps: int = 200  # Number of diffusion steps
    temperature: float = 0.7  # Temperature for sampling
    top_k: Optional[int] = 200  # Top-k filtering
    top_p: Optional[float] = None  # Top-p (nucleus) filtering
    alg: str = "p2"  # Sampling algorithm: "p2", "entropy", "origin", etc.
    alg_temp: Optional[float] = 0.5  # Algorithm-specific temperature
    eps: float = 1e-3  # Minimum timestep
    collect_every_n_steps: int = 1  # Collect logits every N steps (1 = all steps)
    collect_mask_positions_only: bool = True  # Only collect logits at masked positions
    store_on_cpu: bool = True  # Store logits on CPU to save GPU memory


class DiffusionLogitCollector:
    """
    Collects teacher model logits during diffusion sampling steps.
    
    This is used for knowledge distillation where we want to match
    the student's logits to the teacher's logits at each diffusion step.
    """
    
    def __init__(
        self,
        teacher_model,
        tokenizer,
        config: DiffusionLogitCollectionConfig,
        device: str = "cuda"
    ):
        self.teacher_model = teacher_model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        # Ensure teacher is in eval mode
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    def collect_logits_for_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_intermediate_states: bool = False
    ) -> Dict[str, Any]:
        """
        Collect teacher logits during diffusion sampling for a batch of inputs.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask
            return_intermediate_states: Whether to return intermediate x_t states
            
        Returns:
            Dictionary containing:
                - logits: List of logit tensors [num_collected_steps, batch_size, seq_len, vocab_size]
                - timesteps: List of timestep values for each collected step
                - mask_positions: List of mask position tensors for each step
                - intermediate_states: (optional) List of intermediate x_t states
        """
        batch_size, prompt_len = input_ids.shape
        mask_token_id = self.tokenizer.mask_token_id
        
        if mask_token_id is None:
            raise ValueError("Tokenizer must have a mask_token_id")
        
        # Prepare generation config
        max_length = input_ids.shape[1] + 512  # Default max length, can be adjusted
        gen_config = MDMGenerationConfig(
            mask_token_id=mask_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=max_length,
            steps=self.config.num_sampling_steps,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            alg=self.config.alg,
            alg_temp=self.config.alg_temp,
            eps=self.config.eps,
        )
        
        # Pad input_ids with mask tokens
        pad_len = max_length - prompt_len
        masked_suffix = torch.full(
            (batch_size, pad_len),
            mask_token_id,
            device=self.device,
            dtype=torch.long
        )
        x_t = torch.cat([input_ids, masked_suffix], dim=1)
        
        # Fixed tokens (prompt) should never be remasked
        fix_mask = (x_t != mask_token_id)
        
        # Prepare attention mask
        gen_attention_mask = (
            (x_t != self.tokenizer.pad_token_id).long()
            if self.tokenizer.pad_token_id is not None
            else None
        )
        
        # Timesteps from 1.0 down to eps
        timesteps = torch.linspace(1.0, gen_config.eps, gen_config.steps + 1, device=self.device)
        
        # Storage for collected logits
        collected_logits = []
        collected_timesteps = []
        collected_mask_positions = []
        intermediate_states = [] if return_intermediate_states else None
        
        with torch.no_grad():
            for step in range(gen_config.steps):
                t = timesteps[step]
                s = timesteps[step + 1] if step < gen_config.steps - 1 else gen_config.eps
                
                # Find masked positions
                mask_index = (x_t == mask_token_id)
                if not mask_index.any():
                    break
                
                # Forward pass through teacher model
                outputs = self.teacher_model(
                    input_ids=x_t,
                    attention_mask=gen_attention_mask,
                    is_causal=False  # Bidirectional attention for diffusion
                )
                logits = outputs.logits
                
                # CRITICAL: Shift logits to align with training (as in generation_utils.py)
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
                
                # Collect logits if this is a collection step
                if step % self.config.collect_every_n_steps == 0:
                    if self.config.collect_mask_positions_only:
                        # Only collect logits at masked positions
                        mask_logits = logits[mask_index]  # [num_masked, vocab_size]
                        collected_logits.append(mask_logits.cpu() if self.config.store_on_cpu else mask_logits)
                        collected_mask_positions.append(mask_index.cpu() if self.config.store_on_cpu else mask_index)
                    else:
                        # Collect all logits
                        collected_logits.append(logits.cpu() if self.config.store_on_cpu else logits)
                        collected_mask_positions.append(mask_index.cpu() if self.config.store_on_cpu else mask_index)
                    
                    collected_timesteps.append(t.item())
                
                if return_intermediate_states:
                    intermediate_states.append(x_t.clone().cpu() if self.config.store_on_cpu else x_t.clone())
                
                # Update x_t based on sampling algorithm
                x_t = self._update_x_t(
                    x_t=x_t,
                    logits=logits,
                    mask_index=mask_index,
                    fix_mask=fix_mask,
                    t=t,
                    s=s,
                    step=step,
                    gen_config=gen_config
                )
        
        result = {
            "logits": collected_logits,
            "timesteps": collected_timesteps,
            "mask_positions": collected_mask_positions,
        }
        
        if return_intermediate_states:
            result["intermediate_states"] = intermediate_states
        
        return result
    
    def _update_x_t(
        self,
        x_t: torch.Tensor,
        logits: torch.Tensor,
        mask_index: torch.Tensor,
        fix_mask: torch.Tensor,
        t: float,
        s: float,
        step: int,
        gen_config: MDMGenerationConfig
    ) -> torch.Tensor:
        """
        Update x_t based on the sampling algorithm.
        This mirrors the logic from MDMGenerationMixin._mdm_sample
        """
        alg = gen_config.alg
        temperature = gen_config.temperature
        top_p = gen_config.top_p
        top_k = gen_config.top_k
        alg_temp = gen_config.alg_temp
        mask_token_id = gen_config.mask_token_id
        
        mask_logits = logits[mask_index]
        
        if alg == "p2":
            # P2 sampling algorithm
            kappa_t = (step + 1) / gen_config.steps
            
            # Compute confidence and sampled tokens for entire sequence
            conf_full, x0_full = sample_tokens(
                logits, temperature=temperature, top_p=top_p, top_k=top_k, alg=alg
            )
            
            # Construct full confidence matrix
            full_conf = conf_full.clone()
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
                row_mask = torch.arange(max_k, device=x_t.device).unsqueeze(0) < num_to_mask.unsqueeze(1)
                
                to_mask = torch.zeros_like(x_t, dtype=torch.bool)
                batch_arange = torch.arange(x_t.size(0), device=x_t.device).unsqueeze(1).expand_as(topk_idx)
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
            
        elif alg in ["entropy", "topk_margin", "maskgit_plus"]:
            # Confidence-based sampling
            confidence, x0 = sample_tokens(
                mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, alg=alg
            )
            confidence = confidence.to(mask_logits.dtype)
            
            num_mask_tokens_per_sample = mask_index.sum(dim=1)
            
            if step < gen_config.steps - 1:
                number_transfer_tokens_per_sample = (
                    num_mask_tokens_per_sample.float() * (1 - s / t)
                ).long()
            else:
                number_transfer_tokens_per_sample = num_mask_tokens_per_sample
            
            # Build full confidence matrix
            full_confidence = torch.full_like(x_t, -torch.inf, device=self.device, dtype=logits.dtype)
            full_confidence[mask_index] = confidence
            
            max_transfer_tokens = number_transfer_tokens_per_sample.max().item()
            
            if max_transfer_tokens > 0:
                if alg_temp is None or alg_temp == 0:
                    _, all_transfer_indices = torch.topk(full_confidence, max_transfer_tokens, dim=1)
                else:
                    scaled_logits = full_confidence / alg_temp
                    uniform = torch.rand_like(scaled_logits).clamp_(min=1e-20, max=1 - 1e-20)
                    gumbel_noise = -torch.log(-torch.log(uniform))
                    scores = scaled_logits + gumbel_noise
                    _, all_transfer_indices = torch.topk(scores, max_transfer_tokens, dim=1)
                
                batch_size = x_t.size(0)
                valid_mask = (
                    torch.arange(max_transfer_tokens, device=x_t.device).unsqueeze(0)
                    < number_transfer_tokens_per_sample.unsqueeze(1)
                )
                
                valid_transfer_indices = all_transfer_indices[valid_mask]
                valid_batch_indices = (
                    torch.arange(batch_size, device=x_t.device)
                    .unsqueeze(1)
                    .expand_as(all_transfer_indices)[valid_mask]
                )
                
                x_ = torch.zeros_like(x_t, device=self.device, dtype=torch.long) + mask_token_id
                x_[mask_index] = x0.clone()
                
                x_t[valid_batch_indices, valid_transfer_indices] = x_[valid_batch_indices, valid_transfer_indices]
        
        else:
            raise NotImplementedError(f"Algorithm {alg} not implemented in collector")
        
        return x_t


def deepcopy_teacher_weights_to_student(
    teacher_model,
    student_model,
    strict: bool = False
) -> None:
    """
    Deep copy teacher model weights to student model.
    
    This is useful when student has the same architecture as teacher
    but we want to start from teacher's weights.
    
    Args:
        teacher_model: Teacher model (source)
        student_model: Student model (target)
        strict: Whether to strictly match all parameter names
    """
    teacher_state_dict = teacher_model.state_dict()
    student_state_dict = student_model.state_dict()
    
    # Filter to only copy parameters that exist in both models
    copied_params = 0
    skipped_params = 0
    
    for name, param in teacher_state_dict.items():
        if name in student_state_dict:
            if param.shape == student_state_dict[name].shape:
                student_state_dict[name].data.copy_(param.data)
                copied_params += 1
            else:
                skipped_params += 1
                print(f"Skipping {name}: shape mismatch {param.shape} vs {student_state_dict[name].shape}")
        else:
            if strict:
                raise ValueError(f"Parameter {name} not found in student model")
            skipped_params += 1
    
    student_model.load_state_dict(student_state_dict, strict=False)
    print(f"Copied {copied_params} parameters, skipped {skipped_params} parameters")

