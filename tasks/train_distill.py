"""
Knowledge Distillation Training Script for Open-dLLM

This script extends train_torch.py to support knowledge distillation:
- Teacher model: Larger pre-trained model (e.g., Qwen2.5-Coder-7B)
- Student model: Smaller model to be trained (e.g., Qwen2.5-Coder-0.5B)
- Distillation loss: KL divergence between teacher and student logits
- Combined loss: student_loss + alpha * distillation_loss
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from tqdm import trange

from veomni.checkpoint import build_checkpointer, ckpt_to_state_dict
from veomni.data import (
    build_chat_template,
    build_dataloader,
    build_iterative_dataset,
    build_mapping_dataset,
)
from veomni.data.data_transform import process_pretrain_example, process_sft_example
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_foundation_model, build_tokenizer, save_model_assets, save_model_weights
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.utils import helper
from veomni.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args, save_args
from veomni.utils.dist_utils import all_reduce


logger = helper.create_logger(__name__)


@dataclass
class DistillationArguments:
    """Additional arguments for distillation training"""
    teacher_model_path: str = field(
        metadata={"help": "Path to teacher model (larger pre-trained model)"}
    )
    teacher_config_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to teacher model config (if different from model_path)"}
    )
    distillation_alpha: float = field(
        default=0.5,
        metadata={"help": "Weight for distillation loss: total_loss = (1-alpha)*cross_entropy_student_loss + alpha*kl_divergence_distill_loss"}
    )
    temperature: float = field(
        default=4.0,
        metadata={"help": "Temperature for softmax in distillation (higher = softer probabilities)"}
    )
    freeze_teacher: bool = field(
        default=True,
        metadata={"help": "Whether to freeze teacher model (recommended: True)"}
    )
    teacher_use_cache: bool = field(
        default=True,
        metadata={"help": "Whether teacher uses KV cache (faster but more memory)"}
    )


@dataclass
class Arguments:
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "DataArguments" = field(default_factory=DataArguments)
    train: "TrainingArguments" = field(default_factory=TrainingArguments)
    distill: "DistillationArguments" = field(default_factory=DistillationArguments)


def compute_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 4.0,
    labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute KL divergence loss for knowledge distillation.
    
    Args:
        student_logits: Student model logits [batch_size, seq_len, vocab_size]
        teacher_logits: Teacher model logits [batch_size, seq_len, vocab_size]
        temperature: Temperature for softmax
        labels: Optional labels to mask loss (only compute on non-ignored tokens)
    
    Returns:
        KL divergence loss
    """
    # Apply temperature scaling
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    
    # Compute KL divergence: KL(P_teacher || P_student)
    # KL = sum(P_teacher * log(P_teacher / P_student))
    kl_div = F.kl_div(
        student_log_probs.view(-1, student_log_probs.size(-1)),
        teacher_probs.view(-1, teacher_probs.size(-1)),
        reduction='none',
        log_target=False
    ).sum(dim=-1)
    
    # Reshape back to [batch_size, seq_len]
    kl_div = kl_div.view(student_logits.shape[0], student_logits.shape[1])
    
    # Mask out ignored tokens if labels provided
    if labels is not None:
        # Labels shape: [batch_size, seq_len] or [batch_size * seq_len]
        if labels.dim() == 1:
            labels = labels.view(student_logits.shape[0], -1)
        
        # IGNORE_INDEX is typically -100
        from veomni.data.constants import IGNORE_INDEX
        mask = (labels != IGNORE_INDEX).float()
        kl_div = kl_div * mask
        # Average over non-ignored tokens
        return kl_div.sum() / (mask.sum() + 1e-8) * (temperature ** 2)
    else:
        # Average over all tokens
        return kl_div.mean() * (temperature ** 2)


def main():
    args = parse_args(Arguments)
    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))
    torch.cuda.set_device(f"cuda:{args.train.local_rank}")
    dist.init_process_group(backend="nccl")
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)
    if args.train.local_rank == 0:
        helper.enable_third_party_logging()

    if args.train.global_rank == 0:
        save_args(args, args.train.output_dir)

    Checkpointer = build_checkpointer(dist_backend=args.train.data_parallel_mode, ckpt_manager=args.train.ckpt_manager)

    init_parallel_state(
        dp_size=args.train.data_parallel_size,
        tp_size=args.train.tensor_parallel_size,
        ep_size=args.train.expert_parallel_size,
        pp_size=args.train.pipeline_parallel_size,
        cp_size=args.train.context_parallel_size,
        ulysses_size=args.train.ulysses_parallel_size,
        dp_mode=args.train.data_parallel_mode,
    )

    logger.info_rank0("Prepare data")
    tokenizer = build_tokenizer(args.model.tokenizer_path)
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "<M>"})
    print(f'tokenizer.mask_token_id: {tokenizer.mask_token_id}')
    
    if args.data.data_type == "plaintext":
        transform = partial(
            process_pretrain_example,
            tokenizer=tokenizer,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
        )
    elif args.data.data_type == "conversation":
        chat_template = build_chat_template(args.data.chat_template, tokenizer)
        transform = partial(
            process_sft_example,
            chat_template=chat_template,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
        )
    else:
        raise NotImplementedError(f"Unsupported data type: {args.data.data_type}.")

    if args.data.dataloader_type == "native":
        if args.data.datasets_type == "iterable":
            logger.info_rank0("Start building iterative dataset")
            train_dataset = build_iterative_dataset(args.data.train_path, transform=transform, seed=args.train.seed)
            args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size)
        elif args.data.datasets_type == "mapping":
            logger.info_rank0("Start building mapping dataset")
            train_dataset = build_mapping_dataset(args.data.train_path, transform=transform)
            args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size, len(train_dataset))

        train_dataloader = build_dataloader(
            dataset=train_dataset,
            micro_batch_size=args.train.micro_batch_size,
            global_batch_size=args.train.global_batch_size,
            dataloader_batch_size=args.train.dataloader_batch_size,
            seed=args.train.seed,
            max_seq_len=args.data.max_seq_len,
            train_steps=args.train.train_steps,
            rmpad=args.train.rmpad,
            rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
            enable_masking=args.train.enable_masking,
            mask_token_id=tokenizer.mask_token_id,
            bsz_warmup_ratio=args.train.bsz_warmup_ratio,
            bsz_warmup_init_mbtoken=args.train.bsz_warmup_init_mbtoken,
            dyn_bsz_margin=args.train.dyn_bsz_margin,
            dyn_bsz_buffer_size=args.train.dyn_bsz_buffer_size,
            num_workers=args.data.num_workers,
            drop_last=args.data.drop_last,
            pin_memory=args.data.pin_memory,
            prefetch_factor=args.data.prefetch_factor,
        )
    else:
        raise NotImplementedError(f"Unsupported dataloader type: {args.data.dataloader_type}.")

    logger.info_rank0("Prepare student model")
    # Student model (smaller, to be trained)
    student_model = build_foundation_model(
        config_path=args.model.config_path,
        weights_path=args.model.model_path,
        torch_dtype="float32" if args.train.enable_mixed_precision else "bfloat16",
        attn_implementation=args.model.attn_implementation,
        moe_implementation=args.model.moe_implementation,
        init_device=args.train.init_device,
    )
    student_config = student_model.config
    helper.print_device_mem_info("VRAM usage after building student model")

    logger.info_rank0("Prepare teacher model")
    # Teacher model (larger, pre-trained, typically frozen)
    teacher_config_path = args.distill.teacher_config_path or args.distill.teacher_model_path
    teacher_model = build_foundation_model(
        config_path=teacher_config_path,
        weights_path=args.distill.teacher_model_path,
        torch_dtype="bfloat16",  # Teacher can use bfloat16 to save memory
        attn_implementation=args.model.attn_implementation,
        init_device=args.train.init_device,
    )
    
    if args.distill.freeze_teacher:
        logger.info_rank0("Freezing teacher model parameters")
        for param in teacher_model.parameters():
            param.requires_grad = False
        teacher_model.eval()
    else:
        logger.warning_rank0("Teacher model is NOT frozen - this will use more memory!")
    
    helper.print_device_mem_info("VRAM usage after building teacher model")

    # Parallelize student model (teacher typically doesn't need parallelization if frozen)
    get_optimizer_pre_hook = getattr(student_model, "get_optimizer_pre_hook", None)
    student_model = build_parallelize_model(
        student_model,
        init_device=args.train.init_device,
        weights_path=args.model.model_path,
        enable_full_shard=args.train.enable_full_shard,
        enable_mixed_precision=args.train.enable_mixed_precision,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        basic_modules=student_model._no_split_modules + args.model.basic_modules,
        enable_reentrant=args.train.enable_reentrant,
        enable_forward_prefetch=args.train.enable_forward_prefetch,
    )

    optimizer = build_optimizer(
        student_model,
        lr=args.train.lr,
        weight_decay=args.train.weight_decay,
        fused=True,
        optimizer_type=args.train.optimizer,
    )
    if get_optimizer_pre_hook is not None:
        optimizer_pre_hook = get_optimizer_pre_hook(student_model, student_config, args.train.data_parallel_mode)
        optimizer.register_step_pre_hook(optimizer_pre_hook)

    lr_scheduler = build_lr_scheduler(
        optimizer,
        train_steps=args.train.train_steps * args.train.num_train_epochs,
        lr=args.train.lr,
        lr_min=args.train.lr_min,
        lr_decay_style=args.train.lr_decay_style,
        lr_decay_ratio=args.train.lr_decay_ratio,
        lr_warmup_ratio=args.train.lr_warmup_ratio,
        lr_start=args.train.lr_start,
    )

    if args.train.global_rank == 0:
        if args.train.use_wandb:
            wandb.init(
                project=args.train.wandb_project,
                name=args.train.wandb_name,
                config={**vars(args.model), **vars(args.data), **vars(args.train), **vars(args.distill)},
            )

    # Build activation offloading contexts
    model_fwd_context, model_bwd_context = build_activation_offloading_context(
        args.train.enable_activation_offload, args.train.enable_gradient_checkpointing, args.train.activation_gpu_limit
    )
    
    student_model.train()
    if not args.distill.freeze_teacher:
        teacher_model.train()
    
    environ_meter = helper.EnvironMeter(
        config=student_config,
        global_batch_size=args.train.global_batch_size,
        rmpad=args.train.rmpad,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
        empty_cache_steps=args.train.empty_cache_steps,
    )

    # Load checkpoint if resuming
    start_epoch, start_step, global_step = 0, 0, 0
    save_checkpoint_path = None
    if args.train.load_checkpoint_path:
        # ... (checkpoint loading code similar to train_torch.py)
        pass

    model_assets = [student_config, tokenizer if args.data.data_type == "plaintext" else chat_template]
    if args.train.global_rank == 0:
        save_model_assets(args.train.model_assets_dir, model_assets)

    logger.info_rank0(
        f"rank{args.train.local_rank} Start training, train_steps: {args.train.train_steps}, epochs: {args.train.num_train_epochs}"
    )
    
    # ========== MAIN TRAINING LOOP ==========
    for epoch in range(start_epoch, args.train.num_train_epochs):
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)

        data_loader_tqdm = trange(
            args.train.train_steps,
            desc=f"Epoch {epoch + 1}/{args.train.num_train_epochs}",
            total=args.train.train_steps,
            initial=start_step,
            disable=args.train.local_rank != 0,
        )
        data_iterator = iter(train_dataloader)
        
        for _ in range(start_step, args.train.train_steps):
            global_step += 1

            try:
                micro_batches: List[Dict[str, Any]] = next(data_iterator)
            except StopIteration:
                logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.drop_last}")
                break

            if global_step == 1:
                helper.print_example(example=micro_batches[0], rank=args.train.local_rank)

            total_loss = 0
            total_student_loss = 0
            total_distill_loss = 0
            torch.cuda.synchronize()
            start_time = time.time()
            
            for micro_batch in micro_batches:
                environ_meter.add(micro_batch)

                micro_batch = {
                    k: v.cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v 
                    for k, v in micro_batch.items()
                }
                
                labels = micro_batch.get("labels")
                
                # ========== STUDENT FORWARD PASS ==========
                with model_fwd_context:
                    student_outputs = student_model(**micro_batch, use_cache=False)
                    student_loss = student_outputs.loss.mean() / len(micro_batches)
                    student_logits = student_outputs.logits  # [batch, seq_len, vocab_size]

                # ========== TEACHER FORWARD PASS ==========
                with torch.no_grad():  # Teacher doesn't need gradients
                    teacher_outputs = teacher_model(**micro_batch, use_cache=args.distill.teacher_use_cache)
                    teacher_logits = teacher_outputs.logits  # [batch, seq_len, vocab_size]
                
                # ========== DISTILLATION LOSS ==========
                distill_loss = compute_distillation_loss(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    temperature=args.distill.temperature,
                    labels=labels,
                ) / len(micro_batches)
                
                # ========== COMBINED LOSS ==========
                # total_loss = (1 - alpha) * student_loss + alpha * distill_loss
                combined_loss = (
                    (1 - args.distill.distillation_alpha) * student_loss +
                    args.distill.distillation_alpha * distill_loss
                )

                # ========== BACKWARD PASS ==========
                with model_bwd_context:
                    combined_loss.backward()

                total_loss += combined_loss.item()
                total_student_loss += student_loss.item()
                total_distill_loss += distill_loss.item()
                del micro_batch, student_outputs, teacher_outputs

            # Gradient clipping
            if args.train.data_parallel_mode == "fsdp1":
                grad_norm = student_model.clip_grad_norm_(args.train.max_grad_norm).item()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.train.max_grad_norm, foreach=True)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor().item()

            # Collect metrics across data parallel group
            total_loss, total_student_loss, total_distill_loss, grad_norm = all_reduce(
                (total_loss, total_student_loss, total_distill_loss, grad_norm), 
                group=get_parallel_state().fsdp_group
            )
            torch.cuda.synchronize()
            delta_time = time.time() - start_time
            lr = max(lr_scheduler.get_last_lr())
            train_metrics = environ_meter.step(delta_time, global_step=global_step)

            data_loader_tqdm.set_postfix_str(
                f"loss: {total_loss:.2f}, student: {total_student_loss:.2f}, "
                f"distill: {total_distill_loss:.2f}, grad_norm: {grad_norm:.2f}, lr: {lr:.2e}"
            )
            data_loader_tqdm.update()

            if args.train.global_rank == 0:
                if args.train.use_wandb:
                    train_metrics.update({
                        "training/loss": total_loss,
                        "training/student_loss": total_student_loss,
                        "training/distill_loss": total_distill_loss,
                        "training/grad_norm": grad_norm,
                        "training/lr": lr,
                    })
                    wandb.log(train_metrics, step=global_step)

            # Save checkpoint
            if args.train.save_steps and global_step % args.train.save_steps == 0:
                helper.empty_cache()
                save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
                state = {
                    "model": student_model,  # Only save student model
                    "optimizer": optimizer,
                    "extra_state": {
                        "global_step": global_step,
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "train_dataloader": train_dataloader.state_dict(),
                        "environ_meter": environ_meter.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                    },
                }
                Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
                if args.train.global_rank == 0 and args.train.save_hf_weights and save_checkpoint_path is not None:
                    hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
                    model_state_dict = ckpt_to_state_dict(
                        save_checkpoint_path=save_checkpoint_path,
                        output_dir=args.train.output_dir,
                        ckpt_manager=args.train.ckpt_manager,
                    )
                    save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
                    logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")
                dist.barrier()
                logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")

        data_loader_tqdm.close()
        start_step = 0
        helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")

    # Final checkpoint
    if args.train.global_rank == 0:
        save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, "final")
        state = {
            "model": student_model,
            "optimizer": optimizer,
            "extra_state": {
                "global_step": global_step,
                "lr_scheduler": lr_scheduler.state_dict(),
                "train_dataloader": train_dataloader.state_dict(),
                "environ_meter": environ_meter.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
            },
        }
        Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
        if args.train.save_hf_weights:
            hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
            model_state_dict = ckpt_to_state_dict(
                save_checkpoint_path=save_checkpoint_path,
                output_dir=args.train.output_dir,
                ckpt_manager=args.train.ckpt_manager,
            )
            save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
            logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")


if __name__ == "__main__":
    main()

