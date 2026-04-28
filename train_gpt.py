#!/usr/bin/env python3
"""
GPT Training Script with CLI and Monthly Checkpointing.

Usage:
    # Normal training with 1B model
    torchrun --nproc_per_node=8 train_gpt.py --model 1B
    
    # Dry-run with tiny model
    torchrun --nproc_per_node=1 train_gpt.py --dry-run
    
    # Resume from checkpoint
    torchrun --nproc_per_node=8 train_gpt.py --model 1B --resume /path/to/checkpoint.pt
"""
import os
import math
import sys
import argparse

with open(sys.argv[0]) as f:
    code = f.read()  # read the code of this file ASAP, for logging

import uuid
import time

import torch
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP

from models.GPT import GPT
from models.GPTConfig import GPT2_1B, GPT2_4B
from optimizers.MuON import Muon
from optimizers.lr_scheduler import LRScheduler
from dataloaders.DDP import DistributedDataLoader, save_checkpoint
from training.Hyperparams import CSCS_60GPU, CSCS_160GPU_2K

from utils.checkpoint_loading import (
    load_checkpoint_to_model,
    load_checkpoint_to_optimizers,
    load_checkpoint_to_schedulers,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="GPT Training Script")
    
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["1B", "4B"],
        default="1B",
        help="Model size: 1B (~1.5B params), 4B (~4B params)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use tiny model for testing (equivalent to --model 4B)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Directory to save monthly checkpoints (default: checkpoints)"
    )
    parser.add_argument(
        "--config",
        type=str,
        choices=["CSCS_60GPU", "CSCS_160GPU_2K"],
        default="CSCS_160GPU_2K",
        help="Hyperparameter config to use"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override training data directory from config"
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default=None,
        help="Override validation data directory from config"
    )
    parser.add_argument(
        "--total-tokens",
        type=int,
        default=None,
        help="Total tokens for entire training run (for proper LR scheduling across months)"
    )
    
    return parser.parse_args()


def get_model_config(args):
    """Get model configuration based on args."""
    if args.model == "4B":
        return GPT2_4B()
    elif args.model == "1B":
        return GPT2_1B()
    else:
        raise ValueError(f"Unknown model size: {args.model}")


def get_hyperparams(args):
    """Get hyperparameter configuration."""
    if args.config == "CSCS_60GPU":
        return CSCS_60GPU()
    else:
        return CSCS_160GPU_2K()


def save_monthly_checkpoint(
    output_dir: str,
    month: str,
    step: int,
    model,
    optimizers: list,
    schedulers: list,
    code: str,
):
    """Save checkpoint at end of month."""
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint = {
        "month": month,
        "step": step,
        "code": code,
        "model": model.state_dict(),
        "optimizers": [opt.state_dict() for opt in optimizers],
        "schedulers": [sched.state_dict() for sched in schedulers],
    }
    
    filename = f"{month}_step={step}_checkpoint.pt"
    filepath = os.path.join(output_dir, filename)
    torch.save(checkpoint, filepath)
    print(f"✅ Saved monthly checkpoint: {filepath}")
    return filepath


def main():
    args = parse_args()
    
    # Get hyperparameters
    hparams = get_hyperparams(args)
    # Override data directory if specified
    if args.data_dir:
        hparams.input_bin = args.data_dir
    
    # Override validation directory if specified
    if args.val_dir:
        hparams.input_val_bin = args.val_dir
    
    # Override state_dict with resume path and extract start_step from checkpoint
    if args.resume:
        hparams.state_dict = args.resume
        # Load checkpoint to get resume step
        import re
        # Try to extract step from filename (format: *_step=NNNN_*.pt)
        match = re.search(r'step=(\d+)', args.resume)
        if match:
            hparams.start_step = int(match.group(1))
            print(f"✅ Resuming from step {hparams.start_step}")
    
    # Set up DDP (distributed data parallel). torchrun sets this env variable
    assert torch.cuda.is_available()
    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    print(f"using device: {device}")
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.

    # Store per-device batch size and sequence length under convenient variables
    B, T = hparams.device_batch_size, hparams.sequence_length

    # Compute the number of steps to be done in the validation loop
    assert hparams.val_tokens % (B * T * ddp_world_size) == 0
    val_steps = hparams.val_tokens // (B * T * ddp_world_size)

    # Compute the steps of gradient accumulation required to attain the desired global batch size.
    assert hparams.batch_size % (B * ddp_world_size) == 0
    train_accumulation_steps = hparams.batch_size // (B * ddp_world_size)

    # Instantiate the DistributedDataLoader for the training and validation
    # Create callback that uses CLI output-dir
    def on_shard_advance(model, dataloader, optimizer, scheduler):
        save_checkpoint(model, dataloader, optimizer, scheduler, save_dir=args.output_dir)
    
    train_loader = DistributedDataLoader(
        hparams.input_bin, B, T, ddp_rank, ddp_world_size, on_shard_advance, hparams.skip_files
    )
    val_loader = DistributedDataLoader(
        hparams.input_val_bin, B, T, ddp_rank, ddp_world_size, None, None
    )
    if master_process:
        print(f"Model: {args.model} (dry-run: {args.dry_run})")
        print(
            f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files"
        )
        print(f"  Files: {train_loader.files}")
        print(
            f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files"
        )
    x, y = train_loader.next_batch(None, None, None)

    # Instantiate our GPT model and compile it for speed.
    model_config = get_model_config(args)
    model = GPT(model_config)
    if master_process:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,} ({num_params/1e6:.1f}M)")

    # Load the checkpoint to the model if resuming
    if hparams.state_dict is not None:
        if master_process:
            print(f"Loading checkpoint to CPU first (avoids 2x GPU memory): {hparams.state_dict}")
        ckpt = torch.load(hparams.state_dict, map_location='cpu', mmap=True)  # Load to CPU first to avoid 2x GPU memory
        model = load_checkpoint_to_model(ckpt, model)
        del ckpt["model"]
        print("checkpoint loaded in models")

    # NOW WE SEND TO CUDA
    model = model.cuda()

    if hasattr(config, "coordinate_descent_tuning"):
        config.coordinate_descent_tuning = True  # suggested by @Chillee
    model = torch.compile(model)

    # Wrap the model in the DDP container for multi-GPU training
    model = DDP(
        model, device_ids=[ddp_local_rank], 
        bucket_cap_mb=256, 
        # Use gradient_as_bucket_view to reduce memory copying overhead
        gradient_as_bucket_view=True
    )
    raw_model = model.module  # always contains the "raw" unwrapped model
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    # Instantiate the optimizers. We use Muon for the transformers, AdamW for the rest.
    optimizer1 = torch.optim.AdamW(
        raw_model.lm_head.parameters(),
        lr=hparams.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=hparams.weight_decay,
        fused=True,
    )
    optimizer2 = Muon(
        raw_model.transformer.h.parameters(),
        lr=0.1 * hparams.learning_rate,
        momentum=0.95,
        rank=ddp_rank,
        world_size=ddp_world_size,
    )
    optimizers = [optimizer1, optimizer2]

    # Load the checkpoint to the optimizers if resuming
    if hparams.state_dict is not None:
        optimizers = load_checkpoint_to_optimizers(optimizers, ckpt)
        print("checkpoint loaded in optimizers")

    # Compute the number of iterations for THIS file (used for training loop)
    file_iterations = int(
        math.ceil(train_loader.ntok_total / hparams.batch_size / hparams.sequence_length)
    )
    
    # Compute TOTAL iterations for LR scheduling (across all months)
    if args.total_tokens:
        total_iterations = int(
            math.ceil(args.total_tokens / hparams.batch_size / hparams.sequence_length)
        )
        if master_process:
            print(f"LR schedule based on {args.total_tokens:,} total tokens → {total_iterations:,} total steps")
    else:
        # Fallback: just use current file tokens
        total_iterations = file_iterations + hparams.start_step
        if master_process:
            print(f"⚠️  No --total-tokens specified, using current file only ({total_iterations:,} steps)")
    
    warmdown_iters = total_iterations // 2
    num_iterations = file_iterations
    if hparams.state_dict is not None:
        schedulers = load_checkpoint_to_schedulers(
            checkpoint=ckpt,
            optimizers=optimizers,
            scheduler_step=hparams.start_step,
            num_iterations=total_iterations,  # Use total for full training run
            warmdown_iters=warmdown_iters,
            warmup_iters=hparams.warmup_iters,
        )
        print("checkpoint loaded in schedulers")
        # Clean up checkpoint from memory
        del ckpt
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    else:
        # We instantiate our custom learning rate decay scheduler (linear warmup and warmdown)
        lr_scheduler = LRScheduler(total_iterations, warmdown_iters, hparams.warmup_iters)
        schedulers = [
            torch.optim.lr_scheduler.LambdaLR(opt, lr_scheduler.get_lr)
            for opt in optimizers
        ]

    for sched in schedulers:
        print(sched.get_lr())

    print(warmdown_iters)

    # Let's start the Logging
    if master_process:
        run_id = str(uuid.uuid4())
        logdir = "logs/%s/" % run_id
        os.makedirs(logdir, exist_ok=True)
        logfile = "logs/%s.txt" % run_id
        # create the log file
        with open(logfile, "w") as f:
            # begin the log by printing this file (the Python code)
            f.write("=" * 100 + "\n")
            f.write(code)
            f.write("=" * 100 + "\n")
            # log information about the hardware/software environment this is running on
            f.write(
                f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:\n"
            )
            import subprocess
            result = subprocess.run(
                ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            f.write(f"{result.stdout}\n")
            f.write("=" * 100 + "\n")

    # Track current month for monthly checkpointing
    current_month = train_loader.current_file_month() if hasattr(train_loader, 'current_file_month') else None
    
    training_time_ms = 0
    # start the clock
    torch.cuda.synchronize()
    t0 = time.time()
    # begin training
    train_loader.reset()
    
    for step in range(hparams.start_step, hparams.start_step + num_iterations + 1):
        last_step = step == (hparams.start_step + num_iterations)
        
        # Check for month transition and save checkpoint
        if hasattr(train_loader, 'current_file_month'):
            new_month = train_loader.current_file_month()
            if new_month != current_month and current_month is not None and master_process:
                # Month changed - save checkpoint for completed month
                save_monthly_checkpoint(
                    output_dir=args.output_dir,
                    month=current_month,
                    step=step,
                    model=raw_model,
                    optimizers=optimizers,
                    schedulers=schedulers,
                    code=code,
                )
            current_month = new_month
        
        # This effectively ignores timing first 10 steps, which are slower for weird reasons.
        # Use session-relative step count for timing (handles resume correctly)
        session_step = step - hparams.start_step
        if session_step == 10:
            training_time_ms = 0
            t0 = time.time()
        timed_steps = float("nan") if session_step <= 11 else (session_step - 10) + 1

        # once in a while evaluate the validation dataset
        if last_step or (hparams.val_loss_every > 0 and step % hparams.val_loss_every == 0):
            # stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.time() - t0)
            # run validation batches
            model.eval()
            val_loader.reset()
            val_loss = 0.0
            for _ in range(val_steps):
                x_val, y_val = val_loader.next_batch(None, None, None)
                with ctx:
                    _, loss = model(x_val, y_val, return_logits=False)
                    val_loss += loss.detach()
                    del loss
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            val_loss /= val_steps
            # log val loss to console and to logfile
            if master_process:
                print(
                    f"step:{step}/{num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms"
                )
                with open(logfile, "a") as f:
                    f.write(
                        f"step:{step}/{num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms\n"
                    )
            # start the clock again
            torch.cuda.synchronize()
            t0 = time.time()

        if master_process and (
            last_step or (hparams.save_every > 0 and step % hparams.save_every == 0)
        ):
            # stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.time() - t0)
            # save the state of the training process
            log = dict(
                step=step,
                code=code,
                model=raw_model.state_dict(),
                optimizers=[opt.state_dict() for opt in optimizers],
                schedulers=[sched.state_dict() for sched in schedulers],
                dataloader_shard=train_loader.current_shard,
                dataloader_position=train_loader.current_position,
            )
            ckpt_path = os.path.join(args.output_dir, f"step={step}_checkpoint.pt")
            torch.save(log, ckpt_path)
            print(f"✅ Saved checkpoint: {ckpt_path}")
            # start the clock again
            torch.cuda.synchronize()
            t0 = time.time()

        if last_step:
            break

        # --------------- TRAINING SECTION BEGIN -----------------
        torch.cuda.empty_cache()
        model.train()
        for i in range(1, train_accumulation_steps + 1):
            # forward pass
            with ctx:
                _, loss = model(x, y, return_logits=False)
                train_loss = loss.detach()
            # advance the dataset for the next batch
            x, y = train_loader.next_batch(model, optimizers, schedulers)
            # backward pass
            if i < train_accumulation_steps:
                with model.no_sync():  # there's no need to sync gradients every accumulation step
                    loss.backward()
            else:
                loss.backward()  # just sync on the last step
        for p in model.parameters():
            p.grad /= train_accumulation_steps
        # step the optimizers and schedulers
        for opt, sched in zip(optimizers, schedulers):
            opt.step()
            sched.step()

        # null the gradients
        model.zero_grad(set_to_none=True)
        # --------------- TRAINING SECTION END -------------------

        if master_process:
            approx_time = training_time_ms + 1000 * (time.time() - t0)
            print(
                f"step:{step+1}/{num_iterations + hparams.start_step} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms"
            )
            with open(logfile, "a") as f:
                f.write(
                    f"step:{step+1}/{num_iterations + hparams.start_step} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms\n"
                )

    # Save final checkpoint for last month
    if master_process and current_month is not None:
        save_monthly_checkpoint(
            output_dir=args.output_dir,
            month=current_month,
            step=step,
            model=raw_model,
            optimizers=optimizers,
            schedulers=schedulers,
            code=code,
        )

    if master_process:
        print(
            f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB"
        )

    # clean up nice
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
