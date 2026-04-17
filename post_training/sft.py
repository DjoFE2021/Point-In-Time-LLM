# SFT with LoRA — multi-GPU via PyTorch DDP.
# Loads pre-tokenized data from data/sft_tokenized (produced by sft_tokens.py).
#
# Step 1: Tokenize the dataset first:
#     python post_training/sft_tokens.py --max-length 2048
#
# Step 2: Run this training script (8 GPUs):
#     torchrun --nproc_per_node=8 post_training/sft.py

import os
import sys
import math
import time
import torch
import torch.distributed as dist
from collections import OrderedDict
from datasets import load_from_disk
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from peft import LoraConfig, get_peft_model

# Add workspace root to path so we can import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.GPT import GPT
from models.GPTConfig import GPT2_4B  # change to GPT2_1B or GPT2_7B if needed

# -----------------------------
# DDP setup (torchrun sets these env vars)
# -----------------------------
assert torch.cuda.is_available(), "CUDA required for DDP training"
dist.init_process_group(backend="nccl")
ddp_rank = int(os.environ["RANK"])
ddp_local_rank = int(os.environ["LOCAL_RANK"])
ddp_world_size = int(os.environ["WORLD_SIZE"])
device = f"cuda:{ddp_local_rank}"
torch.cuda.set_device(device)
master_process = (ddp_rank == 0)

if master_process:
    print(f"DDP: {ddp_world_size} GPUs, this is rank {ddp_rank}")

# -----------------------------
# Config
# -----------------------------
CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH",
    "./2019-08_step=30931_checkpoint.pt",
)
MODEL_CONFIG = GPT2_4B
TOKENIZED_DATA_DIR = os.environ.get("TOKENIZED_DATA_DIR", "./data/sft_tokenized")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./model-sft")
NUM_VOCAB = 50304

# Extract checkpoint name from path (e.g. "2021-12_step=42325" from "2021-12_step=42325_checkpoint.pt")
_ckpt_basename = os.path.basename(CHECKPOINT_PATH)
CHECKPOINT_NAME = _ckpt_basename.replace('_checkpoint.pt', '').replace('.pt', '')

# Hyperparameters (tuned for H200 141GB VRAM)
# RTX 5090 (32GB): BATCH_SIZE=2, GRAD_ACCUM_STEPS=2, LEARNING_RATE=5e-5
NUM_EPOCHS = 1
LEARNING_RATE = 2e-4         # scaled from 5e-5 (linear scaling with larger batch)
WARMUP_STEPS = 100
BATCH_SIZE = 10              # per-GPU micro-batch (halved to reduce VRAM)
# Effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS * world_size
GRAD_ACCUM_STEPS = 2         # doubled to keep same effective batch size
LOGGING_STEPS = 10
SAVE_STEPS = 10_000
SAVE_TOTAL_LIMIT = 3
MAX_GRAD_NORM = 1.0
MAX_LENGTH = 2048
USE_BF16 = True

# LoRA config
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["c_q", "c_k", "c_v", "c_proj", "c_fc"]

# Performance
torch.set_float32_matmul_precision("high")  # TF32 for matmuls
torch.backends.cudnn.benchmark = True

# -----------------------------
# 1) Build model & load checkpoint
# -----------------------------
if master_process:
    print(f"Building model: {MODEL_CONFIG.__name__}")
model = GPT(MODEL_CONFIG(vocab_size=NUM_VOCAB))

if master_process:
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", mmap=True)

if isinstance(ckpt, dict) and "model" in ckpt:
    raw_sd = ckpt["model"]
else:
    raw_sd = ckpt

# Strip DDP / torch.compile name mangling
new_sd = OrderedDict()
for k, v in raw_sd.items():
    name = k.replace("module._orig_mod.", "")
    name = name.replace("module.", "")
    name = name.replace("_orig_mod.", "")
    new_sd[name] = v

model.load_state_dict(new_sd)
del ckpt, raw_sd, new_sd
torch.cuda.empty_cache()

# -----------------------------
# 1b) Apply LoRA
# -----------------------------
if master_process:
    print(f"\n🔗 Applying LoRA (r={LORA_R}, alpha={LORA_ALPHA})...")
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    target_modules=LORA_TARGET_MODULES,
)
model = get_peft_model(model, lora_config)
if master_process:
    model.print_trainable_parameters()

# Gradient checkpointing: disabled on H200 (141GB VRAM is abundant, saves ~15% time)
# To re-enable for smaller GPUs: model.base_model.model.gradient_checkpointing = True
model.base_model.model.gradient_checkpointing = True
if master_process:
    print("  ⚡ Gradient checkpointing disabled (H200 has ample VRAM)")

model.to(device=device, dtype=torch.bfloat16 if USE_BF16 else torch.float32)
# NOTE: torch.compile not used with LoRA/peft to avoid graph break issues

# Wrap in DDP
model = DDP(model, device_ids=[ddp_local_rank], gradient_as_bucket_view=True)
model.train()
if master_process:
    print(f"Model loaded with LoRA, wrapped in DDP across {ddp_world_size} GPUs")

# -----------------------------
# 2) Load pre-tokenized dataset (memory-mapped, not loaded into RAM)
# -----------------------------
if master_process:
    print(f"Loading pre-tokenized dataset from: {TOKENIZED_DATA_DIR}")
dataset = load_from_disk(TOKENIZED_DATA_DIR)
if master_process:
    print(f"  Examples: {len(dataset)}, Columns: {dataset.column_names}")
    print("  Using memory-mapped Arrow files (lazy loading)")


def collate_fn(batch):
    """Pad input_ids and labels to the longest sequence in the batch."""
    pad_id = 50256  # GPT-2 EOS token used as pad
    input_ids = [torch.tensor(ex["input_ids"][:MAX_LENGTH], dtype=torch.long) for ex in batch]
    labels = [torch.tensor(ex["labels"][:MAX_LENGTH], dtype=torch.long) for ex in batch]

    max_len = max(len(ids) for ids in input_ids)

    padded_input_ids = []
    padded_labels = []
    for ids, lbl in zip(input_ids, labels):
        pad_len = max_len - len(ids)
        padded_input_ids.append(torch.cat([ids, ids.new_full((pad_len,), pad_id)]))
        padded_labels.append(torch.cat([lbl, lbl.new_full((pad_len,), -100)]))

    return {
        "input_ids": torch.stack(padded_input_ids),
        "labels": torch.stack(padded_labels),
    }


sampler = DistributedSampler(dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True)
dataloader = DataLoader(
    dataset, batch_size=BATCH_SIZE, sampler=sampler,
    collate_fn=collate_fn, num_workers=0, pin_memory=True,
)

# -----------------------------
# 3) Optimizer & scheduler
# -----------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

total_steps = (len(dataloader) // GRAD_ACCUM_STEPS) * NUM_EPOCHS
if master_process:
    print(f"Total optimizer steps: {total_steps}, Warmup: {WARMUP_STEPS}")
    print(f"Effective batch size: {BATCH_SIZE} * {GRAD_ACCUM_STEPS} * {ddp_world_size} = {BATCH_SIZE * GRAD_ACCUM_STEPS * ddp_world_size}")


def lr_lambda(current_step):
    """Linear warmup then linear decay."""
    if current_step < WARMUP_STEPS:
        return current_step / max(1, WARMUP_STEPS)
    return max(0.0, (total_steps - current_step) / max(1, total_steps - WARMUP_STEPS))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# -----------------------------
# 4) Training loop
# -----------------------------
if master_process:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
saved_checkpoints = []  # track for save_total_limit

global_step = 0
accum_loss = 0.0
log_start = time.time()


def save_adapter_with_metadata(model_module, save_dir, stage="sft"):
    """Save LoRA adapter and append training hyperparameters to adapter_config.json."""
    import json
    model_module.save_pretrained(save_dir)
    config_path = os.path.join(save_dir, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = {}
    config["training_hyperparameters"] = {
        "stage": stage,
        "base_checkpoint": CHECKPOINT_PATH,
        "checkpoint_name": CHECKPOINT_NAME,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "num_epochs": NUM_EPOCHS,
        "warmup_steps": WARMUP_STEPS,
        "max_grad_norm": MAX_GRAD_NORM,
        "max_length": MAX_LENGTH,
        "use_bf16": USE_BF16,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "lora_target_modules": LORA_TARGET_MODULES,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


if master_process:
    print(f"\n{'='*60}")
    print(f"Starting LoRA SFT training: {NUM_EPOCHS} epochs, effective batch size = {BATCH_SIZE * GRAD_ACCUM_STEPS * ddp_world_size}")
    print(f"{'='*60}\n")

for epoch in range(NUM_EPOCHS):
    sampler.set_epoch(epoch)  # ensure proper shuffling across epochs
    for micro_step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Only sync gradients on the last accumulation step
        sync_grads = ((micro_step + 1) % GRAD_ACCUM_STEPS == 0)
        ctx = model.no_sync if not sync_grads else lambda: torch.enable_grad()

        with ctx():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=USE_BF16):
                logits, loss = model(input_ids, targets=labels)
                loss = loss / GRAD_ACCUM_STEPS  # scale for accumulation
            loss.backward()

        accum_loss += loss.item()

        if sync_grads:
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            # Logging (rank 0 only)
            if master_process and global_step % LOGGING_STEPS == 0:
                elapsed = time.time() - log_start
                avg_loss = accum_loss / LOGGING_STEPS
                lr = scheduler.get_last_lr()[0]
                print(
                    f"  epoch {epoch+1}/{NUM_EPOCHS} | step {global_step}/{total_steps} | "
                    f"loss {avg_loss:.4f} | lr {lr:.2e} | {elapsed:.1f}s"
                )
                accum_loss = 0.0
                log_start = time.time()

            # Save checkpoint (rank 0 only) — save LoRA adapter weights only
            if master_process and global_step % SAVE_STEPS == 0:
                ckpt_dir = os.path.join(OUTPUT_DIR, CHECKPOINT_NAME, f"step={global_step}")
                save_adapter_with_metadata(model.module, ckpt_dir, stage="sft")
                print(f"  💾 Saved LoRA adapter: {ckpt_dir}")

                saved_checkpoints.append(ckpt_dir)
                # Enforce save_total_limit
                while len(saved_checkpoints) > SAVE_TOTAL_LIMIT:
                    old = saved_checkpoints.pop(0)
                    if os.path.exists(old):
                        import shutil
                        shutil.rmtree(old)
                        print(f"  🗑️  Removed old checkpoint: {old}")

            # Barrier to keep all ranks in sync after checkpointing
            dist.barrier()

    # End-of-epoch — save final adapter to model-sft/{checkpoint_name}/
    if master_process:
        ckpt_dir = os.path.join(OUTPUT_DIR, CHECKPOINT_NAME)
        save_adapter_with_metadata(model.module, ckpt_dir, stage="sft")
        print(f"\n✅ Saved LoRA adapter to: {ckpt_dir}")
        print(f"   To merge: python post_training/merge_lora.py --adapter {ckpt_dir}")
    dist.barrier()

dist.barrier()
dist.destroy_process_group()
