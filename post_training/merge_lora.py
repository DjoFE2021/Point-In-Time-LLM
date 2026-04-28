"""
Merge a LoRA adapter with the base model and save in eval-compatible format.

Usage:
    python post_training/merge_lora.py --adapter my-model-sft-lora/epoch=1
    python post_training/merge_lora.py --adapter my-model-sft-lora/final --output my-model-sft-lora/final_merged.pt

Output: a single .pt file with {"model": state_dict} format, directly loadable
by eval_hellaswag_multigpu.py and ifeval_test.py.
"""

import argparse
import os
import sys
import torch
from collections import OrderedDict
from peft import PeftModel, LoraConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.GPT import GPT
from models.GPTConfig import GPT2_1B, GPT2_4B

MODEL_CONFIGS = {
    "1B": GPT2_1B,
    "4B": GPT2_4B,
}

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument("--adapter", type=str, required=True,
                        help="Path to LoRA adapter directory (from save_pretrained)")
    parser.add_argument("--base-checkpoint", type=str,
                        default="./2019-08_step=30931_checkpoint.pt",
                        help="Path to base model checkpoint")
    parser.add_argument("--model", type=str, default="4B", choices=["1B", "4B", "7B"],
                        help="Model config")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .pt path (default: <adapter_dir>_merged.pt)")
    args = parser.parse_args()

    output_path = args.output or f"{args.adapter.rstrip('/')}_merged.pt"
    num_vocab = 50304

    # 1) Load base model
    print(f"Loading base model ({args.model})...")
    config = MODEL_CONFIGS[args.model]
    model = GPT(config(vocab_size=num_vocab))

    print(f"Loading base checkpoint: {args.base_checkpoint}")
    ckpt = torch.load(args.base_checkpoint, map_location="cpu", mmap=True)
    raw_sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    new_sd = OrderedDict()
    for k, v in raw_sd.items():
        name = k.replace("module._orig_mod.", "").replace("module.", "").replace("_orig_mod.", "")
        new_sd[name] = v
    model.load_state_dict(new_sd)
    del ckpt, raw_sd, new_sd

    # 2) Load LoRA adapter
    print(f"Loading LoRA adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)

    # 3) Merge
    print("Merging LoRA weights into base model...")
    merged = model.merge_and_unload()

    # 4) Save
    print(f"Saving merged model to: {output_path}")
    torch.save({"model": merged.state_dict()}, output_path)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ Done! ({size_mb:.0f} MB)")

if __name__ == "__main__":
    main()
