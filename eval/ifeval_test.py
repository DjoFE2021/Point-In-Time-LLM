"""
IFEval: Instruction-Following Evaluation (multi-GPU prompt sharding).

Evaluates how well a model follows verifiable constraints
(word counts, formatting, keyword inclusion, etc.) — no LLM judge needed.

Uses 541 prompts from google/IFEval with programmatic constraint checking
via lm_eval's IFEval implementation.

Usage:
    # Multi-GPU (shard prompts across 4 GPUs):
    torchrun --nproc_per_node=4 eval/ifeval_test.py --checkpoint /path/to/checkpoint.pt --model 4B

    # Single GPU still works:
    python eval/ifeval_test.py --checkpoint /path/to/checkpoint.pt --model 4B

    # Evaluate a HuggingFace model
    torchrun --nproc_per_node=4 eval/ifeval_test.py --candidate Qwen/Qwen1.5-1.8B-Chat

    # Load pre-computed outputs (skip generation)
    python eval/ifeval_test.py --outputs ifeval_outputs/model_outputs.json

    # Quick debug
    torchrun --nproc_per_node=4 eval/ifeval_test.py --candidate Qwen/Qwen1.5-1.8B-Chat --limit 10
"""

import argparse
import json
import os
from collections import OrderedDict
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import GPT2TokenizerFast, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from lm_eval.tasks.ifeval.utils import (
    InputExample,
    test_instruction_following_strict,
    test_instruction_following_loose,
    agg_inst_level_acc,
)

# Add project root so we can import models
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.GPT import GPT
from models.GPTConfig import GPT2_1B, GPT2_4B

MODEL_CONFIGS = {
    "1B": GPT2_1B,
    "4B": GPT2_4B,
}


# ------------------------------------------------------------------
# DDP helpers
# ------------------------------------------------------------------
def setup_distributed():
    """Initialize distributed if launched via torchrun, else single-GPU."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        rank, local_rank, world_size = 0, 0, 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return rank, local_rank, world_size, device


def gather_strings(local_list, world_size):
    """Gather a list of strings from all ranks to rank 0."""
    if world_size == 1:
        return local_list
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_list)
    return [item for sublist in gathered for item in sublist]


# ------------------------------------------------------------------
# Checkpoint loading (same pattern as alpaca.py)
# ------------------------------------------------------------------
def load_custom_checkpoint(ckpt_path: str, model_size: str, device: str = "cuda"):
    """Load a training checkpoint into a fresh GPT model."""
    num_vocab = 50304
    config_class = MODEL_CONFIGS[model_size]
    model = GPT(config_class(vocab_size=num_vocab))

    print(f"[{device}] Loading custom checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True)

    if isinstance(ckpt, dict) and "model" in ckpt:
        raw_sd = ckpt["model"]
        step = ckpt.get("step", "?")
        print(f"  Checkpoint step: {step}")
    else:
        raw_sd = ckpt

    new_sd = OrderedDict()
    for k, v in raw_sd.items():
        name = k.replace("module._orig_mod.", "")
        name = name.replace("module.", "")
        name = name.replace("_orig_mod.", "")
        new_sd[name] = v

    model.load_state_dict(new_sd)
    del ckpt, raw_sd, new_sd
    torch.cuda.empty_cache()

    model.to(device)
    model.eval()
    print(f"  [{device}] Custom model loaded ✅")
    return model


# ------------------------------------------------------------------
# Generation: custom GPT model (autoregressive)
# ------------------------------------------------------------------
@torch.inference_mode()
def generate_custom(model, tokenizer, prompts, max_new_tokens=512,
                    temperature=0.7, top_p=0.9, device="cuda", start_idx=0,
                    label="Custom"):
    """Generate text using the custom GPT model."""
    outputs = []
    total = start_idx + len(prompts)
    for i, prompt in enumerate(prompts):
        input_ids = tokenizer.encode(prompt)
        generated = list(input_ids)

        for _ in range(max_new_tokens):
            inp = torch.tensor([generated], dtype=torch.long, device=device)
            logits, _ = model(inp, targets=None, return_logits=True)
            next_logits = logits[0, -1, :]

            if temperature > 0:
                next_logits = next_logits / temperature
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                next_logits[sorted_indices[sorted_indices_to_remove]] = float("-inf")
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = next_logits.argmax().item()

            if next_token == tokenizer.eos_token_id:
                break
            generated.append(next_token)

        response = tokenizer.decode(generated[len(input_ids):], skip_special_tokens=True)
        outputs.append(response.strip())

        global_i = start_idx + i
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{label} @ {device}] Generated {global_i+1}/{total}")

    return outputs


# ------------------------------------------------------------------
# Generation: HuggingFace model
# ------------------------------------------------------------------
@torch.inference_mode()
def generate_hf(model, tokenizer, prompts, max_new_tokens=512,
                temperature=0.7, top_p=0.9, start_idx=0, label="HF",
                device="cuda"):
    """Generate text using a HuggingFace model."""
    outputs = []
    total = start_idx + len(prompts)
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(gen[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        outputs.append(text.strip())

        global_i = start_idx + i
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{label} @ {device}] Generated {global_i+1}/{total}")

    return outputs


# ------------------------------------------------------------------
# Prompt formatting for instruction-tuned models
# ------------------------------------------------------------------
def to_prompt(instruction: str) -> str:
    return f"### Instruction:\n{instruction.strip()}\n\n### Response:\n"


# ------------------------------------------------------------------
# IFEval scoring
# ------------------------------------------------------------------
def evaluate_ifeval(examples, outputs):
    """
    Evaluate outputs against IFEval constraints.
    Returns dict with 4 metrics: prompt_level_strict, prompt_level_loose,
    inst_level_strict, inst_level_loose.
    """
    prompt_strict = []
    prompt_loose = []
    inst_strict = []
    inst_loose = []

    for ex, output in zip(examples, outputs):
        inp = InputExample(
            key=ex["key"],
            instruction_id_list=ex["instruction_id_list"],
            prompt=ex["prompt"],
            kwargs=ex["kwargs"],
        )

        out_strict = test_instruction_following_strict(inp, output)
        out_loose = test_instruction_following_loose(inp, output)

        prompt_strict.append(out_strict.follow_all_instructions)
        prompt_loose.append(out_loose.follow_all_instructions)
        inst_strict.append(out_strict.follow_instruction_list)
        inst_loose.append(out_loose.follow_instruction_list)

    return {
        "prompt_level_strict": sum(prompt_strict) / len(prompt_strict) * 100,
        "prompt_level_loose": sum(prompt_loose) / len(prompt_loose) * 100,
        "inst_level_strict": agg_inst_level_acc(inst_strict) * 100,
        "inst_level_loose": agg_inst_level_acc(inst_loose) * 100,
        "n_prompts": len(examples),
        "n_instructions": sum(len(s) for s in inst_strict),
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="IFEval: Instruction-Following Evaluation (no LLM judge)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to your .pt checkpoint file")
    parser.add_argument("--model", type=str, default="4B", choices=["1B", "4B"],
                        help="Model config for your checkpoint (default: 4B)")
    parser.add_argument("--candidate", type=str, default=None,
                        help="HuggingFace model ID (e.g. Qwen/Qwen1.5-1.8B-Chat)")
    parser.add_argument("--outputs", type=str, default=None,
                        help="Path to pre-computed outputs JSON (skips generation)")
    parser.add_argument("--max-new-tokens", type=int, default=1280)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit examples (for quick debugging)")
    parser.add_argument("--output-dir", type=str, default="ifeval_outputs_new")
    args = parser.parse_args()

    rank, local_rank, world_size, device = setup_distributed()
    is_main = (rank == 0)

    # --- Derive model name ---
    if args.checkpoint:
        model_name = os.path.basename(args.checkpoint).replace("_checkpoint.pt", "").replace(".pt", "")
    elif args.candidate:
        model_name = args.candidate.split("/")[-1]
    elif args.outputs:
        model_name = Path(args.outputs).stem.replace("_outputs", "")
    else:
        parser.error("Either --checkpoint, --candidate, or --outputs is required")

    out_dir = Path(args.output_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_outputs.json"

    # --- Load IFEval dataset (all ranks) ---
    if is_main:
        print("Loading IFEval dataset...")
    ds = load_dataset("google/IFEval", split="train")
    examples = list(ds)
    if args.limit:
        examples = examples[:args.limit]
    if is_main:
        print(f"Loaded {len(examples)} IFEval examples\n")

    # --- Generate or load outputs ---
    if args.outputs:
        # Pre-computed outputs: no generation needed, single-process is fine
        src = Path(args.outputs)
        assert src.exists(), f"Outputs not found: {src}"
        with open(src) as f:
            records = json.load(f)
        records = records[:len(examples)]
        gen_texts = [r["output"] for r in records]
        n_have = len(gen_texts)
        n_need = len(examples) - n_have
        if is_main:
            print(f"✅ Loaded {n_have} pre-computed outputs from {src}")
    else:
        # Auto-detect existing outputs for resume
        if out_path.exists():
            with open(out_path) as f:
                records = json.load(f)
            records = records[:len(examples)]
            gen_texts = [r["output"] for r in records]
            n_have = len(gen_texts)
            n_need = len(examples) - n_have
            if is_main:
                print(f"📂 Auto-detected {n_have} existing outputs from {out_path}")
        else:
            gen_texts = []
            n_have = 0
            n_need = len(examples)

    # Generate missing outputs if needed — SHARD ACROSS GPUs
    if n_need > 0:
        if is_main:
            if n_have > 0:
                print(f"   Generating {n_need} missing outputs ({n_have+1}..{len(examples)})...")
            else:
                print(f"Generating {len(examples)} outputs across {world_size} GPU(s)...")

        missing_examples = examples[n_have:]
        missing_prompts = [to_prompt(ex["prompt"]) for ex in missing_examples]

        # Shard prompts: rank gets every world_size-th prompt
        my_prompts = missing_prompts[rank::world_size]
        my_start_idx = n_have + rank  # for progress display

        if is_main:
            print(f"  Each GPU handles ~{len(my_prompts)} prompts")

        if args.candidate:
            cand_id = args.candidate
            if "chrono-gpt" in cand_id.lower():
                from models.ChronoGPTLMInstruct import ChronoGPTInstruct
                print(f"[{device}] Loading ChronoGPT Instruct from {cand_id} ...")
                chrono = ChronoGPTInstruct.from_hub(repo_id=cand_id, device=device, dtype="float16")
                chrono_prompts = [chrono.format_prompt(missing_examples[rank + i * world_size]["prompt"])
                                  for i in range(len(my_prompts))]
                my_texts = chrono.generate_batch(
                    chrono_prompts,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    start_idx=my_start_idx,
                )
                del chrono
            else:
                print(f"[{device}] Loading HuggingFace model: {cand_id}")
                tokenizer = AutoTokenizer.from_pretrained(cand_id, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    cand_id,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True,
                ).to(device)
                model.eval()
                my_texts = generate_hf(
                    model, tokenizer, my_prompts,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    start_idx=my_start_idx,
                    label=cand_id.split("/")[-1],
                    device=device,
                )
                del model
            torch.cuda.empty_cache()

        elif args.checkpoint:
            model = load_custom_checkpoint(args.checkpoint, args.model, device)
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            my_texts = generate_custom(
                model, tokenizer, my_prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device,
                start_idx=my_start_idx,
                label=model_name,
            )
            del model
            torch.cuda.empty_cache()

        else:
            parser.error("Need --checkpoint or --candidate to generate missing outputs")

        # Gather sharded results back in original order
        # Each rank produced outputs for indices [rank, rank+ws, rank+2*ws, ...]
        # We need to interleave them back
        if world_size > 1:
            all_shards = [None] * world_size
            dist.all_gather_object(all_shards, my_texts)
            # Interleave: shard[0][0], shard[1][0], ..., shard[0][1], shard[1][1], ...
            missing_texts = []
            max_shard_len = max(len(s) for s in all_shards)
            for i in range(max_shard_len):
                for shard in all_shards:
                    if i < len(shard):
                        missing_texts.append(shard[i])
        else:
            missing_texts = my_texts

        gen_texts.extend(missing_texts)
        if is_main:
            print(f"   ✅ Now have {len(gen_texts)} total outputs")

    # --- From here on, only rank 0 does scoring and saving ---
    if is_main:
        # --- Save outputs ---
        records = [
            {"key": ex["key"], "prompt": ex["prompt"], "output": gen_texts[i],
             "generator": model_name}
            for i, ex in enumerate(examples)
        ]
        with open(out_path, "w") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"\n✅ Saved outputs to {out_path}")

        # --- Evaluate ---
        print("\n" + "=" * 60)
        print("Evaluating IFEval constraints...")
        print("=" * 60)

        results = evaluate_ifeval(examples, gen_texts)

        # --- Print results ---
        print(f"\n{'─' * 55}")
        print(f"  Model:        {model_name}")
        print(f"  Prompts:      {results['n_prompts']}")
        print(f"  Instructions: {results['n_instructions']}")
        print(f"{'─' * 55}")
        print(f"  📋 Prompt-Level Strict Accuracy:  {results['prompt_level_strict']:5.1f}%")
        print(f"  📋 Prompt-Level Loose  Accuracy:  {results['prompt_level_loose']:5.1f}%")
        print(f"  📝 Inst-Level   Strict Accuracy:  {results['inst_level_strict']:5.1f}%")
        print(f"  📝 Inst-Level   Loose  Accuracy:  {results['inst_level_loose']:5.1f}%")
        print(f"{'─' * 55}")

        # --- Save summary ---
        summary_path = out_dir / f"{model_name}_summary.txt"
        with open(summary_path, "w") as f:
            f.write(f"model: {model_name}\n")
            f.write(f"n_prompts: {results['n_prompts']}\n")
            f.write(f"n_instructions: {results['n_instructions']}\n")
            f.write(f"prompt_level_strict_acc: {results['prompt_level_strict']:.1f}%\n")
            f.write(f"prompt_level_loose_acc: {results['prompt_level_loose']:.1f}%\n")
            f.write(f"inst_level_strict_acc: {results['inst_level_strict']:.1f}%\n")
            f.write(f"inst_level_loose_acc: {results['inst_level_loose']:.1f}%\n")
        print(f"\n📄 Summary saved to {summary_path}")

    # Cleanup distributed
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
