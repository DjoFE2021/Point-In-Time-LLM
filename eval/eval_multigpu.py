"""
Multi-GPU benchmark evaluation script with prompt sharding.
Each GPU evaluates a shard of the examples for the SAME checkpoint,
achieving ~Nx speedup with N GPUs on a single checkpoint.

Usage:
    # HellaSwag (0-shot, default) — 4 GPUs, prompt-sharded:
    torchrun --nproc_per_node=4 eval/eval_hellaswag_multigpu.py \
        --checkpoints ckpt1.pt --model 4B --output-dir results/

    # Multiple checkpoints evaluated sequentially, each prompt-sharded:
    torchrun --nproc_per_node=4 eval/eval_hellaswag_multigpu.py \
        --checkpoints ckpt1.pt ckpt2.pt --model 4B --output-dir results/

    # Single GPU still works (no torchrun needed):
    python eval/eval_hellaswag_multigpu.py \
        --checkpoints ckpt1.pt --model 4B --output-dir results/
"""

import argparse
import csv
import os
import torch
import torch.distributed as dist
from collections import OrderedDict

from models.GPT import GPT
from models.GPTConfig import GPT2_1B, GPT2_4B, GPT2_7B
from transformers import GPT2TokenizerFast
from models.PIT import PIT
from lm_eval import tasks, evaluator

MODEL_CONFIGS = {
    "1B": GPT2_1B,
    "4B": GPT2_4B,
    "7B": GPT2_7B,
}

CONTEXT_LENGTHS = {
    "1B": 1024,
    "4B": 2048,
    "7B": 1024,
}

def test_model(model, limit, batch_size=128, max_length=None, task_name="hellaswag", num_fewshot=0, rank=0, world_size=1):
    """
    Evaluate model on a given lm-eval task.
    
    Supported tasks include:
        - hellaswag: commonsense reasoning
        - gsm8k: grade school math
        - arc_easy, arc_challenge: science reasoning
        - winogrande: commonsense
        - math (hendrycks_math): competition math
    """
    # use HuggingFace GPT-2 tokenizer (must match your vocab)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    if max_length is not None:
        tokenizer.model_max_length = max_length
        
    lm = PIT(model, tokenizer, max_length=max_length, batch_size=batch_size)
    lm._rank = rank
    lm._world_size = world_size
    
    print(f"\nRunning evaluation on {task_name} (batch_size={batch_size}, num_fewshot={num_fewshot})...")
    
    task_dict = tasks.get_task_dict(task_name)
    
    # Set num_fewshot on each task's config
    for task_key in task_dict:
        task_obj = task_dict[task_key]
        if hasattr(task_obj, '_config'):
            task_obj._config.num_fewshot = num_fewshot
        elif hasattr(task_obj, 'config'):
            task_obj.config.num_fewshot = num_fewshot
    
    results = evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=limit
    )
    
    # Common metric keys in order of preference
    metric_keys = ["acc_norm,none", "acc,none", "exact_match,none", "exact_match,flexible-extract"]
    
    def _extract_score(subtask_results):
        """Extract the best metric from a subtask's results dict."""
        for key in metric_keys:
            if key in subtask_results:
                return subtask_results[key]
        # Fallback: return first numeric result
        for key, value in subtask_results.items():
            if isinstance(value, (int, float)):
                return value
        return None
    
    # Single-task case: task_name is directly in results
    if task_name in results["results"]:
        task_results = results["results"][task_name]
        score = _extract_score(task_results)
        if score is not None:
            return score
        raise ValueError(f"Could not find accuracy metric in results: {task_results.keys()}")
    
    # Multi-task case (e.g. "glue" -> individual subtasks "cola", "mnli", "sst2", ...):
    # lm_eval returns subtasks without a group prefix, so collect all results
    subtask_scores = {}
    for key, val in results["results"].items():
        score = _extract_score(val)
        if score is not None:
            subtask_scores[key] = score
    
    if subtask_scores:
        return subtask_scores
    
    raise ValueError(f"No results found for task '{task_name}'. Available: {list(results['results'].keys())}")
        


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


def load_checkpoint(ckpt_path: str, model: torch.nn.Module, device: str) -> torch.nn.Module:
    """Load checkpoint with proper state dict key handling."""
    print(f"[{device}] Loading checkpoint: {ckpt_path}")
    
    # Load to CPU first to avoid doubling GPU memory
    ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True)
    
    # Handle both formats: {"model": state_dict} or raw state_dict
    if isinstance(ckpt, dict) and "model" in ckpt:
        raw_sd = ckpt["model"]
    else:
        raw_sd = ckpt  # Raw state dict (e.g., from merged LoRA model)
    
    # Handle DDP and compiled model name mangling
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
    
    return model


def extract_checkpoint_name(ckpt_path: str) -> str:
    """Extract readable name from checkpoint path."""
    basename = os.path.basename(ckpt_path)
    # Remove _checkpoint.pt suffix
    name = basename.replace("_checkpoint.pt", "")
    return name


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU HellaSwag evaluation")
    parser.add_argument(
        "--checkpoints", 
        type=str, 
        nargs="+", 
        required=True,
        help="List of checkpoint files to evaluate"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="4B",
        choices=["1B", "4B", "7B"],
        help="Model configuration to use"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="results/",
        help="Directory to save results"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None,
        help="Limit number of examples for debugging"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=16,
        help="Batch size for evaluation (reduce for large models)"
    )
    CSR_TASKS = ["boolq", "piqa", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa"]
    parser.add_argument(
        "--tasks", 
        type=str,
        nargs="+",
        default=CSR_TASKS,
        help="Evaluation tasks (default: Common Sense Reasoning suite)"
    )
    parser.add_argument(
        "--num-fewshot", 
        type=int, 
        default=0,
        help="Number of few-shot examples (default: 0)"
    )
    args = parser.parse_args()
    
    rank, local_rank, world_size, device = setup_distributed()
    is_main = (rank == 0)
    
    if is_main:
        print(f"Evaluating with {world_size} GPU(s), prompt-sharded")
    
    # Setup model config
    num_vocab = 50304
    config_class = MODEL_CONFIGS[args.model]
    max_len = CONTEXT_LENGTHS[args.model]
    
    # Create output directory (rank 0 only)
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    
    # Evaluate each checkpoint sequentially, but prompt-sharded across GPUs
    for ckpt_path in args.checkpoints:
        # Every rank loads the same checkpoint onto its own GPU
        model = GPT(config_class(vocab_size=num_vocab))
        model = load_checkpoint(ckpt_path, model, device)
        
        ckpt_name = extract_checkpoint_name(ckpt_path)
        if is_main:
            print(f"\n[Rank {rank}] Evaluating {ckpt_name} on {len(args.tasks)} tasks...")
        
        rows = []
        for task_name in args.tasks:
            if is_main:
                print(f"\n{'─'*50}")
                print(f"  {ckpt_name}: {task_name}")
                print(f"{'─'*50}")
            try:
                result = test_model(
                    model, args.limit,
                    batch_size=args.batch_size,
                    max_length=max_len,
                    task_name=task_name,
                    num_fewshot=args.num_fewshot,
                    rank=rank,
                    world_size=world_size,
                )
                
                if isinstance(result, dict):
                    avg = sum(result.values()) / len(result)
                    rows.append({"task": task_name, "metric": "acc_norm,none", "score": f"{avg:.4f}"})
                    if is_main:
                        print(f"  ✅ {task_name}: {avg:.4f} (avg of {len(result)} subtasks)")
                else:
                    rows.append({"task": task_name, "metric": "acc_norm,none", "score": f"{result:.4f}"})
                    if is_main:
                        print(f"  ✅ {task_name}: {result:.4f}")
                    
            except Exception as e:
                if is_main:
                    print(f"  ❌ {task_name}: {e}")
                    import traceback
                    traceback.print_exc()

        # Save results (rank 0 only)
        if is_main:
            result_file = os.path.join(args.output_dir, f"{ckpt_name}.csv")
            with open(result_file, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["task", "metric", "score"])
                w.writeheader()
                w.writerows(rows)
            
            # Print summary
            print(f"\n{'='*50}")
            print(f"📄 {ckpt_name} — saved to {result_file}")
            print(f"{'='*50}")
            for r in rows:
                print(f"  {r['task']:20s} {r['score']}")
            if rows:
                avg_all = sum(float(r["score"]) for r in rows) / len(rows)
                print(f"  {'AVERAGE':20s} {avg_all:.4f}")
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
        if world_size > 1:
            dist.barrier()
    
    if is_main:
        print(f"\nDone evaluating all checkpoints.")
    
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
