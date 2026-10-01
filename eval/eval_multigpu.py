"""Evaluate pre-trained/SFT GPT checkpoints on lm-eval benchmarks (0-shot Common Sense Reasoning by default).

Runs as a single process on one GPU (the PIT lm-eval wrapper has no multi-process support) and writes
<output-dir>/<ckpt_name>.csv with one (task, metric, score) row per task.
Usage: python3 eval/eval_multigpu.py --model 4B --checkpoints <CKPT_1.pt> [<CKPT_2.pt> ...]
"""

import argparse
import csv
import os
import sys
import torch
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

def test_model(model, limit, batch_size=128, max_length=None, task_name="hellaswag", num_fewshot=0):
    """
    Evaluate model on a given lm-eval task.
    Returns (metric, score), or (metric(s), {subtask: score}) for a group task.

    Supported tasks include:
        - hellaswag: commonsense reasoning
        - gsm8k: grade school math
        - arc_easy, arc_challenge: science reasoning
        - winogrande: commonsense
        - math (hendrycks_math): competition math
    """
    # GPT-2 tokenizer, matching the vocabulary used in pre-training
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    if max_length is not None:
        tokenizer.model_max_length = max_length
        
    lm = PIT(model, tokenizer, max_length=max_length, batch_size=batch_size)

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
        """Return (metric, value) for the best metric in a subtask's results dict."""
        for key in metric_keys:
            if key in subtask_results:
                return key, subtask_results[key]
        # Fallback: return first numeric result
        for key, value in subtask_results.items():
            if isinstance(value, (int, float)):
                return key, value
        return None

    # Single-task case: task_name is directly in results
    if task_name in results["results"]:
        task_results = results["results"][task_name]
        found = _extract_score(task_results)
        if found is not None:
            return found
        raise ValueError(f"Could not find accuracy metric in results: {task_results.keys()}")

    # Group task (e.g. "glue"): results are keyed by subtask, so collect every subtask score
    subtask_scores = {}
    for key, val in results["results"].items():
        found = _extract_score(val)
        if found is not None:
            subtask_scores[key] = found

    if subtask_scores:
        # Metric name(s) used by the subtasks, joined with "/" if they differ
        metric = "/".join(sorted({m for m, _ in subtask_scores.values()}))
        return metric, {key: score for key, (_, score) in subtask_scores.items()}
    
    raise ValueError(f"No results found for task '{task_name}'. Available: {list(results['results'].keys())}")
        


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
    
    # Strip DDP ("module.") and torch.compile ("_orig_mod.") key prefixes
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
    name = basename.replace("_checkpoint.pt", "")
    return name


def main():
    parser = argparse.ArgumentParser(description="Evaluate GPT checkpoints on lm-eval benchmarks (single process)")
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

    # Single process only: PIT does not implement lm-eval's multi-process hooks
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        sys.exit("eval_multigpu.py runs as a single process (no multi-GPU support). "
                 "Run it without torchrun: python3 eval/eval_multigpu.py --model 4B --checkpoints ...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating on {device}")

    # GPT-2 vocab (50257) padded to 50304, a multiple of 128
    num_vocab = 50304
    config_class = MODEL_CONFIGS[args.model]
    max_len = CONTEXT_LENGTHS[args.model]

    os.makedirs(args.output_dir, exist_ok=True)

    # Evaluate each checkpoint in turn
    for ckpt_path in args.checkpoints:
        model = GPT(config_class(vocab_size=num_vocab))
        model = load_checkpoint(ckpt_path, model, device)

        ckpt_name = extract_checkpoint_name(ckpt_path)
        print(f"\nEvaluating {ckpt_name} on {len(args.tasks)} tasks...")

        rows = []
        for task_name in args.tasks:
            print(f"\n{'─'*50}")
            print(f"  {ckpt_name}: {task_name}")
            print(f"{'─'*50}")
            try:
                metric, result = test_model(
                    model, args.limit,
                    batch_size=args.batch_size,
                    max_length=max_len,
                    task_name=task_name,
                    num_fewshot=args.num_fewshot,
                )
            except Exception:
                # Abort instead of writing a partial CSV for this checkpoint
                print(f"  ❌ {task_name} failed; no CSV written for {ckpt_name}")
                raise

            if isinstance(result, dict):
                avg = sum(result.values()) / len(result)
                rows.append({"task": task_name, "metric": metric, "score": f"{avg:.4f}"})
                print(f"  ✅ {task_name}: {avg:.4f} {metric} (avg of {len(result)} subtasks)")
            else:
                rows.append({"task": task_name, "metric": metric, "score": f"{result:.4f}"})
                print(f"  ✅ {task_name}: {result:.4f} {metric}")

        # Save results
        result_file = os.path.join(args.output_dir, f"{ckpt_name}.csv")
        with open(result_file, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["task", "metric", "score"])
            w.writeheader()
            w.writerows(rows)

        print(f"\n{'='*50}")
        print(f"📄 {ckpt_name} — saved to {result_file}")
        print(f"{'='*50}")
        for r in rows:
            print(f"  {r['task']:20s} {r['metric']:15s} {r['score']}")
        if rows:
            avg_all = sum(float(r["score"]) for r in rows) / len(rows)
            print(f"  {'AVERAGE':20s} {'':15s} {avg_all:.4f}")

        # Free GPU memory before the next checkpoint
        del model
        torch.cuda.empty_cache()

    print("\nDone evaluating all checkpoints.")


if __name__ == "__main__":
    main()
