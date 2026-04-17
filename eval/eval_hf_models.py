"""
Evaluate standard HuggingFace models on the same benchmark suite
as eval_hellaswag_multigpu.py (Common Sense Reasoning by default).

Uses lm-eval's built-in 'hf' model type — no custom adapter needed.

Usage:
    # Evaluate GPT-2 on the full CSR suite (0-shot)
    python eval/eval_hf_models.py --models gpt2

    # Compare multiple models
    python eval/eval_hf_models.py \
        --models gpt2 Qwen/Qwen1.5-1.8B-Chat meta-llama/Llama-2-7b-hf

    # Quick debug with limits
    python eval/eval_hf_models.py --models gpt2 --limit 20 --tasks hellaswag

    # Custom tasks and few-shot
    python eval/eval_hf_models.py --models gpt2 --tasks mmlu --num-fewshot 5
"""

import argparse
import csv
import os

from lm_eval.evaluator import simple_evaluate


CSR_TASKS = [
    "boolq", "piqa", "hellaswag", "winogrande",
    "arc_easy", "arc_challenge", "openbookqa",
]


def build_model_args(model_name: str, args) -> str:
    """Build the model_args string for lm-eval's hf backend."""
    parts = [f"pretrained={model_name}"]
    if args.dtype:
        parts.append(f"dtype={args.dtype}")
    if args.trust_remote_code:
        parts.append("trust_remote_code=True")
    if args.device:
        parts.append(f"device={args.device}")
    return ",".join(parts)


def extract_scores(results: dict) -> list[dict]:
    """
    Pull task-level scores from lm-eval results dict into
    [{"task": ..., "metric": ..., "score": ...}, ...] rows.
    """
    rows = []
    for task_name, task_results in results["results"].items():
        for metric, value in sorted(task_results.items()):
            if not isinstance(value, (int, float)):
                continue
            if metric.endswith(",none") or metric.endswith(",flexible-extract"):
                rows.append({
                    "task": task_name,
                    "metric": metric,
                    "score": f"{value:.4f}",
                })
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate HuggingFace models on lm-eval benchmarks"
    )
    parser.add_argument(
        "--models", type=str, nargs="+", required=True,
        help="HuggingFace model name(s) to evaluate",
    )
    parser.add_argument(
        "--tasks", type=str, nargs="+", default=CSR_TASKS,
        help="Evaluation tasks (default: Common Sense Reasoning suite)",
    )
    parser.add_argument(
        "--num-fewshot", type=int, default=0,
        help="Number of few-shot examples (default: 0)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of examples per task (for debugging)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/",
        help="Directory to save result CSVs",
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device for model (e.g. cuda, cuda:0). Default: auto",
    )
    parser.add_argument(
        "--trust-remote-code", action="store_true",
        help="Trust remote code when loading model/tokenizer",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"  Evaluating: {model_name}")
        print(f"  Tasks:      {args.tasks}")
        print(f"  Few-shot:   {args.num_fewshot}")
        print(f"  Batch size: {args.batch_size}")
        print(f"{'='*60}\n")

        model_args = build_model_args(model_name, args)

        results = simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            limit=args.limit,
        )

        rows = extract_scores(results)

        # Save CSV (same format as eval_hellaswag_multigpu.py)
        safe_name = model_name.replace("/", "_")
        result_file = os.path.join(args.output_dir, f"{safe_name}.csv")
        with open(result_file, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["task", "metric", "score"])
            w.writeheader()
            w.writerows(rows)

        # Print summary
        print(f"\n{'='*60}")
        print(f"📄 {model_name} — saved to {result_file}")
        print(f"{'='*60}")
        for r in rows:
            print(f"  {r['task']:30s} {r['metric']:25s} {r['score']}")

        # Compute and print average (over acc_norm or acc metrics)
        acc_rows = [r for r in rows if "acc" in r["metric"]]
        if acc_rows:
            # Take the best metric per task (prefer acc_norm over acc)
            task_best = {}
            for r in acc_rows:
                task = r["task"]
                metric = r["metric"]
                score = float(r["score"])
                if task not in task_best or "norm" in metric:
                    task_best[task] = score
            avg = sum(task_best.values()) / len(task_best)
            print(f"  {'AVERAGE':30s} {'':25s} {avg:.4f}")

    print("\n✅ All evaluations complete.")


if __name__ == "__main__":
    main()
