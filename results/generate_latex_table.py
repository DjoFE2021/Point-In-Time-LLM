#!/usr/bin/env python3
"""
Generate LaTeX table from results CSVs.

Usage:
    python results/generate_latex_table.py

Reads results/{model_name}.csv files and produces a LaTeX table with
zero-shot accuracy on BoolQ, PIQA, HellaSwag, WinoGrande, ARC-easy,
ARC-challenge, and OpenBookQA.  All scores use acc_norm,none.
"""

import csv
import os
import sys

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Define which models to include and their display names ──────────
# Format: (csv_filename_without_ext, display_name, is_baseline)
MODELS = [
    ("2024-12_step=56984",                "PIT-4B\\_2024 (Ours)",    False),
    ("manelalab_chrono-gpt-v1-20241231",  "ChronoGPT\\_2024",       False),
    ("google_gemma-3-1b-pt",              "Gemma-3-1B",             True),
    ("google_gemma-3-4b-pt",              "Gemma-3-4B",             True),
    ("huggyllama_llama-7b",               "LLaMA-7B",               True),
    ("microsoft_phi-4",                   "Phi-4",                  True),
]

# ── Benchmark tasks (all use acc_norm,none) ────────────────────────
METRIC = "acc_norm,none"
TASKS = ["boolq", "piqa", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa"]
TASK_DISPLAY = ["BoolQ", "PIQA", "HellaSwag", "WinoGrande", "ARC-easy", "ARC-chal.", "OBQA"]


def load_results(csv_path):
    """Load a results CSV and return {task: score} for acc_norm."""
    scores = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task = row["task"].strip()
            metric = row["metric"].strip().strip('"')
            if metric == METRIC:
                try:
                    scores[task] = float(row["score"])
                except (ValueError, KeyError):
                    continue
    return scores


def main():
    # Load all model results
    model_data = []
    for csv_name, display_name, is_baseline in MODELS:
        csv_path = os.path.join(RESULTS_DIR, f"{csv_name}.csv")
        if not os.path.exists(csv_path):
            print(f"⚠️  Skipping {display_name}: {csv_path} not found", file=sys.stderr)
            continue
        scores = load_results(csv_path)
        model_data.append((display_name, scores, is_baseline))

    if not model_data:
        print("No results found!", file=sys.stderr)
        sys.exit(1)

    # Find max score per task (for bolding) — only among non-baseline models
    task_max = {}
    for task in TASKS:
        values = []
        for _, scores, is_baseline in model_data:
            if not is_baseline and task in scores:
                values.append(scores[task])
        task_max[task] = max(values) if values else 0

    # Find max average among non-baseline models
    avg_max = 0
    for _, scores, is_baseline in model_data:
        if not is_baseline:
            vals = [scores[t] for t in TASKS if t in scores]
            if vals:
                avg_max = max(avg_max, sum(vals) / len(vals))

    # ── Generate LaTeX ─────────────────────────────────────────────
    task_headers = " & ".join(TASK_DISPLAY)
    print(r"\begin{table}[htb]")
    print(r"\centering")
    print(r"\caption{Zero-shot accuracy (\%) on standard common sense reasoning benchmarks.}")
    print(r"\label{tab:pt_results}")
    print(r"\resizebox{\textwidth}{!}{")
    print(r"\begin{tabular}{l" + "c" * len(TASKS) + "c}")
    print(r"\toprule")
    print(f"Model & {task_headers} & Avg. \\\\")
    print(r"\midrule")

    prev_baseline = False
    for display_name, scores, is_baseline in model_data:
        # Add midrule before baselines
        if is_baseline and not prev_baseline:
            print(r"\midrule")
        prev_baseline = is_baseline

        cells = []
        for task in TASKS:
            val = scores.get(task)
            if val is None:
                cells.append("--")
            else:
                pct = val * 100
                formatted = f"{pct:.1f}"
                if not is_baseline and abs(val - task_max[task]) < 1e-6:
                    formatted = r"\textbf{" + formatted + "}"
                cells.append(formatted)

        # Compute average
        vals = [scores[t] for t in TASKS if t in scores]
        if vals:
            avg = sum(vals) / len(vals)
            avg_fmt = f"{avg * 100:.1f}"
            if not is_baseline and abs(avg - avg_max) < 1e-6:
                avg_fmt = r"\textbf{" + avg_fmt + "}"
        else:
            avg_fmt = "--"
        cells.append(avg_fmt)

        row_str = " & ".join(cells)
        print(f"{display_name}")
        print(f"& {row_str} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
