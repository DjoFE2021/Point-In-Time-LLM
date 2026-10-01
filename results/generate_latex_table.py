#!/usr/bin/env python3
"""Print a LaTeX table of zero-shot accuracy (BoolQ, PIQA, HellaSwag, WinoGrande, ARC, OBQA).
Reads results/<name>.csv for the PIT checkpoints (--pit-1b, --pit-4b) and the models in MODELS;
the best non-baseline score per column is bolded.

Usage: python results/generate_latex_table.py [--pit-4b <CSV_NAME>] [--pit-1b <CSV_NAME>] > table.tex
"""

import argparse
import csv
import os
import sys

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))

# Rows after the PIT rows: (CSV file name without .csv, LaTeX display name, is_baseline)
# The paper's DatedGPT row is copied from Yan et al. (2026) and is not generated here.
MODELS = [
    ("manelalab_chrono-gpt-v1-20241231",  "ChronoGPT\\_2024",       False),
    ("google_gemma-3-1b-pt",              "Gemma-3-1B",             True),
    ("google_gemma-3-4b-pt",              "Gemma-3-4B",             True),
    ("huggyllama_llama-7b",               "LLaMA-7B",               True),
]

# Columns: lm-eval task names and headers
TASKS = ["boolq", "piqa", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa"]
TASK_DISPLAY = ["BoolQ", "PIQA", "HellaSwag", "WinoGrande", "ARC-easy", "ARC-chal.", "OBQA"]

# Per task: acc_norm if the CSV has it, otherwise acc (lm-eval reports only acc for BoolQ and WinoGrande)
METRICS = ["acc_norm,none", "acc,none"]


def load_results(csv_path):
    """Load a results CSV and return {task: score}, preferring acc_norm over acc."""
    by_metric = {metric: {} for metric in METRICS}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task = row["task"].strip()
            metric = row["metric"].strip().strip('"')
            if metric in by_metric:
                try:
                    by_metric[metric][task] = float(row["score"])
                except (ValueError, KeyError):
                    continue
    scores = {}
    for metric in reversed(METRICS):  # preferred metrics overwrite the fallback
        scores.update(by_metric[metric])
    return scores


def main():
    parser = argparse.ArgumentParser(description="Print the zero-shot CSR results as a LaTeX table")
    parser.add_argument("--pit-4b", type=str, default="2024-12_step=56984",
                        help="PIT-4B results CSV name in results/, without .csv (default: 2024-12_step=56984)")
    parser.add_argument("--pit-1b", type=str, default=None,
                        help="PIT-1B results CSV name in results/, without .csv (row skipped if not given)")
    args = parser.parse_args()

    models = [
        (args.pit_1b, "PIT-1B\\_2024 (Ours)", False),
        (args.pit_4b, "PIT-4B\\_2024 (Ours)", False),
    ] + MODELS

    # Load all model results
    model_data = []
    for csv_name, display_name, is_baseline in models:
        if csv_name is None:
            print(f"⚠️  Skipping {display_name}: no CSV name given", file=sys.stderr)
            continue
        csv_path = os.path.join(RESULTS_DIR, f"{csv_name}.csv")
        if not os.path.exists(csv_path):
            print(f"⚠️  Skipping {display_name}: {csv_path} not found", file=sys.stderr)
            continue
        scores = load_results(csv_path)
        model_data.append((display_name, scores, is_baseline))

    if not model_data:
        print("No results found!", file=sys.stderr)
        sys.exit(1)

    # Best score per task among non-baseline models (bolded in the table)
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

    # Print the LaTeX table to stdout
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

        # Average over the tasks this model has scores for
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
