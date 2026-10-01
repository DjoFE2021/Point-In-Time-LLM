"""Grouped bar chart of IFEval accuracy (prompt/instruction level, strict/loose, average) per model.

Reads the <name>_summary.txt files written by eval/ifeval_test.py; each argument is LABEL=PATH, or just PATH
(labelled with the model name in the file). Writes <output>.png and <output>.pdf (default: results/plots/ifeval_histogram).
Usage: python results/plotting/plot_ifeval_histogram.py LABEL=ifeval_outputs_new/<name>_summary.txt [...] [--output OUT]
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

# Bar groups and the summary-file keys they come from (the average is computed here)
metrics = ['Prompt Strict', 'Prompt Loose', 'Inst Strict', 'Inst Loose', 'Average']
SUMMARY_KEYS = ['prompt_level_strict_acc', 'prompt_level_loose_acc', 'inst_level_strict_acc', 'inst_level_loose_acc']


def read_summary(path):
    """Parse an ifeval_test.py summary file ("key: value" lines) into a dict of strings."""
    fields = {}
    with open(path) as f:
        for line in f:
            key, sep, value = line.partition(":")
            if sep:
                fields[key.strip()] = value.strip()
    return fields


def main():
    parser = argparse.ArgumentParser(description="Plot IFEval accuracy from eval/ifeval_test.py summary files")
    parser.add_argument("summaries", nargs="+", metavar="LABEL=PATH",
                        help="Summary file, optionally prefixed with its legend label, "
                             "e.g. Qwen1.5-1.8B=ifeval_outputs_new/Qwen1.5-1.8B-Chat_summary.txt")
    parser.add_argument("--output", default="results/plots/ifeval_histogram",
                        help="Output path without extension (default: results/plots/ifeval_histogram)")
    args = parser.parse_args()

    # Scores (%) per model: the four accuracies from the summary file, then their average
    data = {}
    for item in args.summaries:
        # Split at the first "=" only, since checkpoint names contain "=" (e.g. 2019-12_step=32559)
        label, path = (None, item) if os.path.isfile(item) or "=" not in item else item.split("=", 1)
        if not os.path.isfile(path):
            parser.error(f"summary file not found: {path}")
        fields = read_summary(path)
        values = [float(fields[key].rstrip('%')) for key in SUMMARY_KEYS]
        values.append(round(np.mean(values), 1))
        data[label or fields.get('model', path)] = values

    x = np.arange(len(metrics))
    width = 0.75 / len(data)  # 0.25 for three models

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#2563eb', '#f59e0b', '#10b981', '#ef4444', '#8b5cf6', '#6b7280']

    for i, (model, values) in enumerate(data.items()):
        bars = ax.bar(x + i * width, values, width, label=model, color=colors[i % len(colors)],
                      edgecolor='white', linewidth=0.5)
        # Add value labels on top of each bar
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xticks(x + width * (len(data) - 1) / 2)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.set_ylim(0, max(45, 1.2 * max(max(values) for values in data.values())))
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(f'{args.output}.png', dpi=300)
    plt.savefig(f'{args.output}.pdf', bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"Saved to {args.output}.png and .pdf")


if __name__ == "__main__":
    main()
