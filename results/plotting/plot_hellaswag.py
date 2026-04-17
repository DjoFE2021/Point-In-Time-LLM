"""
Plot HellaSwag accuracy over time — filtered to dates with full CSR results.
Similar to plot_hellaswag.py but with fewer data points (only evaluated checkpoints).

Usage:
    python plots/plot_csr_benchmarks.py
"""

import csv
import glob
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Scientific paper style ─────────────────────────────────────
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 14,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 1.2,
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#333333',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'grid.alpha': 0.4,
    'grid.linestyle': '--',
    'grid.linewidth': 0.6,
})


def get_csr_dates(results_dir="results"):
    """Get dates that have full CSR eval CSVs."""
    pattern = os.path.join(results_dir, "20*_step=*.csv")
    dates = set()
    for f in glob.glob(pattern):
        m = re.match(r"(\d{4}-\d{2})_step=\d+", os.path.basename(f))
        if m:
            with open(f, "r") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    if row["task"].strip() == "hellaswag":
                        dates.add(m.group(1))
                        break
    return dates


def main():
    csr_dates = get_csr_dates()
    csr_dates_dt = {pd.Timestamp(d) for d in csr_dates}

    print(f"Found {len(csr_dates)} dates with CSR results:")
    for d in sorted(csr_dates):
        print(f"  {d}")

    # ── Load and filter HellaSwag time series ──
    results_1b = pd.read_csv("results/hellaswag_1B.csv")
    results_1b["date"] = pd.to_datetime(results_1b["date"], format="%Y-%m")
    results_1b = results_1b[results_1b["date"].isin(csr_dates_dt)]
    results_1b = results_1b.sort_values("date").reset_index(drop=True)

    results_4b = pd.read_csv("results/hellaswag_4B.csv")
    results_4b["date"] = pd.to_datetime(results_4b["date"], format="%Y-%m")
    results_4b = results_4b[results_4b["date"].isin(csr_dates_dt)]
    results_4b = results_4b.sort_values("date").reset_index(drop=True)
    
    # ── Merge dates for x-axis ──
    all_dates = sorted(csr_dates_dt)

    # ── Plot ────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Token distribution (if available)
    tokens = None
    if os.path.exists("data/tokens_dist.csv"):
        tokens = pd.read_csv("data/tokens_dist.csv")
        tokens["date"] = pd.to_datetime(tokens["date"], format="%Y-%m")

    # 1.5B model
    x_1b = [all_dates.index(d) for d in results_1b["date"] if d in all_dates]
    if len(x_1b) > 0:
        last_1b = results_1b["hellaswag_accuracy"].iloc[-1] * 100
        ax1.plot(x_1b, results_1b["hellaswag_accuracy"] * 100,
                 marker="o", color="#1f77b4", linewidth=2.5, markersize=6,
                 markeredgecolor="white", markeredgewidth=1,
                 label=rf"PIT-1.5B (2024) $\bf{{({last_1b:.1f}\%)}}$", zorder=5)

    # 4B model
    x_4b = [all_dates.index(d) for d in results_4b["date"] if d in all_dates]
    if len(x_4b) > 0:
        last_4b = results_4b["hellaswag_accuracy"].iloc[-1] * 100
        ax1.plot(x_4b, results_4b["hellaswag_accuracy"] * 100,
                 marker="D", color="#9467bd", linewidth=2.5, markersize=8,
                 markeredgecolor="white", markeredgewidth=1,
                 label=rf"PIT-4B (2024) $\bf{{({last_4b:.1f}\%)}}$", zorder=5)

    ax1.set_ylabel("HellaSwag Accuracy (%)")
    ax1.set_ylim(0, 100)

    # Baselines
    baselines = [
        (77,   "#2ca02c", "-",  r"Gemma3-4B $\bf{(77\%)}$"),
        (76,   "#17becf", "-",  r"LLaMA-7B $\bf{(76\%)}$"),
        (62.3, "#ff7f0e", "-",  r"Gemma3-1B $\bf{(62.3\%)}$"),
        (53.2, "#e377c2", "-", r"DatedGPT $\bf{(53.2\%)}$"),
        (48.0, "#d62728", "-",  r"GPT2-XL-1.5B $\bf{(50.9\%)}$"),
        (44,   "#8c564b", "-",  r"ChronoGPT (2024) $\bf{(44\%)}$"),
        (25,   "#7f7f7f", "--", r"Random Guess $\bf{(25\%)}$"),
    ]
    for y, color, ls, label in baselines:
        ax1.axhline(y=y, color=color, linestyle=ls, linewidth=1.8, alpha=0.8, label=label, zorder=2)

    # X-axis: dates
    ax1.set_xticks(range(len(all_dates)))
    ax1.set_xticklabels([d.strftime("%Y-%m") for d in all_dates], rotation=45, ha="right")
    ax1.set_xlabel("Date")

    # Legend
    legend = ax1.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                        frameon=True, fancybox=True, shadow=True,
                        edgecolor='#cccccc', facecolor='white')
    legend.get_frame().set_linewidth(1.0)

    ax1.grid(True, zorder=1)
    ax1.set_axisbelow(True)

    for spine in ax1.spines.values():
        spine.set_linewidth(1.2)

    plt.tight_layout()
    plt.subplots_adjust(right=0.68)

    plt.savefig("results/plots/csr_benchmarks.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig("results/plots/csr_benchmarks.pdf", bbox_inches='tight', facecolor='white')
    print("\n✅ Saved results/plots/csr_benchmarks.png and .pdf")


if __name__ == "__main__":
    main()
