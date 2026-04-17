import matplotlib.pyplot as plt
import numpy as np

# Data from summary files
models = ['GPT-4B-SFT_2019', 'ChronoGPT-SFT_2020', 'Qwen1.5-1.8B']
metrics = ['Prompt Strict', 'Prompt Loose', 'Inst Strict', 'Inst Loose', 'Average']

data = {
    'GPT-4B-SFT_2019':       [20.7, 23.3, 34.3, 36.9],
    'ChronoGPT-SFT_2020':   [18.7, 19.8, 28.8, 30.7],
    'Qwen1.5-1.8B':     [14.0, 16.5, 24.1, 26.3],
}
# Append the average for each model
for model in data:
    avg = round(np.mean(data[model]), 1)
    data[model].append(avg)

x = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))

colors = ['#2563eb', '#f59e0b', '#10b981']

for i, (model, values) in enumerate(data.items()):
    bars = ax.bar(x + i * width, values, width, label=model, color=colors[i], edgecolor='white', linewidth=0.5)
    # Add value labels on top of each bar
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{val}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_xticks(x + width)
ax.set_xticklabels(metrics, fontsize=11)
ax.legend(fontsize=10, loc='upper left')
ax.set_ylim(0, 45)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('/home/tengandrea/Documents/FinGPT/ifeval_outputs/ifeval_histogram.png', dpi=300)
plt.savefig('/home/tengandrea/Documents/FinGPT/ifeval_outputs/ifeval_histogram.pdf', bbox_inches='tight', facecolor='white')
plt.show()
print("Saved to ifeval_outputs/ifeval_histogram.png and .pdf")
