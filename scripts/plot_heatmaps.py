#!/usr/bin/env python3
"""Heatmap visualization for learning/forgetting rate sweep results.

Plots statistics as heatmaps with:
- X-axis: learning rate
- Y-axis: forgetting rate
- Color: statistic value

Run with: python scripts/plot_heatmaps.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load data
df = pd.read_csv("output/learning_forgetting_sweep.csv")

# Get unique sorted values for axes
learning_rates = np.sort(df["learning_rate"].unique())
forgetting_rates = np.sort(df["forgetting_rate"].unique())

# Statistics to plot
stats = [
    ("avg_weight", "Average Weight", "viridis"),
    ("std_weight", "Std Weight", "viridis"),
    ("avg_firing_count", "Average Firing Count", "plasma"),
    ("std_firing_count", "Std Firing Count", "plasma"),
    ("avg_avalanche_size", "Average Avalanche Size", "inferno"),
    ("std_avalanche_size", "Std Avalanche Size", "inferno"),
    ("avg_avalanche_duration", "Average Avalanche Duration", "cividis"),
    ("std_avalanche_duration", "Std Avalanche Duration", "cividis"),
]


def create_heatmap_matrix(df, stat_col, learning_rates, forgetting_rates):
    """Create 2D matrix for heatmap from dataframe."""
    matrix = np.zeros((len(forgetting_rates), len(learning_rates)))
    matrix[:] = np.nan

    for _, row in df.iterrows():
        lr_idx = np.where(learning_rates == row["learning_rate"])[0][0]
        fr_idx = np.where(forgetting_rates == row["forgetting_rate"])[0][0]
        matrix[fr_idx, lr_idx] = row[stat_col]

    return matrix


# Create 4x2 subplot for all statistics
fig, axes = plt.subplots(4, 2, figsize=(14, 20))
fig.suptitle(
    "Learning/Forgetting Rate Sweep - Heatmaps", fontsize=16, fontweight="bold"
)

for idx, (stat_col, title, cmap) in enumerate(stats):
    ax = axes[idx // 2, idx % 2]

    matrix = create_heatmap_matrix(df, stat_col, learning_rates, forgetting_rates)

    im = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        extent=[
            learning_rates[0],
            learning_rates[-1],
            forgetting_rates[0],
            forgetting_rates[-1],
        ],
    )

    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Forgetting Rate")
    ax.set_title(title)

    # Set tick positions
    ax.set_xticks(learning_rates)
    ax.set_yticks(forgetting_rates)
    ax.set_xticklabels([f"{lr:.4f}" for lr in learning_rates], rotation=45, ha="right")
    ax.set_yticklabels([f"{fr:.4f}" for fr in forgetting_rates])

    plt.colorbar(im, ax=ax, label=title)

plt.tight_layout()
plt.savefig("output/learning_forgetting_heatmaps.png", dpi=150, bbox_inches="tight")
print("Plot saved to output/learning_forgetting_heatmaps.png")
plt.show()
