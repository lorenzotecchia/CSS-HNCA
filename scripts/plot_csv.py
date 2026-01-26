import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load data
df = pd.read_csv("output/learning_forgetting_sweep.csv")

# Create a 2x2 subplot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Learning/Forgetting Rate Sweep Analysis", fontsize=14, fontweight="bold")

# Sort by learning rate for better visualization
df_sorted = df.sort_values("learning_rate")

# Plot 1: Average Weight vs Learning Rate
ax1 = axes[0, 0]
ax1.scatter(
    df_sorted["learning_rate"],
    df_sorted["avg_weight"],
    c=df_sorted["n_avalanches"],
    cmap="viridis",
    alpha=0.7,
    s=40,
)
ax1.set_xlabel("Learning Rate")
ax1.set_ylabel("Average Weight")
ax1.set_title("Avg Weight vs Learning Rate")
ax1.grid(True, alpha=0.3)

# Plot 2: Average Firing Count vs Learning Rate
ax2 = axes[0, 1]
ax2.scatter(
    df_sorted["learning_rate"],
    df_sorted["avg_firing_count"],
    c=df_sorted["n_avalanches"],
    cmap="viridis",
    alpha=0.7,
    s=40,
)
ax2.set_xlabel("Learning Rate")
ax2.set_ylabel("Average Firing Count")
ax2.set_title("Avg Firing Count vs Learning Rate")
ax2.grid(True, alpha=0.3)

# Plot 3: Average Avalanche Size vs Learning Rate
ax3 = axes[1, 0]
mask = df_sorted["avg_avalanche_size"] > 0  # Only show non-zero
ax3.scatter(
    df_sorted.loc[mask, "learning_rate"],
    df_sorted.loc[mask, "avg_avalanche_size"],
    c=df_sorted.loc[mask, "n_avalanches"],
    cmap="viridis",
    alpha=0.7,
    s=40,
)
ax3.set_xlabel("Learning Rate")
ax3.set_ylabel("Average Avalanche Size")
ax3.set_title("Avg Avalanche Size vs Learning Rate")
ax3.grid(True, alpha=0.3)
ax3.set_yscale("log")

# Plot 4: Number of Avalanches vs Learning Rate
ax4 = axes[1, 1]
scatter = ax4.scatter(
    df_sorted["learning_rate"],
    df_sorted["n_avalanches"],
    c=df_sorted["avg_weight"],
    cmap="plasma",
    alpha=0.7,
    s=40,
)
ax4.set_xlabel("Learning Rate")
ax4.set_ylabel("Number of Avalanches")
ax4.set_title("N Avalanches vs Learning Rate")
ax4.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax4, label="Avg Weight")

plt.tight_layout()
plt.savefig("output/learning_forgetting_sweep_plot.png", dpi=150, bbox_inches="tight")
print("Plot saved to output/learning_forgetting_sweep_plot.png")
plt.show()
