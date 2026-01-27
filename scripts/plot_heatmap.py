import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

# Load data
df = pd.read_csv("output/learning_forgetting_sweep_heatmap.csv")

# Create grid for interpolation
n_grid = 50
learning_rates = np.linspace(
    df["learning_rate"].min(), df["learning_rate"].max(), n_grid
)
forgetting_rates = np.linspace(
    df["forgetting_rate"].min(), df["forgetting_rate"].max(), n_grid
)
L, F = np.meshgrid(learning_rates, forgetting_rates)

# Statistics to plot
stats = [
    ("avg_weight", "Average Weight"),
    ("avg_firing_count", "Average Firing Count"),
    ("avg_avalanche_size", "Average Avalanche Size"),
    ("n_avalanches", "Number of Avalanches"),
]

# Create a 2x2 subplot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Learning/Forgetting Rate Sweep Heatmaps", fontsize=14, fontweight="bold")

for ax, (stat, title) in zip(axes.flat, stats):
    # Interpolate data onto grid
    Z = griddata(
        (df["learning_rate"], df["forgetting_rate"]),
        df[stat],
        (L, F),
        method="linear",
    )

    # Plot heatmap
    im = ax.pcolormesh(L, F, Z, shading="auto", cmap="viridis")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Forgetting Rate")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig(
    "output/learning_forgetting_sweep_heatmap.png", dpi=150, bbox_inches="tight"
)
print("Plot saved to output/learning_forgetting_sweep_heatmap.png")
plt.show()
