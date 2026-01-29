#!/usr/bin/env python3
"""Plot Snellius supercomputer parameter sweep results.

Loads all sweep_config_*.csv files, extracts the grid parameters from each,
and produces multi-panel figures emphasising how the input parameters
(firing_count, leak_rate/reset_potential, k_prop, decay_alpha/oja_alpha)
influence the simulation outputs.

Run with: python scripts/plot_snellius_sweep.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

OUTPUT_DIR = Path("output")
PLOT_DIR = OUTPUT_DIR / "plots_snellius"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load all sweep CSVs ──────────────────────────────────────────────────────
csvs = sorted(OUTPUT_DIR.glob("sweep_config_*.csv"))
print(f"Found {len(csvs)} sweep config files")

frames = []
for csv_path in csvs:
    df = pd.read_csv(csv_path)
    frames.append(df)

df_all = pd.concat(frames, ignore_index=True)
print(f"Total rows: {len(df_all)}")

# ── Derived labels for grouping ──────────────────────────────────────────────
df_all["firing_frac"] = df_all["firing_count"] / df_all["n_neurons"]
df_all["leak_reset"] = df_all.apply(
    lambda r: f"leak={r['leak_rate']:.2f}, reset={r['reset_potential']:.2f}", axis=1
)
df_all["regularisation"] = df_all.apply(
    lambda r: "Oja+decay" if r["oja_alpha"] > 0 else "None", axis=1
)

# Unique parameter values
firing_fracs = sorted(df_all["firing_frac"].unique())
leak_reset_labels = sorted(df_all["leak_reset"].unique())
k_props = sorted(df_all["k_prop"].unique())
reg_labels = sorted(df_all["regularisation"].unique())

# Build a config key per unique combination
df_all["config"] = df_all.apply(
    lambda r: (
        f"{int(r['firing_count'])}_{r['leak_rate']:.4f}_{r['reset_potential']:.4f}_"
        f"{r['k_prop']:.4f}_{r['decay_alpha']:.6f}_{r['oja_alpha']:.6f}"
    ),
    axis=1,
)

# ── Summary per configuration ────────────────────────────────────────────────
summary_rows = []
for config_key, grp in df_all.groupby("config"):
    row = {
        "firing_count": int(grp["firing_count"].iloc[0]),
        "firing_frac": grp["firing_frac"].iloc[0],
        "leak_rate": grp["leak_rate"].iloc[0],
        "reset_potential": grp["reset_potential"].iloc[0],
        "k_prop": grp["k_prop"].iloc[0],
        "decay_alpha": grp["decay_alpha"].iloc[0],
        "oja_alpha": grp["oja_alpha"].iloc[0],
        "leak_reset": grp["leak_reset"].iloc[0],
        "regularisation": grp["regularisation"].iloc[0],
        "n_samples": len(grp),
        "mean_avg_weight": grp["avg_weight"].mean(),
        "mean_avg_firing": grp["avg_firing_count"].mean(),
        "mean_n_avalanches": grp["n_avalanches"].mean(),
        "mean_avg_aval_size": grp["avg_avalanche_size"].mean(),
        "mean_avg_aval_dur": grp["avg_avalanche_duration"].mean(),
        "median_n_avalanches": grp["n_avalanches"].median(),
        "frac_with_avalanches": (grp["n_avalanches"] > 0).mean(),
    }
    summary_rows.append(row)

df_summary = pd.DataFrame(summary_rows)

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 1: Configuration overview — bar chart of key metrics per config
# ──────────────────────────────────────────────────────────────────────────────
print("Plotting Figure 1: Configuration overview...")

df_summary_sorted = df_summary.sort_values(
    ["firing_frac", "leak_rate", "k_prop", "regularisation"]
).reset_index(drop=True)

fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True)
fig.suptitle(
    "Snellius Sweep — Summary per Configuration\n"
    "(500 neurons, 3000 LHS samples each, lr=fr along diagonal)",
    fontsize=14,
    fontweight="bold",
)

# Config labels
labels = []
for _, r in df_summary_sorted.iterrows():
    labels.append(
        f"fc={int(r['firing_count'])}\n{r['leak_reset']}\n"
        f"k={r['k_prop']:.3f}\n{r['regularisation']}"
    )

x = np.arange(len(df_summary_sorted))
width = 0.7

# Colors by firing fraction
cmap = plt.cm.Set2
ff_colors = {ff: cmap(i / len(firing_fracs)) for i, ff in enumerate(firing_fracs)}
bar_colors = [ff_colors[r["firing_frac"]] for _, r in df_summary_sorted.iterrows()]

# Panel 1: Mean average weight
axes[0].bar(x, df_summary_sorted["mean_avg_weight"], width, color=bar_colors, edgecolor="k", linewidth=0.3)
axes[0].set_ylabel("Mean Avg Weight")
axes[0].set_title("Average Synaptic Weight")
axes[0].grid(axis="y", alpha=0.3)

# Panel 2: Mean average firing count
axes[1].bar(x, df_summary_sorted["mean_avg_firing"], width, color=bar_colors, edgecolor="k", linewidth=0.3)
axes[1].set_ylabel("Mean Avg Firing Count")
axes[1].set_title("Average Firing Count")
axes[1].grid(axis="y", alpha=0.3)

# Panel 3: Fraction of samples with avalanches + mean n_avalanches
ax3a = axes[2]
ax3b = ax3a.twinx()
ax3a.bar(x, df_summary_sorted["frac_with_avalanches"] * 100, width, color=bar_colors,
         edgecolor="k", linewidth=0.3, alpha=0.7, label="% samples with avalanches")
ax3b.plot(x, df_summary_sorted["mean_n_avalanches"], "ko-", markersize=3, linewidth=1,
          label="Mean n_avalanches")
ax3a.set_ylabel("% Samples with Avalanches")
ax3b.set_ylabel("Mean n_avalanches")
ax3a.set_title("Avalanche Activity")
ax3a.grid(axis="y", alpha=0.3)

axes[2].set_xticks(x)
axes[2].set_xticklabels(labels, fontsize=5.5, rotation=0, ha="center")
axes[2].set_xlabel("Configuration")

# Legend for firing fractions
legend_elements = [
    plt.Rectangle((0, 0), 1, 1, facecolor=ff_colors[ff], edgecolor="k", linewidth=0.5,
                   label=f"firing frac = {ff:.2f}")
    for ff in firing_fracs
]
axes[0].legend(handles=legend_elements, loc="upper right", fontsize=8)

plt.tight_layout()
fig.savefig(PLOT_DIR / "config_overview.png", dpi=150, bbox_inches="tight")
print(f"  Saved {PLOT_DIR / 'config_overview.png'}")
plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 2: Parameter influence — grouped box plots
# ──────────────────────────────────────────────────────────────────────────────
print("Plotting Figure 2: Parameter influence on avalanche metrics...")

metrics = [
    ("n_avalanches", "Number of Avalanches"),
    ("avg_avalanche_size", "Avg Avalanche Size"),
    ("avg_avalanche_duration", "Avg Avalanche Duration"),
    ("avg_weight", "Avg Synaptic Weight"),
]

param_dims = [
    ("firing_frac", "Firing Fraction", firing_fracs),
    ("leak_reset", "Leak / Reset Pair", leak_reset_labels),
    ("k_prop", "Connectivity (k_prop)", k_props),
    ("regularisation", "Regularisation", reg_labels),
]

fig, axes = plt.subplots(len(metrics), len(param_dims), figsize=(22, 16))
fig.suptitle(
    "Influence of Input Parameters on Simulation Metrics\n"
    "(each box aggregates all samples sharing that parameter value)",
    fontsize=14, fontweight="bold",
)

for row_idx, (metric_col, metric_label) in enumerate(metrics):
    for col_idx, (param_col, param_label, param_vals) in enumerate(param_dims):
        ax = axes[row_idx, col_idx]
        data_groups = []
        tick_labels = []
        for val in param_vals:
            subset = df_all[df_all[param_col] == val][metric_col].dropna()
            data_groups.append(subset.values)
            if isinstance(val, float):
                tick_labels.append(f"{val:.4f}")
            else:
                tick_labels.append(str(val))

        bp = ax.boxplot(data_groups, patch_artist=True, showfliers=False,
                        medianprops=dict(color="red", linewidth=1.5))
        colors_box = plt.cm.tab10(np.linspace(0, 0.5, len(data_groups)))
        for patch, c in zip(bp["boxes"], colors_box):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)

        ax.set_xticklabels(tick_labels, fontsize=7, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.3)

        if row_idx == 0:
            ax.set_title(param_label, fontsize=10, fontweight="bold")
        if col_idx == 0:
            ax.set_ylabel(metric_label, fontsize=9)

plt.tight_layout()
fig.savefig(PLOT_DIR / "parameter_influence_boxplots.png", dpi=150, bbox_inches="tight")
print(f"  Saved {PLOT_DIR / 'parameter_influence_boxplots.png'}")
plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 3: Learning rate vs metrics — one panel per metric, colored by config
# ──────────────────────────────────────────────────────────────────────────────
print("Plotting Figure 3: Learning rate sweeps coloured by parameter group...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    "Metrics vs Learning Rate (= Forgetting Rate)\n"
    "Colour = firing fraction, marker = leak/reset pair, hollow = Oja+decay",
    fontsize=13, fontweight="bold",
)

ff_cmap = {0.05: "tab:blue", 0.10: "tab:orange", 0.15: "tab:green"}
lr_markers = {}
for i, lab in enumerate(leak_reset_labels):
    lr_markers[lab] = ["o", "s", "D"][i % 3]

metric_list = [
    ("avg_weight", "Avg Weight"),
    ("avg_firing_count", "Avg Firing Count"),
    ("avg_avalanche_size", "Avg Avalanche Size"),
    ("n_avalanches", "Number of Avalanches"),
]

# Subsample for readability
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    df_sub = df_all.groupby("config", group_keys=False).apply(
        lambda g: g.sample(min(300, len(g)), random_state=42)
    ).reset_index(drop=True)

for idx, (metric_col, metric_label) in enumerate(metric_list):
    ax = axes[idx // 2, idx % 2]

    for _, row in df_sub.iterrows():
        ff = row["firing_frac"]
        lr_lab = row["leak_reset"]
        reg = row["regularisation"]
        color = ff_cmap.get(ff, "gray")

        if reg == "None":
            ax.scatter(
                row["learning_rate"], row[metric_col],
                color=color, marker=lr_markers.get(lr_lab, "o"),
                s=6, alpha=0.15, edgecolors="none",
            )
        else:
            ax.scatter(
                row["learning_rate"], row[metric_col],
                facecolors="none", edgecolors=color,
                marker=lr_markers.get(lr_lab, "o"),
                s=6, alpha=0.15, linewidths=0.5,
            )

    ax.set_xlabel("Learning Rate (= Forgetting Rate)")
    ax.set_ylabel(metric_label)
    ax.set_title(metric_label)
    ax.grid(True, alpha=0.3)
    if metric_col in ("avg_avalanche_size", "n_avalanches"):
        ax.set_yscale("symlog", linthresh=1)

# Legend
legend_elements = []
for ff, c in ff_cmap.items():
    legend_elements.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                                   markersize=8, label=f"firing frac={ff:.2f}"))
for lab, m in lr_markers.items():
    legend_elements.append(Line2D([0], [0], marker=m, color="gray", markerfacecolor="gray",
                                   markersize=8, label=lab, linestyle="None"))
legend_elements.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue",
                               markersize=8, label="No regularisation (filled)"))
legend_elements.append(Line2D([0], [0], marker="o", color="tab:blue", markerfacecolor="none",
                               markersize=8, label="Oja+decay (hollow)"))

fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=8,
           bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=[0, 0.06, 1, 0.96])
fig.savefig(PLOT_DIR / "lr_vs_metrics_by_params.png", dpi=150, bbox_inches="tight")
print(f"  Saved {PLOT_DIR / 'lr_vs_metrics_by_params.png'}")
plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 4: Heatmap — mean avalanche size across parameter grid
# ──────────────────────────────────────────────────────────────────────────────
print("Plotting Figure 4: Heatmap of mean avalanche activity per grid cell...")

# Pivot: rows = (leak_reset, regularisation), cols = (firing_frac, k_prop)
# Value = mean_avg_aval_size
pivot_data = df_summary.copy()
pivot_data["row_label"] = pivot_data["leak_reset"] + "\n" + pivot_data["regularisation"]
pivot_data["col_label"] = pivot_data.apply(
    lambda r: f"fc={r['firing_frac']:.2f}\nk={r['k_prop']:.3f}", axis=1
)

heatmap_metrics = [
    ("mean_avg_aval_size", "Mean Avg Avalanche Size"),
    ("mean_n_avalanches", "Mean Number of Avalanches"),
    ("frac_with_avalanches", "Fraction with Avalanches"),
    ("mean_avg_weight", "Mean Avg Weight"),
]

fig, axes = plt.subplots(2, 2, figsize=(20, 14))
fig.suptitle(
    "Parameter Grid Heatmaps (rows: leak/reset + regularisation, cols: firing_frac + k_prop)",
    fontsize=13, fontweight="bold",
)

for idx, (metric_col, metric_title) in enumerate(heatmap_metrics):
    ax = axes[idx // 2, idx % 2]

    pivot_table = pivot_data.pivot_table(
        values=metric_col, index="row_label", columns="col_label", aggfunc="mean"
    )
    # Sort columns and rows deterministically
    pivot_table = pivot_table.reindex(
        sorted(pivot_table.columns), axis=1
    ).reindex(sorted(pivot_table.index))

    im = ax.imshow(pivot_table.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(np.arange(len(pivot_table.columns)))
    ax.set_yticks(np.arange(len(pivot_table.index)))
    ax.set_xticklabels(pivot_table.columns, fontsize=7, rotation=45, ha="right")
    ax.set_yticklabels(pivot_table.index, fontsize=7)
    ax.set_title(metric_title, fontsize=11)

    # Annotate cells
    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            val = pivot_table.values[i, j]
            if not np.isnan(val):
                text_color = "white" if val > pivot_table.values[~np.isnan(pivot_table.values)].max() * 0.6 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=6, color=text_color)

    plt.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
fig.savefig(PLOT_DIR / "parameter_grid_heatmaps.png", dpi=150, bbox_inches="tight")
print(f"  Saved {PLOT_DIR / 'parameter_grid_heatmaps.png'}")
plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 5: Per-config scatter — lr vs avg_avalanche_size, faceted by params
# ──────────────────────────────────────────────────────────────────────────────
print("Plotting Figure 5: Faceted scatter by firing fraction and leak/reset...")

fig, axes = plt.subplots(
    len(firing_fracs), len(leak_reset_labels), figsize=(18, 14), sharex=True, sharey=True
)
fig.suptitle(
    "Avg Avalanche Size vs Learning Rate\n"
    "Faceted by Firing Fraction (rows) and Leak/Reset (cols)\n"
    "Colour = k_prop, Shape = regularisation",
    fontsize=13, fontweight="bold",
)

kp_colors = {kp: c for kp, c in zip(k_props, ["tab:blue", "tab:orange", "tab:green"])}
reg_markers_map = {"None": "o", "Oja+decay": "^"}

for i, ff in enumerate(firing_fracs):
    for j, lr_lab in enumerate(leak_reset_labels):
        ax = axes[i, j]
        subset = df_all[(df_all["firing_frac"] == ff) & (df_all["leak_reset"] == lr_lab)]

        for kp in k_props:
            for reg in reg_labels:
                sub2 = subset[(subset["k_prop"] == kp) & (subset["regularisation"] == reg)]
                if len(sub2) == 0:
                    continue
                # Subsample for speed
                sub2 = sub2.sample(min(200, len(sub2)), random_state=42)
                ax.scatter(
                    sub2["learning_rate"],
                    sub2["avg_avalanche_size"],
                    c=kp_colors[kp],
                    marker=reg_markers_map[reg],
                    s=8, alpha=0.3, edgecolors="none",
                )

        if i == 0:
            ax.set_title(lr_lab, fontsize=9)
        if j == 0:
            ax.set_ylabel(f"ff={ff:.2f}\nAvg Aval Size", fontsize=9)
        if i == len(firing_fracs) - 1:
            ax.set_xlabel("Learning Rate (= Forgetting Rate)", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("symlog", linthresh=1)

# Legend
legend_elements = []
for kp, c in kp_colors.items():
    legend_elements.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                                   markersize=8, label=f"k_prop={kp:.3f}"))
for reg, m in reg_markers_map.items():
    legend_elements.append(Line2D([0], [0], marker=m, color="gray", markerfacecolor="gray",
                                   markersize=8, label=f"reg: {reg}", linestyle="None"))

fig.legend(handles=legend_elements, loc="lower center", ncol=5, fontsize=9,
           bbox_to_anchor=(0.5, -0.01))
plt.tight_layout(rect=[0, 0.04, 1, 0.95])
fig.savefig(PLOT_DIR / "faceted_avalanche_vs_lr.png", dpi=150, bbox_inches="tight")
print(f"  Saved {PLOT_DIR / 'faceted_avalanche_vs_lr.png'}")
plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 6: Input parameter summary table as figure
# ──────────────────────────────────────────────────────────────────────────────
print("Plotting Figure 6: Parameter summary table...")

fig, ax = plt.subplots(figsize=(14, 8))
ax.axis("off")
ax.set_title(
    "Snellius Sweep — Input Parameter Summary\n"
    "500 neurons, 3000 Latin Hypercube samples per config, "
    "lr=fr along diagonal [0.00001, 0.1]",
    fontsize=13, fontweight="bold", pad=20,
)

table_data = []
for _, r in df_summary_sorted.iterrows():
    table_data.append([
        int(r["firing_count"]),
        f"{r['firing_frac']:.2f}",
        f"{r['leak_rate']:.2f}",
        f"{r['reset_potential']:.2f}",
        f"{r['k_prop']:.4f}",
        r["regularisation"],
        int(r["n_samples"]),
        f"{r['mean_avg_weight']:.4f}",
        f"{r['mean_avg_firing']:.1f}",
        f"{r['mean_n_avalanches']:.0f}",
        f"{r['frac_with_avalanches']:.1%}",
    ])

col_labels = [
    "Firing\nCount", "Firing\nFrac", "Leak\nRate", "Reset\nPotential",
    "k_prop", "Regularisation", "N\nSamples",
    "Mean\nAvg Weight", "Mean\nAvg Firing", "Mean\nN Aval", "% With\nAval",
]

table = ax.table(
    cellText=table_data,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(6)
table.scale(1, 1.2)

# Color header
for j in range(len(col_labels)):
    cell = table[0, j]
    cell.set_facecolor("#4472C4")
    cell.set_text_props(color="white", fontweight="bold")

# Alternate row colors
for i in range(1, len(table_data) + 1):
    for j in range(len(col_labels)):
        cell = table[i, j]
        if i % 2 == 0:
            cell.set_facecolor("#D9E2F3")
        else:
            cell.set_facecolor("#FFFFFF")

plt.tight_layout()
fig.savefig(PLOT_DIR / "parameter_summary_table.png", dpi=150, bbox_inches="tight")
print(f"  Saved {PLOT_DIR / 'parameter_summary_table.png'}")
plt.close(fig)

print(f"\nAll plots saved to {PLOT_DIR}/")
