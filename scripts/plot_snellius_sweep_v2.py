#!/usr/bin/env python3
"""Plot Snellius supercomputer parameter sweep v2 results (job 18823540).

Loads all sweep_v2_config_*.csv files and produces multi-panel figures
emphasising how the input parameters (firing_count, leak_rate/reset_potential,
k_prop, decay_alpha, oja_alpha) influence simulation outputs -- particularly
avalanche dynamics.

Run with: python scripts/plot_snellius_sweep_v2.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

DATA_DIR = Path("output_snellius_v2")
PLOT_DIR = Path("output") / "plots_snellius_v2"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load all sweep v2 CSVs ──────────────────────────────────────────────────
csvs = sorted(DATA_DIR.glob("sweep_v2_config_*.csv"))
print(f"Found {len(csvs)} sweep v2 config files")

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
df_all["reg_label"] = df_all.apply(
    lambda r: f"decay={r['decay_alpha']:.4f}, oja={r['oja_alpha']:.3f}", axis=1
)

# Unique parameter values
firing_counts = sorted(df_all["firing_count"].unique())
firing_fracs = sorted(df_all["firing_frac"].unique())
leak_reset_labels = sorted(df_all["leak_reset"].unique())
k_props = sorted(df_all["k_prop"].unique())
decay_alphas = sorted(df_all["decay_alpha"].unique())
oja_alphas = sorted(df_all["oja_alpha"].unique())

print(f"\nParameter space:")
print(f"  firing_count: {firing_counts}")
print(f"  leak/reset: {leak_reset_labels}")
print(f"  k_prop: {k_props}")
print(f"  decay_alpha: {decay_alphas}")
print(f"  oja_alpha: {oja_alphas}")
print(f"\nAvalanche stats: {(df_all['n_avalanches'] > 0).mean()*100:.1f}% of samples have avalanches")

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
        "reg_label": grp["reg_label"].iloc[0],
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
# FIGURE 1: Leak/Reset effect — avalanche metrics by leak/reset pair
# ──────────────────────────────────────────────────────────────────────────────
print("\nPlotting Figure 1: Leak/Reset effect on avalanche metrics...")

metrics = [
    ("n_avalanches", "Number of Avalanches"),
    ("avg_avalanche_size", "Avg Avalanche Size"),
    ("avg_avalanche_duration", "Avg Avalanche Duration"),
    ("avg_weight", "Avg Synaptic Weight"),
]

fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 14), sharex=True)
fig.suptitle(
    "Effect of Leak/Reset Configuration on Simulation Metrics\n"
    "(each box aggregates all samples sharing that leak/reset pair)",
    fontsize=14, fontweight="bold",
)

for row_idx, (metric_col, metric_label) in enumerate(metrics):
    ax = axes[row_idx]
    data_groups = []
    tick_labels = []
    for lr_lab in leak_reset_labels:
        subset = df_all[df_all["leak_reset"] == lr_lab][metric_col].dropna()
        data_groups.append(subset.values)
        tick_labels.append(lr_lab)

    bp = ax.boxplot(data_groups, patch_artist=True, showfliers=False,
                    medianprops=dict(color="red", linewidth=1.5))
    colors_box = plt.cm.viridis(np.linspace(0.2, 0.8, len(data_groups)))
    for patch, c in zip(bp["boxes"], colors_box):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)

    ax.set_xticks(range(1, len(tick_labels) + 1))
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_ylabel(metric_label)
    ax.grid(axis="y", alpha=0.3)
    if metric_col in ("avg_avalanche_size", "n_avalanches"):
        ax.set_yscale("symlog", linthresh=1)

axes[-1].set_xlabel("Leak/Reset Configuration")
plt.tight_layout()
fig.savefig(PLOT_DIR / "leak_reset_effect.png", dpi=150, bbox_inches="tight")
print(f"  Saved {PLOT_DIR / 'leak_reset_effect.png'}")
plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 2: Regularisation onset — heatmap of avalanche activity
#            rows: (decay_alpha, oja_alpha), cols: leak_reset + firing_count
# ──────────────────────────────────────────────────────────────────────────────
print("Plotting Figure 2: Regularisation onset heatmap...")

pivot_data = df_summary.copy()
pivot_data["row_label"] = pivot_data.apply(
    lambda r: f"decay={r['decay_alpha']:.4f}\noja={r['oja_alpha']:.3f}", axis=1
)
pivot_data["col_label"] = pivot_data.apply(
    lambda r: f"fc={int(r['firing_count'])}\n{r['leak_reset']}", axis=1
)

heatmap_metrics = [
    ("mean_avg_aval_size", "Mean Avg Avalanche Size"),
    ("mean_n_avalanches", "Mean Number of Avalanches"),
    ("frac_with_avalanches", "Fraction with Avalanches"),
    ("mean_avg_weight", "Mean Avg Weight"),
]

fig, axes = plt.subplots(2, 2, figsize=(20, 14))
fig.suptitle(
    "Regularisation Onset Heatmaps\n"
    "(rows: decay/oja pair, cols: firing_count + leak/reset)",
    fontsize=13, fontweight="bold",
)

for idx, (metric_col, metric_title) in enumerate(heatmap_metrics):
    ax = axes[idx // 2, idx % 2]

    pivot_table = pivot_data.pivot_table(
        values=metric_col, index="row_label", columns="col_label", aggfunc="mean"
    )
    pivot_table = pivot_table.reindex(
        sorted(pivot_table.columns), axis=1
    ).reindex(sorted(pivot_table.index))

    im = ax.imshow(pivot_table.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(np.arange(len(pivot_table.columns)))
    ax.set_yticks(np.arange(len(pivot_table.index)))
    ax.set_xticklabels(pivot_table.columns, fontsize=7, rotation=45, ha="right")
    ax.set_yticklabels(pivot_table.index, fontsize=8)
    ax.set_title(metric_title, fontsize=11)

    # Annotate cells
    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            val = pivot_table.values[i, j]
            if not np.isnan(val):
                max_val = np.nanmax(pivot_table.values)
                text_color = "white" if max_val > 0 and val > max_val * 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=text_color)

    plt.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
fig.savefig(PLOT_DIR / "regularisation_onset_heatmap.png", dpi=150, bbox_inches="tight")
print(f"  Saved {PLOT_DIR / 'regularisation_onset_heatmap.png'}")
plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 3: Learning rate vs metrics, faceted by k_prop
# ──────────────────────────────────────────────────────────────────────────────
print("Plotting Figure 3: LR sweeps faceted by k_prop...")

metric_list = [
    ("avg_weight", "Avg Weight"),
    ("avg_firing_count", "Avg Firing Count"),
    ("avg_avalanche_size", "Avg Avalanche Size"),
    ("n_avalanches", "Number of Avalanches"),
]

fig, axes = plt.subplots(len(k_props), len(metric_list), figsize=(20, 4 * len(k_props)))
fig.suptitle(
    "Metrics vs Learning Rate, faceted by k_prop\n"
    "Colour = leak/reset pair, Shape = firing_count",
    fontsize=13, fontweight="bold",
)

lr_cmap = {lab: c for lab, c in zip(leak_reset_labels, ["tab:blue", "tab:orange"])}
fc_markers = {fc: m for fc, m in zip(firing_counts, ["o", "s"])}

import warnings

# Subsample for readability
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    df_sub = df_all.groupby("config", group_keys=False).apply(
        lambda g: g.sample(min(100, len(g)), random_state=42)
    ).reset_index(drop=True)

for i, kp in enumerate(k_props):
    kp_data = df_sub[df_sub["k_prop"] == kp]
    for j, (metric_col, metric_label) in enumerate(metric_list):
        ax = axes[i, j]

        for lr_lab in leak_reset_labels:
            for fc in firing_counts:
                sub = kp_data[(kp_data["leak_reset"] == lr_lab) & (kp_data["firing_count"] == fc)]
                if len(sub) == 0:
                    continue
                ax.scatter(
                    sub["learning_rate"], sub[metric_col],
                    c=lr_cmap[lr_lab], marker=fc_markers[fc],
                    s=6, alpha=0.2, edgecolors="none",
                )

        ax.grid(True, alpha=0.3)
        if metric_col in ("avg_avalanche_size", "n_avalanches"):
            ax.set_yscale("symlog", linthresh=1)
        if i == 0:
            ax.set_title(metric_label, fontsize=10, fontweight="bold")
        if j == 0:
            ax.set_ylabel(f"k={kp:.3f}\n{metric_label}", fontsize=8)
        if i == len(k_props) - 1:
            ax.set_xlabel("Learning Rate", fontsize=8)

# Legend
legend_elements = []
for lab, c in lr_cmap.items():
    legend_elements.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                                   markersize=8, label=lab))
for fc, m in fc_markers.items():
    legend_elements.append(Line2D([0], [0], marker=m, color="gray", markerfacecolor="gray",
                                   markersize=8, label=f"firing={fc}", linestyle="None"))

fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=8,
           bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=[0, 0.04, 1, 0.96])
fig.savefig(PLOT_DIR / "lr_vs_metrics_by_kprop.png", dpi=150, bbox_inches="tight")
print(f"  Saved {PLOT_DIR / 'lr_vs_metrics_by_kprop.png'}")
plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 4: Decay x Oja interaction — faceted heatmap by leak/reset
# ──────────────────────────────────────────────────────────────────────────────
print("Plotting Figure 4: Decay x Oja interaction heatmaps per leak/reset pair...")

fig, axes = plt.subplots(len(leak_reset_labels), 2, figsize=(14, 4 * len(leak_reset_labels)))
fig.suptitle(
    "Avalanche Metrics: Decay Alpha x Oja Alpha\n"
    "Faceted by Leak/Reset Pair (rows), Metric (cols)",
    fontsize=13, fontweight="bold",
)

interaction_metrics = [
    ("frac_with_avalanches", "Fraction with Avalanches"),
    ("mean_avg_aval_size", "Mean Avg Avalanche Size"),
]

for i, lr_lab in enumerate(leak_reset_labels):
    lr_summary = df_summary[df_summary["leak_reset"] == lr_lab]
    for j, (metric_col, metric_label) in enumerate(interaction_metrics):
        ax = axes[i, j]

        pivot = lr_summary.pivot_table(
            values=metric_col, index="decay_alpha", columns="oja_alpha", aggfunc="mean"
        )
        pivot = pivot.reindex(sorted(pivot.columns), axis=1).reindex(sorted(pivot.index))

        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", origin="lower")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels([f"{v:.3f}" for v in pivot.columns], fontsize=8)
        ax.set_yticklabels([f"{v:.4f}" for v in pivot.index], fontsize=8)

        # Annotate
        for ii in range(len(pivot.index)):
            for jj in range(len(pivot.columns)):
                val = pivot.values[ii, jj]
                if not np.isnan(val):
                    max_val = np.nanmax(pivot.values)
                    text_color = "white" if max_val > 0 and val > max_val * 0.6 else "black"
                    ax.text(jj, ii, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color=text_color)

        plt.colorbar(im, ax=ax, shrink=0.8)
        if i == 0:
            ax.set_title(metric_label, fontsize=10, fontweight="bold")
        if j == 0:
            ax.set_ylabel(f"{lr_lab}\ndecay_alpha", fontsize=9)
        if i == len(leak_reset_labels) - 1:
            ax.set_xlabel("oja_alpha", fontsize=9)

plt.tight_layout()
fig.savefig(PLOT_DIR / "decay_oja_interaction.png", dpi=150, bbox_inches="tight")
print(f"  Saved {PLOT_DIR / 'decay_oja_interaction.png'}")
plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 5: K_prop effect — comparison across connectivity levels
# ──────────────────────────────────────────────────────────────────────────────
print("Plotting Figure 5: K_prop effect on avalanche metrics...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle(
    "Effect of Connectivity (k_prop) on Network Dynamics\n"
    "(each box aggregates all samples sharing that k_prop)",
    fontsize=14, fontweight="bold",
)

metrics_kprop = [
    ("n_avalanches", "Number of Avalanches"),
    ("avg_avalanche_size", "Avg Avalanche Size"),
    ("avg_weight", "Avg Synaptic Weight"),
    ("avg_firing_count", "Avg Firing Count"),
]

for idx, (metric_col, metric_label) in enumerate(metrics_kprop):
    ax = axes[idx // 2, idx % 2]
    data_groups = []
    tick_labels = []
    for kp in k_props:
        subset = df_all[df_all["k_prop"] == kp][metric_col].dropna()
        data_groups.append(subset.values)
        tick_labels.append(f"{kp:.3f}")

    bp = ax.boxplot(data_groups, patch_artist=True, showfliers=False,
                    medianprops=dict(color="red", linewidth=1.5))
    colors_box = plt.cm.viridis(np.linspace(0.2, 0.8, len(data_groups)))
    for patch, c in zip(bp["boxes"], colors_box):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)

    ax.set_xticks(range(1, len(tick_labels) + 1))
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("k_prop (connectivity)")
    ax.set_ylabel(metric_label)
    ax.grid(axis="y", alpha=0.3)
    if metric_col in ("avg_avalanche_size", "n_avalanches"):
        ax.set_yscale("symlog", linthresh=1)

plt.tight_layout()
fig.savefig(PLOT_DIR / "kprop_effect.png", dpi=150, bbox_inches="tight")
print(f"  Saved {PLOT_DIR / 'kprop_effect.png'}")
plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 6: Summary table
# ──────────────────────────────────────────────────────────────────────────────
print("Plotting Figure 6: Parameter summary table...")

df_summary_sorted = df_summary.sort_values(
    ["firing_count", "leak_rate", "k_prop", "decay_alpha", "oja_alpha"]
).reset_index(drop=True)

# Only show configs with any avalanche activity
df_interesting = df_summary_sorted[df_summary_sorted["frac_with_avalanches"] > 0].head(50)

if len(df_interesting) > 0:
    fig, ax = plt.subplots(figsize=(16, max(6, len(df_interesting) * 0.35)))
    ax.axis("off")
    ax.set_title(
        "Sweep v2 (job 18823540) — Configurations with Avalanche Activity\n"
        "500 neurons, 500 LHS samples per config, 2D (lr, fr) sampling",
        fontsize=13, fontweight="bold", pad=20,
    )

    table_data = []
    for _, r in df_interesting.iterrows():
        table_data.append([
            int(r["firing_count"]),
            f"{r['firing_frac']:.2f}",
            f"{r['leak_rate']:.2f}",
            f"{r['reset_potential']:.2f}",
            f"{r['k_prop']:.4f}",
            f"{r['decay_alpha']:.4f}",
            f"{r['oja_alpha']:.3f}",
            int(r["n_samples"]),
            f"{r['mean_avg_weight']:.4f}",
            f"{r['mean_avg_firing']:.1f}",
            f"{r['mean_n_avalanches']:.0f}",
            f"{r['frac_with_avalanches']:.1%}",
        ])

    col_labels = [
        "Firing\nCount", "Firing\nFrac", "Leak\nRate", "Reset\nPot",
        "k_prop", "Decay\nAlpha", "Oja\nAlpha", "N\nSamples",
        "Mean\nAvg Wt", "Mean\nAvg Fire", "Mean\nN Aval", "% With\nAval",
    ]

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
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
    fig.savefig(PLOT_DIR / "parameter_summary_table_v2.png", dpi=150, bbox_inches="tight")
    print(f"  Saved {PLOT_DIR / 'parameter_summary_table_v2.png'}")
    plt.close(fig)
else:
    print("  No configs with avalanche activity found -- skipping summary table.")

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 7: 2D Learning/Forgetting rate heatmaps
# ──────────────────────────────────────────────────────────────────────────────
print("Plotting Figure 7: 2D Learning vs Forgetting rate heatmaps...")

# Bin learning and forgetting rates for heatmap
df_all["lr_bin"] = pd.cut(df_all["learning_rate"], bins=20, labels=False)
df_all["fr_bin"] = pd.cut(df_all["forgetting_rate"], bins=20, labels=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(
    "2D Heatmap: Learning Rate vs Forgetting Rate\n"
    "(aggregated across all configurations)",
    fontsize=13, fontweight="bold",
)

# Heatmap of avalanche fraction
pivot_aval = df_all.pivot_table(
    values="n_avalanches", index="fr_bin", columns="lr_bin",
    aggfunc=lambda x: (x > 0).mean()
)
im1 = axes[0].imshow(pivot_aval.values, aspect="auto", cmap="YlOrRd", origin="lower")
axes[0].set_xlabel("Learning Rate (binned)")
axes[0].set_ylabel("Forgetting Rate (binned)")
axes[0].set_title("Fraction with Avalanches")
plt.colorbar(im1, ax=axes[0], shrink=0.8)

# Heatmap of mean avalanche count
pivot_count = df_all.pivot_table(
    values="n_avalanches", index="fr_bin", columns="lr_bin", aggfunc="mean"
)
im2 = axes[1].imshow(pivot_count.values, aspect="auto", cmap="YlOrRd", origin="lower")
axes[1].set_xlabel("Learning Rate (binned)")
axes[1].set_ylabel("Forgetting Rate (binned)")
axes[1].set_title("Mean Number of Avalanches")
plt.colorbar(im2, ax=axes[1], shrink=0.8)

plt.tight_layout()
fig.savefig(PLOT_DIR / "lr_fr_2d_heatmap.png", dpi=150, bbox_inches="tight")
print(f"  Saved {PLOT_DIR / 'lr_fr_2d_heatmap.png'}")
plt.close(fig)

print(f"\nAll plots saved to {PLOT_DIR}/")
