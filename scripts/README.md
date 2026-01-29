# Analysis Scripts

Tools for parameter exploration and visualization of simulation output.

## Parameter Sweeps

### parameter_sweep.py

Comprehensive parameter sweep for finding critical dynamics. Explores multiple parameters and measures SOC metrics.

```bash
python scripts/parameter_sweep.py
```

Output: `output/soc_sweep_results.csv`

### learning_rate_sweep.py

2D Latin Hypercube Sampling sweep over learning and forgetting rates.

```bash
python scripts/learning_rate_sweep.py
```

Output: `output/learning_forgetting_sweep.csv`

### diagonal_sweep.py

1D sweep where learning_rate equals forgetting_rate (diagonal exploration).

```bash
python scripts/diagonal_sweep.py
```

Output: `output/learning_forgetting_sweep.csv`

## Visualization

### plot_timeseries.py

Plot firing count and weight evolution from CSV output.

```bash
python scripts/plot_timeseries.py output/timeseries.csv
```

### plot_weight_heatmap.py

Visualize weight matrix as heatmap.

```bash
python scripts/plot_weight_heatmap.py
```
