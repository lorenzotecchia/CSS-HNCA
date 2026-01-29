# Results

## Output Format

### Time Series CSV

Location: `output/timeseries.csv`

| Column | Type | Description |
|--------|------|-------------|
| time_step | int | Simulation step |
| firing_count | int | Neurons firing this step |
| avg_weight | float | Mean weight across all synapses |

### Sample Output

```csv
time_step,firing_count,avg_weight
0,30,0.060000
1,42,0.060123
2,38,0.060089
...
```

## Interpreting Results

### Firing Dynamics

- **Healthy range**: 5-15% of neurons firing per step
- **Saturated**: >50% indicates runaway excitation
- **Dead**: <1% indicates insufficient activity

### Weight Evolution

- **Stable**: Weights fluctuate around a mean value
- **Runaway growth**: Weights approaching w_max suggests insufficient decay
- **Collapse**: Weights approaching w_min suggests excessive decay

### Avalanche Metrics

#### Power-Law Slope

The distribution of avalanche sizes should follow a power law:
```
P(size=s) ~ s^(-τ)
```

| Slope (τ) | Interpretation |
|-----------|----------------|
| ≈ -1.5 | Critical (optimal) |
| > -1.5 | Subcritical (activity dies out) |
| < -1.5 | Supercritical (runaway activity) |

#### Branching Ratio

The average number of neurons activated by each firing neuron:
```
σ = mean(firing(t+1) / firing(t))
```

| Ratio (σ) | Interpretation |
|-----------|----------------|
| ≈ 1.0 | Critical (activity propagates sustainably) |
| < 1.0 | Subcritical (activity decays) |
| > 1.0 | Supercritical (activity explodes) |

## Visualization

### Real-Time Plots

The matplotlib view shows:
1. **Firing count over time**: Should show variable, non-saturated activity
2. **Average weight over time**: Should stabilize after initial transient
3. **Weight histogram**: Should show characteristic distribution

### Avalanche Analysis

Run `examples/avalanche_analysis.py` to visualize:
- Avalanche size distribution (log-log plot)
- Avalanche duration distribution
- Branching ratio over time

## Reproducing Results

All simulations are reproducible via the `seed` parameter:

```toml
seed = 42
```

To reproduce a specific experiment:
1. Use the same configuration file
2. Use the same seed value
3. Run the same number of steps

## Sample Results

The `output/sample/` directory contains example output from a typical run demonstrating expected format and values.
