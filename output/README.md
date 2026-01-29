# Output Directory

Simulation output files are stored here.

## File Types

### CSV Time Series
- `timeseries.csv` - Metrics per simulation step
- Columns: time_step, firing_count, avg_weight

### NPZ Snapshots
- `snapshot_XXXXXX.npz` - Full weight matrix at step XXXXXX
- Contains: weight_matrix, positions, link_matrix

## Why Files Are Gitignored

Simulation runs produce extensive output (millions of rows for long runs). Only the `sample/` subdirectory is tracked to demonstrate expected format.

## Reproducing Results

Results are reproducible using the same:
1. Configuration file
2. Random seed
3. Number of steps

Example:
```bash
python main.py -c config/default.toml --headless --steps 10000
```

## Sample Output

See `sample/` for example output demonstrating expected format and values.
