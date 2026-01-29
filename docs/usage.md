# Usage

## Installation

### Requirements
- Python 3.12+
- NumPy
- Optional: JAX (for GPU acceleration)

### Setup

```bash
# Create virtual environment
make venv

# Activate environment
source .venv/bin/activate

# Install dependencies
make install
```

## Running Simulations

### Interactive Mode (GUI)

```bash
# Default configuration with visualization
python main.py

# Custom configuration
python main.py -c config/custom.toml
```

### Headless Mode (HPC)

```bash
# Run without visualization
python main.py --headless --steps 10000

# With verbose terminal output
python main.py --headless --steps 10000 -v

# Save output to specific directory
python main.py --headless --steps 10000 --output output/experiment1/
```

### Keyboard Controls (Interactive Mode)

| Key | Action |
|-----|--------|
| SPACE | Step simulation |
| P | Pause/resume |
| R | Reset simulation |
| ESC | Quit |

## Examples

The `examples/` directory contains demonstration scripts:

```bash
# Basic simulation example
python examples/basic_simulation.py

# Real-time matplotlib visualization
python examples/realtime_visualization.py

# Avalanche analysis with SOC metrics
python examples/avalanche_analysis.py
```

## Analysis Scripts

The `scripts/` directory contains analysis tools:

```bash
# Parameter sweep for critical dynamics
python scripts/parameter_sweep.py

# Learning rate exploration
python scripts/learning_rate_sweep.py

# Plot simulation output
python scripts/plot_timeseries.py output/timeseries.csv

# Visualize weight matrix evolution
python scripts/plot_weight_heatmap.py
```

## Output Files

### CSV Time Series

Each row contains metrics for one time step:
- `time_step`: Simulation step number
- `firing_count`: Number of neurons firing
- `avg_weight`: Mean synaptic weight

### NPZ Snapshots

Periodic snapshots of the full weight matrix for offline analysis.

## GPU Acceleration

If JAX is installed, GPU acceleration is automatic:

```bash
# Install JAX (example for CUDA 12)
pip install jax[cuda12]

# The simulation will automatically use GPU
python main.py --headless --steps 100000
```

Verify GPU is being used:
```python
from src.core.backend import get_backend
backend = get_backend(prefer_gpu=True)
print(backend.__class__.__name__)  # JAXBackend if available
```
