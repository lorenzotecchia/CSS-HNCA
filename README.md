# CSS-HNCA

[![Tests](https://img.shields.io/badge/tests-212%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.12+-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

**Complex System Simulation for Hebbian Neural Cellular Automaton**

A computational neuroscience framework for simulating self-organized criticality (SOC) in spiking neural networks with spike-timing-dependent plasticity (STDP).

## Overview

CSS-HNCA models networks of leaky integrate-and-fire neurons with Hebbian learning to study how neural systems naturally evolve toward critical dynamics. The simulation implements biologically-inspired plasticity rules (STDP with LTP/LTD) and stability mechanisms (weight decay, Oja normalization) to explore parameter regimes that produce first order phase transitions characteristic of criticality.

Key features:
- **300+ spiking neurons** with 3D spatial organization
- **STDP learning** with configurable LTP/LTD rates
- **LIF dynamics** with membrane potential and leak
- **Avalanche detection** for SOC metrics (power-law slope, branching ratio)
- **Backend abstraction** for CPU (NumPy) or GPU (JAX) execution

## Quick Start

```bash
# Setup
make venv && source .venv/bin/activate && make install

# Run with visualization
python main.py

# Run headless (HPC)
python main.py --headless --steps 10000 -v
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_neurons` | 300 | Network size |
| `threshold` | 0.4 | Firing threshold |
| `learning_rate` | 0.01 | STDP potentiation (LTP) |
| `forgetting_rate` | 0.01 | STDP depression (LTD) |
| `leak_rate` | 0.08 | Membrane potential decay |
| `decay_alpha` | 0.0005 | Baseline weight decay |
| `oja_alpha` | 0.002 | Activity-dependent decay |

See [docs/configuration.md](docs/configuration.md) for all parameters.

## Examples

```bash
# Basic simulation
python examples/basic_simulation.py

# Real-time visualization
python examples/realtime_visualization.py

# Avalanche analysis
python examples/avalanche_analysis.py

# Parameter sweep for critical dynamics
python scripts/parameter_sweep.py
```

## Documentation

- [Introduction](docs/introduction.md) - Scientific background and motivation
- [Methods](docs/methods.md) - Network model, STDP, LIF dynamics
- [Architecture](docs/architecture.md) - Code structure and design decisions
- [Configuration](docs/configuration.md) - All parameters explained
- [Usage](docs/usage.md) - Installation, CLI, GPU acceleration

## Testing

```bash
make test          # Run all 212 tests
make test-fast     # Parallel execution
make test-cov      # With coverage report
```

## License

MIT License - see [LICENSE](LICENSE)
