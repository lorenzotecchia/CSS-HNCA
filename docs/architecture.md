# Architecture

## Overview

CSS-HNCA follows a modular architecture separating core simulation logic from visualization, configuration, and output concerns. The design enables:
- Backend abstraction (NumPy for CPU, JAX for GPU)
- Event-driven communication between components
- Headless operation for HPC environments

## Module Structure

```
src/
├── core/                    # Simulation engine
│   ├── network.py           # Network topology and weights
│   ├── neuron_state.py      # Neuron firing states and membrane potential
│   ├── simulation.py        # Simulation orchestrator
│   ├── backend.py           # NumPy backend implementation
│   └── backend_jax.py       # JAX backend (optional GPU)
├── learning/                # Plasticity rules
│   └── hebbian.py           # STDP with weight decay
├── config/                  # Configuration
│   └── loader.py            # TOML parsing and validation
├── events/                  # Event system
│   ├── bus.py               # Typed pub/sub event bus
│   └── avalanche.py         # Avalanche detection
├── visualization/           # Display components
│   ├── matplotlib_view.py   # Real-time analytics plots
│   ├── tui_logger.py        # Terminal output
│   └── avalanche_view.py    # Avalanche metrics display
└── output/                  # Data persistence
    └── recorder.py          # CSV and NPZ output
```

## Core Components

### Network

Stores the structural and weighted connectivity:
- `positions`: (n, 3) array of 3D coordinates
- `link_matrix`: (n, n) boolean array of connections
- `weight_matrix`: (n, n) float array of synaptic weights
- `neuron_types`: (n,) boolean array (True=excitatory)

### NeuronState

Tracks dynamic neuron variables:
- `firing`: Current binary firing state
- `firing_prev`: Previous state (for STDP)
- `membrane_potential`: LIF membrane voltage

### Simulation

Orchestrates the simulation loop:
1. Compute input signal: `v = W^T · s`
2. Update neuron states via LIF dynamics
3. Apply STDP learning rules
4. Emit events for visualization/recording

## Backend Abstraction

The `ArrayBackend` protocol defines array operations:
- `zeros`, `ones`, `random_uniform`
- `matmul`, `transpose`
- `where`, `sum`, `mean`
- `clip`, `maximum`, `minimum`

Two implementations:
- `NumPyBackend`: Default, works everywhere
- `JAXBackend`: Optional GPU acceleration, falls back to NumPy if JAX unavailable

## Event System

Components communicate via typed events:
- `StepEvent`: Emitted each simulation step with metrics
- `ResetEvent`: Emitted when simulation resets

Subscribers register handlers for specific event types, enabling loose coupling.

## Data Flow

```
Configuration (TOML)
        ↓
    Loader → SimulationConfig
        ↓
    Network.create_random()
    NeuronState.create()
    HebbianLearner()
        ↓
    Simulation
        ↓ step()
    ├── input = W^T · firing
    ├── state.update_firing(input)
    ├── learner.apply(network, state)
    └── event_bus.emit(StepEvent)
              ↓
    ├── TUILogger (terminal output)
    ├── Recorder (CSV/NPZ files)
    ├── MatplotlibView (plots)
    └── AvalancheDetector (SOC metrics)
```

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Dimensionality | 3D spatial | Biologically realistic neural organization |
| Connectivity | Directed, distance-based | STDP requires causal pre→post; radius mimics local circuits |
| Backend | NumPy default, JAX optional | Portable consumer→HPC; automatic GPU when available |
| Events | Typed pub/sub | Decouples core from visualization |
| State | Immutable between steps | Simplifies reasoning about causality |
