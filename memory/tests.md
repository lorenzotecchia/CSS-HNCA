# Test Coverage Report

**Last Updated**: 2026-01-22
**Total Tests**: 162
**Status**: ALL PASSING

## Test Summary by Category

### Unit Tests (test/unit/)

| Module | Tests | Status |
|--------|-------|--------|
| `test_network.py` | Network creation, positions, links, weights, reproducibility | PASS |
| `test_neuron_state.py` | NeuronState creation, firing arrays, thresholds, updates | PASS |
| `test_simulation.py` | SimulationState, creation, start/pause/step/reset, metrics | PASS |
| `test_hebbian.py` | LTP/LTD, weight bounds, STDP timing effects | PASS |
| `test_config_loader.py` | TOML parsing, validation, defaults, error handling | PASS |

### Integration Tests (test/integration/)

| Module | Tests | Status |
|--------|-------|--------|
| `test_simulation_core.py` | Simulation with Network + NeuronState, dynamics, determinism | PASS |
| `test_learning_simulation.py` | Hebbian learning integration, LTP/LTD in simulation | PASS |
| `test_cli.py` | CLI config loading, headless mode, flags, error handling | PASS |

### Property-Based Tests (test/property/)

| Module | Tests | Status |
|--------|-------|--------|
| `test_network_props.py` | Positions in bounds, link matrix invariants | PASS |
| `test_simulation_props.py` | Step count monotonicity, state invariants | PASS |
| `test_hebbian_props.py` | Weight bounds invariants, STDP properties | PASS |
| `test_config_props.py` | Valid configs produce valid simulations | PASS |

## Test Files Structure

```
test/
├── __init__.py
├── conftest.py              # Shared fixtures
├── unit/
│   ├── __init__.py
│   ├── test_network.py
│   ├── test_neuron_state.py
│   ├── test_simulation.py
│   ├── test_hebbian.py
│   └── test_config_loader.py
├── integration/
│   ├── __init__.py
│   ├── test_simulation_core.py
│   ├── test_learning_simulation.py
│   └── test_cli.py
└── property/
    ├── __init__.py
    ├── test_network_props.py
    ├── test_simulation_props.py
    ├── test_hebbian_props.py
    └── test_config_props.py
```

## Running Tests

```bash
make test          # Run all tests
make test-fast     # Run tests in parallel
make test-cov      # Run with coverage report
```
