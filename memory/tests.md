# Test Coverage Report

**Last Updated**: 2026-01-22
**Total Tests**: 212
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
| `test_event_bus.py` | Subscribe, emit, unsubscribe, multiple handlers, type filtering | PASS |
| `test_tui_logger.py` | TUI output format, step event handling | PASS |
| `test_recorder.py` | CSV format, NPZ structure | PASS |

### Integration Tests (test/integration/)

| Module | Tests | Status |
|--------|-------|--------|
| `test_simulation_core.py` | Simulation with Network + NeuronState, dynamics, determinism | PASS |
| `test_learning_simulation.py` | Hebbian learning integration, LTP/LTD in simulation | PASS |
| `test_cli.py` | CLI config loading, headless mode, flags, error handling | PASS |
| `test_simulation_events.py` | Simulation emits StepEvent/ResetEvent, event bus integration | PASS |
| `test_headless_run.py` | Full simulation with TUI + recorder | PASS |

### Property-Based Tests (test/property/)

| Module | Tests | Status |
|--------|-------|--------|
| `test_network_props.py` | Positions in bounds, link matrix invariants | PASS |
| `test_simulation_props.py` | Step count monotonicity, state invariants | PASS |
| `test_hebbian_props.py` | Weight bounds invariants, STDP properties | PASS |
| `test_config_props.py` | Valid configs produce valid simulations | PASS |
| `test_event_bus_props.py` | All handlers receive events, type isolation | PASS |
| `test_recorder_props.py` | Recorded steps match simulation steps | PASS |

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
│   ├── test_config_loader.py
│   ├── test_event_bus.py
│   ├── test_tui_logger.py
│   └── test_recorder.py
├── integration/
│   ├── __init__.py
│   ├── test_simulation_core.py
│   ├── test_learning_simulation.py
│   ├── test_cli.py
│   ├── test_simulation_events.py
│   └── test_headless_run.py
└── property/
    ├── __init__.py
    ├── test_network_props.py
    ├── test_simulation_props.py
    ├── test_hebbian_props.py
    ├── test_config_props.py
    ├── test_event_bus_props.py
    └── test_recorder_props.py
```

## Running Tests

```bash
make test          # Run all tests
make test-fast     # Run tests in parallel
make test-cov      # Run with coverage report
```

## Phase 7: TUI Logger + Output Recorder

### test/unit/test_tui_logger.py (10 tests)
- **TestTUILoggerCreation**: default stream, custom stream
- **TestTUILoggerOutput**: format, padding, large steps, decimals, zeros
- **TestTUILoggerMultipleSteps**: sequential logging
- **TestTUILoggerVerbosity**: silent mode, verbose mode

### test/unit/test_recorder.py (15 tests)
- **TestRecorderCreation**: directory, auto-create, intervals
- **TestRecorderCSV**: file creation, header, data format, appending
- **TestRecorderNPZ**: snapshot creation, weight matrix, time step, naming
- **TestRecorderAutoSnapshot**: interval saving
- **TestRecorderClose**: flush behavior

### test/integration/test_headless_run.py (5 tests)
- **TestHeadlessSimulationWithTUI**: logs all steps
- **TestHeadlessSimulationWithRecorder**: captures all steps, saves snapshots
- **TestHeadlessSimulationWithBoth**: combined TUI and recorder
- **TestRecorderDataIntegrity**: firing count matches

### test/property/test_recorder_props.py (5 tests)
- **TestRecorderStepCountProperty**: CSV row count matches steps
- **TestRecorderTimeStepOrdering**: time steps sequential
- **TestRecorderSnapshotProperty**: snapshot count matches interval
- **TestRecorderDataPreservation**: values preserved
- **TestRecorderWeightMatrixPreservation**: weight matrix preserved
