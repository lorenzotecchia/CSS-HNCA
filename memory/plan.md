# Neural Cellular Automata - Architecture Design
**Date**: 2026-01-21  
**Status**: Approved (2026-01-22)

## Problem Statement
Build a neural cellular automata simulating 300+ neurons with Hebbian learning and STDP. Must work on consumer devices (NumPy) and HPC with GPU acceleration (JAX).

## Architecture Overview

```
src/
├── core/                    # Simulation engine (backend-agnostic)
│   ├── network.py           # Network: positions, Link matrix, weights
│   ├── neuron_state.py      # NeuronState: firing states
│   ├── simulation.py        # Simulation: step/reset/start/pause
│   └── backend.py           # Array backend abstraction (NumPy/JAX)
├── learning/                # Plasticity rules
│   ├── hebbian.py           # Basic Hebbian + STDP (LTP/LTD)
│   └── oja.py               # Oja rule for unsupervised learning
├── config/                  # Configuration loading
│   └── loader.py            # TOML parser → typed dataclass
├── events/                  # Event system
│   └── bus.py               # EventBus: typed events, subscribe/emit
├── visualization/           # Decoupled from core
│   ├── pygame_view.py       # 3D network rendering + connection lines
│   ├── matplotlib_view.py   # Real-time plots (firing, weights, histogram)
│   └── tui_logger.py        # Terminal output for headless HPC runs
├── output/                  # Data persistence
│   └── recorder.py          # CSV time series + NPZ snapshots
└── __init__.py
```

---

## Core Data Structures

### Network (`core/network.py`)
```python
@dataclass
class Network:
    positions: ndarray       # Shape: (N, 3) - 3D coordinates
    link_matrix: ndarray     # Shape: (N, N) - directed connectivity (bool, asymmetric)
    weight_matrix: ndarray   # Shape: (N, N) - synaptic strengths (directed)
    n_neurons: int
    radius: float
    box_size: tuple[float, float, float]
    
    @classmethod
    def create_random(cls, n_neurons: int, box_size: tuple[float, float, float],
                      radius: float, seed: int | None = None) -> "Network": ...
```

### NeuronState (`core/neuron_state.py`)
```python
@dataclass
class NeuronState:
    firing: ndarray          # Shape: (N,) - current firing state (bool)
    firing_prev: ndarray     # Shape: (N,) - previous step (for STDP)
    threshold: float         # γ
```

### Simulation (`core/simulation.py`)
```python
class SimulationState(Enum):
    STOPPED = auto()
    RUNNING = auto()
    PAUSED = auto()

@dataclass
class Simulation:
    network: Network
    state: NeuronState
    time_step: int
    sim_state: SimulationState
    learning_rate: float      # l (LTP)
    forgetting_rate: float    # f (LTD)
    
    def step(self) -> None: ...
    def reset(self, seed: int | None = None) -> None: ...
    def start(self) -> None: ...
    def pause(self) -> None: ...
```

---

## Event System (`events/bus.py`)

```python
@dataclass
class StepEvent:
    time_step: int
    firing_count: int
    avg_weight: float

@dataclass  
class ResetEvent:
    seed: int | None

class EventBus:
    def subscribe[T](self, event_type: type[T], handler: Callable[[T], None]) -> None: ...
    def emit[T](self, event: T) -> None: ...
```

---

## Backend Abstraction (`core/backend.py`)

```python
class ArrayBackend(Protocol):
    def zeros(self, shape: tuple[int, ...], dtype: type) -> ndarray: ...
    def random_uniform(self, low: float, high: float, shape: tuple[int, ...]) -> ndarray: ...
    def matmul(self, a: ndarray, b: ndarray) -> ndarray: ...
    def where(self, condition: ndarray, x: ndarray, y: ndarray) -> ndarray: ...
    def sum(self, a: ndarray, axis: int | None = None) -> ndarray | float: ...
    def to_numpy(self, a: ndarray) -> ndarray: ...

class NumPyBackend: ...  # Default CPU
class JAXBackend: ...    # Optional GPU

def get_backend(prefer_gpu: bool = False) -> ArrayBackend: ...
```

---

## Configuration (`config/loader.py`)

```python
@dataclass(frozen=True)
class NetworkConfig:
    n_neurons: int
    box_size: tuple[float, float, float]
    radius: float
    initial_weight: float
    weight_min: float
    weight_max: float
    initial_firing_fraction: float  # Fraction of neurons firing at t=0

@dataclass(frozen=True)  
class LearningConfig:
    threshold: float         # γ
    learning_rate: float     # l (LTP)
    forgetting_rate: float   # f (LTD)
    decay_alpha: float       # α for Oja/decay

@dataclass(frozen=True)
class VisualizationConfig:
    pygame_enabled: bool
    matplotlib_enabled: bool
    window_width: int
    window_height: int
    fps: int

@dataclass(frozen=True)
class SimulationConfig:
    network: NetworkConfig
    learning: LearningConfig
    visualization: VisualizationConfig
    seed: int | None
```

### Example TOML (`config/default.toml`)
```toml
seed = 42

[network]
n_neurons = 300
box_size = [10.0, 10.0, 10.0]
radius = 2.5
initial_weight = 0.1
weight_min = 0.0
weight_max = 1.0
initial_firing_fraction = 0.1  # 10% of neurons fire at t=0

[learning]
threshold = 0.5
learning_rate = 0.01
forgetting_rate = 0.005
decay_alpha = 0.001

[visualization]
pygame_enabled = true
matplotlib_enabled = true
window_width = 800
window_height = 600
fps = 30
```

---

## Visualization

### PygameNetworkView
- 3D→2D projection with rotation controls
- Neurons as circles (red=firing, blue=not firing)
- **Connection lines**: thickness/opacity proportional to weight strength
- Keyboard controls: SPACE=step, R=reset, P=pause, ESC=quit

### MatplotlibAnalyticsView
- Firing count over time (line plot)
- Average weight over time (line plot)
- **Weight distribution histogram** (updates each step)
- Weight matrix heatmap (optional)

### TUILogger (headless)
- Terminal output: `[t=00042] firing: 23 | avg_weight: 0.1523`
- Works on HPC without display

---

## CLI Interface

```bash
python main.py                           # Default config, full visualization
python main.py -c config/custom.toml     # Custom config
python main.py --headless --steps 10000  # HPC batch run
python main.py -v                        # Verbose TUI output
```

---

## Output Recording

- **CSV**: `output/timeseries.csv` - metrics per step
- **NPZ**: `output/snapshot_XXXXXX.npz` - full weight matrices at intervals

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Dimensionality | 3D volume | Biologically realistic |
| Connectivity | **Directed (asymmetric)**, all within radius | A→B can exist without B→A; matches STDP semantics |
| Initial firing | **Random fraction** | Configurable `initial_firing_fraction` in config |
| Weight initialization | **Uniform constant** | All weights start at `initial_weight` value |
| Backend | NumPy default, JAX optional | Portable: consumer → HPC |
| Events | Typed event bus | Decouples core from viz |
| Config | TOML → dataclass | Type-safe, human-readable |
| Typing | Python 3.12 built-in + numpy.ndarray | Native, no extra packages |

---

## Implementation Phases (Strict TDD)

Each phase follows RED → GREEN → REFACTOR:
1. **RED**: Write failing tests first
2. **GREEN**: Implement minimal code to pass
3. **REFACTOR**: Clean up while keeping tests green

### Test Organization
```
test/
├── unit/           # Fast, isolated, mocked dependencies
├── integration/    # Components working together
├── property/       # Hypothesis-based invariant tests
└── conftest.py     # Shared fixtures
```

---

### Phase 1: Core Skeleton ✅ COMPLETE
**Modules**: `core/network.py`, `core/neuron_state.py`, `core/simulation.py`

- [x] **RED**: Write failing unit tests
  - [x] `test/unit/test_network.py` - positions in bounds, link matrix shape, weight initialization
  - [x] `test/unit/test_neuron_state.py` - firing array shape, threshold validation
  - [x] `test/unit/test_simulation.py` - state enum transitions, step increment
- [x] **RED**: Write failing integration tests
  - [x] `test/integration/test_simulation_core.py` - Simulation uses Network + NeuronState correctly
- [x] **RED**: Write failing property tests
  - [x] `test/property/test_network_props.py` - positions always in bounds, link matrix matches distance
  - [x] `test/property/test_simulation_props.py` - step count always increases
- [x] **GREEN**: Implement `core/network.py`
- [x] **GREEN**: Implement `core/neuron_state.py`
- [x] **GREEN**: Implement `core/simulation.py`
- [x] **REFACTOR**: Clean up core module

---

### Phase 2: Config Loader and CLI ✅ COMPLETE
**Modules**: `config/loader.py`, `main.py` CLI

- [x] **RED**: Write failing unit tests
  - [x] `test/unit/test_config_loader.py` - TOML parsing, validation errors, defaults
- [x] **RED**: Write failing integration tests
  - [x] `test/integration/test_cli.py` - CLI loads config and creates Simulation
- [x] **RED**: Write failing property tests
  - [x] `test/property/test_config_props.py` - any valid config produces valid Simulation
- [x] **GREEN**: Implement `config/loader.py`
- [x] **GREEN**: Implement CLI in `main.py`
- [x] **REFACTOR**: Clean up config module

---

### Phase 3: Event Bus
**Modules**: `events/bus.py`

- [ ] **RED**: Write failing unit tests
  - [ ] `test/unit/test_event_bus.py` - subscribe, emit, multiple handlers, type filtering
- [ ] **RED**: Write failing integration tests
  - [ ] `test/integration/test_simulation_events.py` - Simulation emits events on step/reset
- [ ] **RED**: Write failing property tests
  - [ ] `test/property/test_event_bus_props.py` - all subscribed handlers receive events
- [ ] **GREEN**: Implement `events/bus.py`
- [ ] **GREEN**: Integrate events into `core/simulation.py`
- [ ] **REFACTOR**: Clean up events module

---

### Phase 4: Pygame Visualization
**Modules**: `visualization/pygame_view.py`

> **Note**: Visualization tests skipped (tested by running)

- [ ] Implement `visualization/pygame_view.py`
- [ ] Manual testing: verify 3D rendering, controls work

---

### Phase 5: Hebbian Learning + STDP ✅ COMPLETE
**Modules**: `learning/hebbian.py`

- [x] **RED**: Write failing unit tests
  - [x] `test/unit/test_hebbian.py` - LTP increases weights, LTD decreases, bounds respected
- [x] **RED**: Write failing integration tests
  - [x] `test/integration/test_learning_simulation.py` - Simulation applies learning each step
- [x] **RED**: Write failing property tests
  - [x] `test/property/test_hebbian_props.py` - weights always in [min, max], STDP timing effects
- [x] **GREEN**: Implement `learning/hebbian.py`
- [x] **GREEN**: Integrate learning into `core/simulation.py`
- [x] **REFACTOR**: Clean up learning module

---

### Phase 6: Matplotlib Analytics
**Modules**: `visualization/matplotlib_view.py`

> **Note**: Visualization tests skipped (tested by running)

- [ ] Implement `visualization/matplotlib_view.py`
- [ ] Manual testing: verify plots update correctly

---

### Phase 7: TUI Logger + Output Recorder
**Modules**: `visualization/tui_logger.py`, `output/recorder.py`

- [ ] **RED**: Write failing unit tests
  - [ ] `test/unit/test_tui_logger.py` - output format, handles step events
  - [ ] `test/unit/test_recorder.py` - CSV format, NPZ structure
- [ ] **RED**: Write failing integration tests
  - [ ] `test/integration/test_headless_run.py` - full simulation with TUI + recorder
- [ ] **RED**: Write failing property tests
  - [ ] `test/property/test_recorder_props.py` - recorded steps match simulation steps
- [ ] **GREEN**: Implement `visualization/tui_logger.py`
- [ ] **GREEN**: Implement `output/recorder.py`
- [ ] **REFACTOR**: Clean up output module

---

### Phase 8: Backend Abstraction (JAX)
**Modules**: `core/backend.py`

- [ ] **RED**: Write failing unit tests
  - [ ] `test/unit/test_backend.py` - NumPy backend operations
- [ ] **RED**: Write failing integration tests
  - [ ] `test/integration/test_backend_simulation.py` - Simulation works with NumPy backend
- [ ] **RED**: Write failing property tests
  - [ ] `test/property/test_backend_props.py` - NumPy and JAX produce equivalent results
- [ ] **GREEN**: Implement `core/backend.py` (NumPy)
- [ ] **GREEN**: Implement JAX backend (optional, if JAX available)
- [ ] **GREEN**: Refactor core modules to use backend
- [ ] **REFACTOR**: Clean up backend abstraction

---

### Phase 9: Oja Rule + Advanced Plasticity
**Modules**: `learning/oja.py`

- [ ] **RED**: Write failing unit tests
  - [ ] `test/unit/test_oja.py` - weight normalization, decay behavior
- [ ] **RED**: Write failing integration tests
  - [ ] `test/integration/test_oja_simulation.py` - Oja rule integrates with Simulation
- [ ] **RED**: Write failing property tests
  - [ ] `test/property/test_oja_props.py` - weights converge, norm bounded
- [ ] **GREEN**: Implement `learning/oja.py`
- [ ] **GREEN**: Integrate Oja rule option into Simulation
- [ ] **REFACTOR**: Clean up plasticity options
