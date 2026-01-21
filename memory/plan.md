# Neural Cellular Automata - Architecture Design
**Date**: 2026-01-21  
**Status**: Draft

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
    link_matrix: ndarray     # Shape: (N, N) - structural connectivity (bool)
    weight_matrix: ndarray   # Shape: (N, N) - synaptic strengths
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
| Connectivity | Directed, all within radius | Dense skeleton, prune via weights |
| Backend | NumPy default, JAX optional | Portable: consumer → HPC |
| Events | Typed event bus | Decouples core from viz |
| Config | TOML → dataclass | Type-safe, human-readable |
| Typing | Python 3.12 built-in + numpy.ndarray | Native, no extra packages |

---

## Implementation Phases

- [ ] **Phase 1**: Core skeleton (Network, NeuronState, Simulation without learning)
- [ ] **Phase 2**: Config loader and CLI
- [ ] **Phase 3**: Event bus
- [ ] **Phase 4**: Basic visualization (Pygame)
- [ ] **Phase 5**: Hebbian learning + STDP
- [ ] **Phase 6**: Matplotlib analytics
- [ ] **Phase 7**: TUI logger + output recorder
- [ ] **Phase 8**: Backend abstraction (JAX support)
- [ ] **Phase 9**: Oja rule + advanced plasticity
