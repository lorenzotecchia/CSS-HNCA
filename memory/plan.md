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

### Phase 3: Event Bus ✅ COMPLETE
**Modules**: `events/bus.py`

- [x] **RED**: Write failing unit tests
  - [x] `test/unit/test_event_bus.py` - subscribe, emit, multiple handlers, type filtering
- [x] **RED**: Write failing integration tests
  - [x] `test/integration/test_simulation_events.py` - Simulation emits events on step/reset
- [x] **RED**: Write failing property tests
  - [x] `test/property/test_event_bus_props.py` - all subscribed handlers receive events
- [x] **GREEN**: Implement `events/bus.py`
- [x] **GREEN**: Integrate events into `core/simulation.py`
- [x] **REFACTOR**: Clean up events module

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

### Phase 6: Matplotlib Analytics ✅ COMPLETE
**Modules**: `visualization/matplotlib_view.py`

> **Note**: Visualization tests skipped (tested by running)

- [x] Implement `visualization/matplotlib_view.py`
- [x] Manual testing: verify plots update correctly (demo script: `scripts/demo_matplotlib.py`)

---

### Phase 7: TUI Logger + Output Recorder ✅ COMPLETE
**Modules**: `visualization/tui_logger.py`, `output/recorder.py`

- [x] **RED**: Write failing unit tests
  - [x] `test/unit/test_tui_logger.py` - output format, handles step events
  - [x] `test/unit/test_recorder.py` - CSV format, NPZ structure
- [x] **RED**: Write failing integration tests
  - [x] `test/integration/test_headless_run.py` - full simulation with TUI + recorder
- [x] **RED**: Write failing property tests
  - [x] `test/property/test_recorder_props.py` - recorded steps match simulation steps
- [x] **GREEN**: Implement `visualization/tui_logger.py`
- [x] **GREEN**: Implement `output/recorder.py`
- [x] **REFACTOR**: Clean up output module

---

### Phase 8: Backend Abstraction (JAX) ✅ COMPLETE
**Modules**: `core/backend.py`, `core/backend_jax.py`

- [x] **RED**: Write failing unit tests
  - [x] `test/unit/test_backend.py` - 36 tests for NumPy backend operations
- [x] **RED**: Write failing integration tests
  - [x] `test/integration/test_backend_simulation.py` - 10 tests for backend + simulation compatibility
- [x] **RED**: Write failing property tests
  - [x] `test/property/test_backend_props.py` - 22 property tests for backend invariants
- [x] **GREEN**: Implement `core/backend.py` (NumPy) - ArrayBackend Protocol + NumPyBackend
- [x] **GREEN**: Implement JAX backend (`core/backend_jax.py`) - graceful fallback when JAX unavailable
- [x] **GREEN**: Verified core modules compatible with backend (arrays interoperate)
- [x] **REFACTOR**: Clean up backend abstraction

---

### Phase 9: Leaky Integrate-and-Fire (LIF) Neurons ✅ COMPLETE
**Modules**: `core/neuron_state.py`, `core/simulation.py`, `config/loader.py`

**Goal**: Replace binary threshold firing with membrane potential that integrates and leaks.

**LIF Dynamics**:
```
V_i(t+1) = (1 - λ) * V_i(t) + Σ(w_ji * s_j(t)) - V_reset * fired(t)
```
- `λ` = leak rate (potential decays toward 0)
- `V_reset` = amount subtracted after firing (refractory-like behavior)
- Fires when `V_i >= threshold`

- [x] **RED**: Write failing unit tests
  - [x] `test/unit/test_lif.py` - 16 tests for LIF dynamics
- [x] **RED**: Write failing integration tests
  - [x] `test/integration/test_lif_simulation.py` - 4 tests for LIF preventing runaway
- [x] **GREEN**: Add `membrane_potential: ndarray` to `NeuronState`
- [x] **GREEN**: Modify `update_firing()` to use LIF dynamics
- [x] **GREEN**: Add config params: `leak_rate`, `reset_potential`
- [x] **REFACTOR**: Ensure backward compatibility with existing tests

---

### Phase 10: Weight Decay Mechanisms ✅ COMPLETE
**Modules**: `learning/hebbian.py`, `core/simulation.py`

**Goal**: Prevent unbounded weight growth with baseline decay + Oja competitive decay.

**A. Baseline Weight Decay** (uses existing `decay_alpha`):
```
W(t+1) = (1 - α) * W(t)
```

**B. Oja Rule (Activity-Dependent Decay)**:
```
W_AB(t+1) = W_AB(t) + l*(LTP) - f*(LTD) - α_oja * activity_B² * W_AB
```

- [x] **RED**: Write failing unit tests
  - [x] `test/unit/test_weight_decay.py` - 10 tests for decay mechanisms
- [x] **GREEN**: Add baseline decay step in HebbianLearner
- [x] **GREEN**: Add Oja decay term to `HebbianLearner.apply()`
- [x] **GREEN**: Add `oja_alpha` config parameter
- [x] **REFACTOR**: Clean up decay logic

---

### Phase 11: Avalanche Detection & SOC Metrics ✅ COMPLETE
**Modules**: `events/avalanche.py` (new)

**Goal**: Detect avalanches and compute SOC metrics (power-law distributions, branching ratio).

**Avalanche Definition**:
1. Quiet state: firing count < `quiet_threshold` (e.g., 5% of neurons)
2. Avalanche starts: firing rises above quiet threshold
3. Track size: sum of firing events during avalanche
4. Avalanche ends: firing returns below quiet threshold
5. Record: size, duration, peak activity

**SOC Metrics**:
- Avalanche size distribution → power law slope ≈ -1.5
- Avalanche duration distribution → power law
- Branching ratio: avg(firing_t+1 / firing_t) → ≈ 1.0 at criticality

- [x] **RED**: Write failing unit tests
  - [x] `test/unit/test_avalanche.py` - 16 tests for detection logic
- [x] **RED**: Write failing integration tests
  - [x] `test/integration/test_avalanche_events.py` - 3 tests for event bus wiring
- [x] **GREEN**: Implement `events/avalanche.py` - AvalancheDetector class
- [x] **GREEN**: Implement compute_branching_ratio() and get_size_distribution()

---

### Phase 12: SOC Parameter Tuning & Validation
**Goal**: Find parameter values that produce critical dynamics.

**Experiments**:
- [ ] Run parameter sweeps: leak_rate, reset_potential, decay_alpha, oja_alpha
- [ ] Measure avalanche size distribution slope (target: -1.5)
- [ ] Measure branching ratio (target: 1.0)
- [ ] Verify homeostasis: stable average activity over long runs

**Expected Behavior After All Phases**:
1. **Early simulation**: Random initial firing triggers small cascades
2. **Learning phase**: Weights self-organize, some pathways strengthen
3. **Critical regime**: Variable firing (not saturated), power-law avalanches
4. **Homeostasis**: Average activity stabilizes

---

### Phase 12.5: Optional Backend Integration (JAX Acceleration) ✅ COMPLETE
**Goal**: Make core modules use ArrayBackend optionally for JAX GPU acceleration.

**Status**: Completed 2026-01-23. All 333 tests pass.

**Architecture**: Add optional `backend` parameter to Network, NeuronState, and HebbianLearner. Default to NumPyBackend for backward compatibility. When backend is provided, use its methods instead of raw NumPy. Simulation orchestrates backend selection.

**Files to Modify**:
- `src/core/network.py` - Add backend parameter to `create_random()`
- `src/core/neuron_state.py` - Add backend parameter to `create()` and `update_firing()`
- `src/learning/hebbian.py` - Add backend parameter to `HebbianLearner`
- `src/core/simulation.py` - Add backend parameter, pass to components
- `scripts/soc_parameter_sweep.py` - Use backend with `prefer_gpu=True`

---

#### Task 1: Update Network.create_random() with Optional Backend

**Files:**
- Modify: `src/core/network.py`
- Test: `test/unit/test_network.py` (existing tests should still pass)

**Step 1: Add backend import and parameter**

```python
# At top of file, add:
from src.core.backend import ArrayBackend, get_backend

# In create_random signature, add:
def create_random(
    cls,
    n_neurons: int,
    box_size: tuple[float, float, float],
    radius: float,
    initial_weight: float,
    seed: int | None = None,
    backend: ArrayBackend | None = None,  # NEW
) -> "Network":
```

**Step 2: Use backend in method body (fallback to NumPy if None)**

Replace NumPy calls with backend calls when backend is provided:
```python
if backend is None:
    backend = get_backend()

# Replace: rng.uniform(0, box_size[0], n_neurons)
# With: backend.random_uniform(0, box_size[0], (n_neurons,), seed)
```

**Step 3: Run existing tests**

```bash
pytest test/unit/test_network.py -v
```
Expected: All tests pass (backward compatible)

**Step 4: Commit**

```bash
git add src/core/network.py
git commit -m "feat(network): add optional backend parameter to create_random"
```

---

#### Task 2: Update NeuronState.create() with Optional Backend

**Files:**
- Modify: `src/core/neuron_state.py`

**Step 1: Add backend import and parameter**

```python
from src.core.backend import ArrayBackend, get_backend

# In create() signature, add:
backend: ArrayBackend | None = None,
```

**Step 2: Use backend for array creation**

```python
if backend is None:
    backend = get_backend()

# Replace: rng.random(n_neurons) < initial_firing_fraction
# With: backend.random_bool(initial_firing_fraction, (n_neurons,), seed)

# Replace: np.zeros(n_neurons, dtype=np.bool_)
# With: backend.zeros((n_neurons,), dtype=np.bool_)
```

**Step 3: Run existing tests**

```bash
pytest test/unit/test_neuron_state.py test/unit/test_lif.py -v
```

**Step 4: Commit**

```bash
git add src/core/neuron_state.py
git commit -m "feat(neuron_state): add optional backend parameter"
```

---

#### Task 3: Update NeuronState.update_firing() to Use Backend

**Files:**
- Modify: `src/core/neuron_state.py`

**Step 1: Store backend instance in dataclass**

```python
@dataclass
class NeuronState:
    # ... existing fields ...
    backend: ArrayBackend = field(default_factory=get_backend)
```

**Step 2: Use backend in update_firing()**

Replace in-place NumPy operations with backend equivalents:
```python
# self.membrane_potential *= (1 - self.leak_rate)
# becomes:
self.membrane_potential = self.backend.to_numpy(
    (1 - self.leak_rate) * self.membrane_potential
)
```

Note: For JAX compatibility, avoid in-place mutations. Use assignment instead.

**Step 3: Run LIF tests**

```bash
pytest test/unit/test_lif.py -v
```

**Step 4: Commit**

```bash
git add src/core/neuron_state.py
git commit -m "feat(neuron_state): use backend in update_firing"
```

---

#### Task 4: Update HebbianLearner with Optional Backend

**Files:**
- Modify: `src/learning/hebbian.py`

**Step 1: Add backend parameter**

```python
from src.core.backend import ArrayBackend, get_backend

@dataclass
class HebbianLearner:
    # ... existing fields ...
    backend: ArrayBackend = field(default_factory=get_backend)
```

**Step 2: Use backend in apply()**

Replace NumPy calls:
```python
# np.outer(prev, curr) -> backend-compatible
# np.clip(...) -> backend.where() with bounds
# np.where(link_matrix, ...) -> backend.where(...)
```

**Step 3: Run existing tests**

```bash
pytest test/unit/test_hebbian.py test/unit/test_weight_decay.py -v
```

**Step 4: Commit**

```bash
git add src/learning/hebbian.py
git commit -m "feat(hebbian): add optional backend parameter"
```

---

#### Task 5: Update Simulation with Backend Selection

**Files:**
- Modify: `src/core/simulation.py`

**Step 1: Add backend parameter**

```python
from src.core.backend import ArrayBackend, get_backend

@dataclass
class Simulation:
    # ... existing fields ...
    backend: ArrayBackend = field(default_factory=get_backend)
```

**Step 2: Use backend in step()**

```python
# Replace: self.network.weight_matrix.T @ self.state.firing.astype(float)
# With: backend.matmul(backend.transpose(self.network.weight_matrix), ...)
```

**Step 3: Run integration tests**

```bash
pytest test/integration/test_simulation_core.py -v
```

**Step 4: Commit**

```bash
git add src/core/simulation.py
git commit -m "feat(simulation): add optional backend parameter"
```

---

#### Task 6: Update Parameter Sweep to Use Backend

**Files:**
- Modify: `scripts/soc_parameter_sweep.py`

**Step 1: Add backend selection**

```python
from src.core.backend import get_backend

# At start of run_single_sweep:
backend = get_backend(prefer_gpu=True)
```

**Step 2: Pass backend to all components**

```python
network = Network.create_random(..., backend=backend)
state = NeuronState.create(..., backend=backend)
learner = HebbianLearner(..., backend=backend)
sim = Simulation(..., backend=backend)
```

**Step 3: Test sweep runs**

```bash
python scripts/soc_parameter_sweep.py 2>&1 | head -20
```

**Step 4: Commit**

```bash
git add scripts/soc_parameter_sweep.py
git commit -m "feat(sweep): use optional GPU backend when available"
```

---

#### Task 7: Add Backend Integration Tests

**Files:**
- Create: `test/integration/test_backend_integration.py`

**Step 1: Write test for backend propagation**

```python
def test_simulation_with_explicit_backend():
    """Verify all components use the same backend."""
    from src.core.backend import NumPyBackend
    backend = NumPyBackend()
    
    network = Network.create_random(n_neurons=10, ..., backend=backend)
    state = NeuronState.create(n_neurons=10, ..., backend=backend)
    learner = HebbianLearner(..., backend=backend)
    sim = Simulation(network=network, state=state, learner=learner, backend=backend)
    
    sim.start()
    sim.step()
    assert sim.time_step == 1
```

**Step 2: Run test**

```bash
pytest test/integration/test_backend_integration.py -v
```

**Step 3: Commit**

```bash
git add test/integration/test_backend_integration.py
git commit -m "test: add backend integration tests"
```

---

#### Task 8: Final Verification

**Step 1: Run full test suite**

```bash
pytest -v
```

**Step 2: Verify backward compatibility**

All existing tests must pass without modification (they don't pass backend parameter).

**Step 3: Document in README**

Add to README.md:
```markdown
## GPU Acceleration (Optional)

Install JAX for GPU acceleration:
\`\`\`bash
pip install jax jaxlib  # CPU
pip install jax[cuda12]  # NVIDIA GPU
pip install jax[metal]   # Apple Silicon
\`\`\`

The simulation will automatically use GPU when available.
```

**Step 4: Final commit**

```bash
git add README.md
git commit -m "docs: add GPU acceleration instructions"
```

---

## Bug Fixes Log

### [2026-01-22] Saturation Bug Fix

**Problem**: Network saturated at 300/300 neurons firing permanently.

**Root Cause**: `main.py` was not passing LIF and decay parameters to simulation:
- `leak_rate` and `reset_potential` → defaulted to 0 (no LIF dynamics)
- `decay_alpha` and `oja_alpha` → defaulted to 0 (no weight decay)

**Solution**:
1. **main.py**: Pass all config parameters to `NeuronState.create()` and `HebbianLearner()`
2. **neuron_state.py**: Initialize membrane potential at threshold for initially firing neurons
3. **config/default.toml**: Tuned for balanced dynamics (not saturating, not dying instantly)

**Parameter Tuning Insights**:
- `threshold` must be > `avg_connections * initial_weight` to prevent saturation
- `forgetting_rate >= learning_rate` provides stability (LTD >= LTP)
- `leak_rate` ~0.1 with `reset_potential` ~0.4-0.5 gives natural transients
- Without external input, activity eventually dies (expected for recurrent network)

### [2026-01-23] Code Review Bug Fixes

**Problem 1**: `Simulation.reset()` lost LIF parameters after reset.

**Root Cause**: `NeuronState.create()` was called without `leak_rate` and `reset_potential` params.

**Solution**: Pass `leak_rate=self.state.leak_rate` and `reset_potential=self.state.reset_potential` to `NeuronState.create()`.

---

**Problem 2**: `Simulation.reset()` didn't update membrane potential.

**Root Cause**: Only `firing` and `firing_prev` were copied from new state.

**Solution**: Added `self.state.membrane_potential[:] = new_state.membrane_potential`.

---

**Problem 3**: `average_weight` metric included zeros (non-connections).

**Root Cause**: `np.mean(weight_matrix)` averaged over all N×N elements.

**Solution**: Changed to `np.mean(weight_matrix[link_matrix])` to average only connected weights.

---

**Problem 4**: Branching ratio only used current avalanche data.

**Root Cause**: `compute_branching_ratio()` only looked at `_firing_history` from current avalanche.

**Solution**: Added `_all_ratios` list to accumulate ratios across all completed avalanches.

---

### Phase 13: Excitatory/Inhibitory Neurons
**Goal**: Differentiate neurons into excitatory (positive weights) and inhibitory (negative weights) types with configurable fractions and separate weight bounds.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Status**: Planned

**Architecture**: 
- Neurons assigned types at network creation based on `excitatory_fraction` (default 80%)
- Excitatory neurons have positive outgoing weights, inhibitory have negative
- HebbianLearner enforces separate weight bounds per type during learning
- Weight signs preserved: excitatory always positive, inhibitory always negative

**Tech Stack**: Python, NumPy, pytest, hypothesis

---

#### Task 13.1: Add Config Parameters

**Files:**
- Modify: `src/config/loader.py`
- Modify: `config/default.toml`
- Test: `test/unit/test_config_loader.py`

**Step 1: Write the failing test**

```python
# Add to test/unit/test_config_loader.py

def test_excitatory_fraction_parsed():
    """Test excitatory_fraction is parsed from config."""
    config = load_config(Path("config/default.toml"))
    assert hasattr(config.network, "excitatory_fraction")
    assert 0 <= config.network.excitatory_fraction <= 1


def test_inhibitory_weight_bounds_parsed():
    """Test inhibitory weight bounds are parsed from config."""
    config = load_config(Path("config/default.toml"))
    assert hasattr(config.network, "weight_min_inh")
    assert hasattr(config.network, "weight_max_inh")
    assert config.network.weight_min_inh <= config.network.weight_max_inh


def test_excitatory_fraction_validation():
    """Test excitatory_fraction must be in [0, 1]."""
    # This would need a custom invalid config - skip for now
    pass
```

**Step 2: Run test to verify it fails**

```bash
pytest test/unit/test_config_loader.py::test_excitatory_fraction_parsed -v
```

Expected: FAIL with `AttributeError: 'NetworkConfig' object has no attribute 'excitatory_fraction'`

**Step 3: Add fields to NetworkConfig**

```python
# In src/config/loader.py, update NetworkConfig dataclass:

@dataclass(frozen=True)
class NetworkConfig:
    # ... existing fields ...
    excitatory_fraction: float  # Fraction of neurons that are excitatory
    weight_min_inh: float       # Minimum weight for inhibitory neurons
    weight_max_inh: float       # Maximum weight for inhibitory neurons
```

**Step 4: Add validation in _validate_network_config**

```python
# Add to _validate_network_config():

excitatory_fraction = data.get("excitatory_fraction", 0.8)
if not 0 <= excitatory_fraction <= 1:
    raise ConfigValidationError("excitatory_fraction must be in [0, 1]")

weight_min_inh = data.get("weight_min_inh", -0.3)
weight_max_inh = data.get("weight_max_inh", 0.0)
if weight_min_inh > weight_max_inh:
    raise ConfigValidationError("weight_min_inh must be <= weight_max_inh")
```

**Step 5: Update _parse_network_config**

```python
# Add to return statement in _parse_network_config():

excitatory_fraction=float(data.get("excitatory_fraction", 0.8)),
weight_min_inh=float(data.get("weight_min_inh", -0.3)),
weight_max_inh=float(data.get("weight_max_inh", 0.0)),
```

**Step 6: Add to config/default.toml**

```toml
# Add to [network] section:
excitatory_fraction = 0.8      # 80% excitatory, 20% inhibitory
weight_min_inh = -0.3          # inhibitory weight bounds
weight_max_inh = 0.0
```

**Step 7: Run tests to verify they pass**

```bash
pytest test/unit/test_config_loader.py -v
```

Expected: PASS

**Step 8: Commit**

```bash
git add src/config/loader.py config/default.toml test/unit/test_config_loader.py
git commit -m "feat(config): add excitatory/inhibitory neuron config parameters"
```

---

#### Task 13.2: Add neuron_types to Network

**Files:**
- Modify: `src/core/network.py`
- Test: `test/unit/test_network.py`

**Step 1: Write failing tests**

```python
# Add to test/unit/test_network.py

def test_network_has_neuron_types():
    """Network should have neuron_types array."""
    network = Network.create_random(
        n_neurons=100,
        box_size=(10.0, 10.0, 10.0),
        radius=2.0,
        initial_weight=0.1,
        seed=42,
    )
    assert hasattr(network, "neuron_types")
    assert network.neuron_types.shape == (100,)
    assert network.neuron_types.dtype == bool


def test_excitatory_fraction_respected():
    """Approximately excitatory_fraction of neurons should be excitatory."""
    network = Network.create_random(
        n_neurons=1000,
        box_size=(10.0, 10.0, 10.0),
        radius=2.0,
        initial_weight=0.1,
        excitatory_fraction=0.8,
        seed=42,
    )
    actual_fraction = np.mean(network.neuron_types)
    assert 0.75 < actual_fraction < 0.85  # Allow some variance


def test_excitatory_neurons_have_positive_weights():
    """Excitatory neurons should have positive outgoing weights."""
    network = Network.create_random(
        n_neurons=100,
        box_size=(10.0, 10.0, 10.0),
        radius=2.0,
        initial_weight=0.1,
        excitatory_fraction=0.5,
        seed=42,
    )
    for i in range(network.n_neurons):
        if network.neuron_types[i]:  # Excitatory
            row_weights = network.weight_matrix[i, network.link_matrix[i]]
            if len(row_weights) > 0:
                assert np.all(row_weights >= 0), f"Excitatory neuron {i} has negative weights"


def test_inhibitory_neurons_have_negative_weights():
    """Inhibitory neurons should have negative outgoing weights."""
    network = Network.create_random(
        n_neurons=100,
        box_size=(10.0, 10.0, 10.0),
        radius=2.0,
        initial_weight=0.1,
        excitatory_fraction=0.5,
        seed=42,
    )
    for i in range(network.n_neurons):
        if not network.neuron_types[i]:  # Inhibitory
            row_weights = network.weight_matrix[i, network.link_matrix[i]]
            if len(row_weights) > 0:
                assert np.all(row_weights <= 0), f"Inhibitory neuron {i} has positive weights"
```

**Step 2: Run tests to verify they fail**

```bash
pytest test/unit/test_network.py::test_network_has_neuron_types -v
```

Expected: FAIL with `AttributeError`

**Step 3: Add neuron_types to Network dataclass**

```python
# In src/core/network.py, update Network dataclass:

@dataclass
class Network:
    positions: ndarray
    link_matrix: ndarray
    weight_matrix: ndarray
    neuron_types: ndarray  # True = excitatory, False = inhibitory
    n_neurons: int
    radius: float
    box_size: tuple[float, float, float]
```

**Step 4: Update create_random signature**

```python
@classmethod
def create_random(
    cls,
    n_neurons: int,
    box_size: tuple[float, float, float],
    radius: float,
    initial_weight: float,
    excitatory_fraction: float = 0.8,  # NEW
    seed: int | None = None,
    backend: ArrayBackend | None = None,
) -> "Network":
```

**Step 5: Implement neuron type assignment and signed weights**

```python
# After computing positions, add:

# Assign neuron types: True = excitatory, False = inhibitory
rng = np.random.default_rng(seed)
neuron_types = rng.random(n_neurons) < excitatory_fraction

# ... existing distance/link_matrix code ...

# Weight matrix: sign depends on presynaptic neuron type
weight_signs = np.where(neuron_types, 1.0, -1.0)  # (N,)
weight_matrix = np.where(link_matrix, initial_weight * weight_signs[:, np.newaxis], 0.0)

# Update return:
return cls(
    positions=positions,
    link_matrix=link_matrix,
    weight_matrix=weight_matrix,
    neuron_types=neuron_types,
    n_neurons=n_neurons,
    radius=radius,
    box_size=box_size,
)
```

**Step 6: Run tests to verify they pass**

```bash
pytest test/unit/test_network.py -v
```

Expected: PASS (may need to fix other tests that create Network without neuron_types)

**Step 7: Fix any broken tests**

Other tests may need to add `neuron_types` when creating Network manually. Add:
```python
neuron_types=np.ones(n_neurons, dtype=bool)  # All excitatory for backward compat
```

**Step 8: Commit**

```bash
git add src/core/network.py test/unit/test_network.py
git commit -m "feat(network): add neuron_types for excitatory/inhibitory differentiation"
```

---

#### Task 13.3: HebbianLearner Per-Type Weight Bounds

**Files:**
- Modify: `src/learning/hebbian.py`
- Test: `test/unit/test_hebbian.py`

**Step 1: Write failing tests**

```python
# Add to test/unit/test_hebbian.py

def test_learner_accepts_inhibitory_bounds():
    """HebbianLearner should accept weight_min_inh and weight_max_inh."""
    learner = HebbianLearner(
        learning_rate=0.1,
        forgetting_rate=0.05,
        weight_min=0.0,
        weight_max=0.3,
        weight_min_inh=-0.3,
        weight_max_inh=0.0,
    )
    assert learner.weight_min_inh == -0.3
    assert learner.weight_max_inh == 0.0


def test_apply_enforces_excitatory_bounds():
    """Excitatory neurons should be clamped to [weight_min, weight_max]."""
    learner = HebbianLearner(
        learning_rate=0.5,
        forgetting_rate=0.0,
        weight_min=0.0,
        weight_max=0.2,
        weight_min_inh=-0.2,
        weight_max_inh=0.0,
    )
    
    weights = np.array([[0.0, 0.15], [0.0, 0.0]])
    link_matrix = np.array([[False, True], [False, False]])
    neuron_types = np.array([True, True])  # Both excitatory
    firing_prev = np.array([True, False])
    firing_curr = np.array([False, True])
    
    new_weights = learner.apply(weights, link_matrix, firing_prev, firing_curr, neuron_types)
    
    # LTP would push weight above 0.2, should be clamped
    assert new_weights[0, 1] <= 0.2


def test_apply_enforces_inhibitory_bounds():
    """Inhibitory neurons should be clamped to [weight_min_inh, weight_max_inh]."""
    learner = HebbianLearner(
        learning_rate=0.5,
        forgetting_rate=0.0,
        weight_min=0.0,
        weight_max=0.2,
        weight_min_inh=-0.2,
        weight_max_inh=0.0,
    )
    
    weights = np.array([[0.0, -0.1], [0.0, 0.0]])
    link_matrix = np.array([[False, True], [False, False]])
    neuron_types = np.array([False, True])  # First inhibitory
    firing_prev = np.array([True, False])
    firing_curr = np.array([False, True])
    
    new_weights = learner.apply(weights, link_matrix, firing_prev, firing_curr, neuron_types)
    
    # Inhibitory weights should stay in [-0.2, 0.0]
    assert -0.2 <= new_weights[0, 1] <= 0.0
```

**Step 2: Run tests to verify they fail**

```bash
pytest test/unit/test_hebbian.py::test_learner_accepts_inhibitory_bounds -v
```

Expected: FAIL with `TypeError: unexpected keyword argument 'weight_min_inh'`

**Step 3: Add inhibitory bounds to HebbianLearner**

```python
# In src/learning/hebbian.py, update HebbianLearner:

@dataclass
class HebbianLearner:
    learning_rate: float
    forgetting_rate: float
    weight_min: float
    weight_max: float
    decay_alpha: float = 0.0
    oja_alpha: float = 0.0
    weight_min_inh: float = -0.3  # NEW: inhibitory min
    weight_max_inh: float = 0.0   # NEW: inhibitory max
    backend: ArrayBackend = field(default_factory=get_backend)
```

**Step 4: Update apply() signature**

```python
def apply(
    self,
    weights: ndarray,
    link_matrix: ndarray,
    firing_prev: ndarray,
    firing_current: ndarray,
    neuron_types: ndarray | None = None,  # NEW: optional for backward compat
) -> ndarray:
```

**Step 5: Implement per-type bounds in apply()**

```python
# Replace the simple clip at the end with:

# Enforce per-type bounds
if neuron_types is not None:
    # Create per-row min/max based on neuron type
    min_bounds = np.where(neuron_types, self.weight_min, self.weight_min_inh)
    max_bounds = np.where(neuron_types, self.weight_max, self.weight_max_inh)
    
    # Clip each row according to its neuron type
    for i in range(new_weights.shape[0]):
        new_weights[i] = np.clip(new_weights[i], min_bounds[i], max_bounds[i])
else:
    # Backward compatibility: all excitatory
    new_weights = np.clip(new_weights, self.weight_min, self.weight_max)
```

**Step 6: Run tests to verify they pass**

```bash
pytest test/unit/test_hebbian.py -v
```

Expected: PASS

**Step 7: Commit**

```bash
git add src/learning/hebbian.py test/unit/test_hebbian.py
git commit -m "feat(hebbian): enforce per-type weight bounds for excitatory/inhibitory neurons"
```

---

#### Task 13.4: Wire neuron_types Through Simulation

**Files:**
- Modify: `src/core/simulation.py`
- Test: `test/unit/test_simulation.py`

**Step 1: Write failing test**

```python
# Add to test/unit/test_simulation.py

def test_simulation_passes_neuron_types_to_learner():
    """Simulation should pass network.neuron_types to learner.apply()."""
    from unittest.mock import MagicMock
    
    network = Network.create_random(
        n_neurons=10,
        box_size=(5.0, 5.0, 5.0),
        radius=2.0,
        initial_weight=0.1,
        excitatory_fraction=0.5,
        seed=42,
    )
    state = NeuronState.create(n_neurons=10, threshold=0.5, seed=42)
    learner = MagicMock()
    learner.apply.return_value = network.weight_matrix.copy()
    
    sim = Simulation(
        network=network,
        state=state,
        learning_rate=0.1,
        forgetting_rate=0.05,
        learner=learner,
    )
    sim.start()
    sim.step()
    
    # Verify neuron_types was passed
    call_kwargs = learner.apply.call_args.kwargs
    assert "neuron_types" in call_kwargs or len(learner.apply.call_args.args) >= 5
```

**Step 2: Run test to verify it fails**

```bash
pytest test/unit/test_simulation.py::test_simulation_passes_neuron_types_to_learner -v
```

Expected: FAIL (neuron_types not passed)

**Step 3: Update Simulation.step() to pass neuron_types**

```python
# In src/core/simulation.py, update step() method:

if self.learner is not None:
    new_weights = self.learner.apply(
        weights=self.network.weight_matrix,
        link_matrix=self.network.link_matrix,
        firing_prev=firing_prev,
        firing_current=self.state.firing,
        neuron_types=getattr(self.network, 'neuron_types', None),  # NEW
    )
    self.network.weight_matrix[:] = new_weights
```

**Step 4: Run tests to verify they pass**

```bash
pytest test/unit/test_simulation.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/core/simulation.py test/unit/test_simulation.py
git commit -m "feat(simulation): pass neuron_types to HebbianLearner"
```

---

#### Task 13.5: Wire Config to main.py

**Files:**
- Modify: `main.py`

**Step 1: Update Network.create_random call**

```python
# In main.py, update network creation:

network = Network.create_random(
    n_neurons=config.network.n_neurons,
    box_size=config.network.box_size,
    radius=config.network.radius,
    initial_weight=config.network.initial_weight,
    excitatory_fraction=config.network.excitatory_fraction,  # NEW
    seed=config.seed,
)
```

**Step 2: Update HebbianLearner creation**

```python
# In main.py, update learner creation:

learner = HebbianLearner(
    learning_rate=config.learning.learning_rate,
    forgetting_rate=config.learning.forgetting_rate,
    weight_min=config.network.weight_min,
    weight_max=config.network.weight_max,
    weight_min_inh=config.network.weight_min_inh,  # NEW
    weight_max_inh=config.network.weight_max_inh,  # NEW
    decay_alpha=config.learning.decay_alpha,
    oja_alpha=config.learning.oja_alpha,
)
```

**Step 3: Run simulation to verify**

```bash
python main.py 2>&1 | head -20
```

Expected: Simulation runs without errors

**Step 4: Commit**

```bash
git add main.py
git commit -m "feat: wire excitatory/inhibitory config to simulation"
```

---

#### Task 13.6: Add Property Tests

**Files:**
- Modify: `test/property/test_network_props.py`
- Modify: `test/property/test_hebbian_props.py`

**Step 1: Add network property tests**

```python
# Add to test/property/test_network_props.py

from hypothesis import given, strategies as st

@given(
    n_neurons=st.integers(min_value=10, max_value=100),
    excitatory_fraction=st.floats(min_value=0.0, max_value=1.0),
    seed=st.integers(min_value=0, max_value=10000),
)
def test_neuron_types_length_matches_n_neurons(n_neurons, excitatory_fraction, seed):
    """neuron_types array should have length n_neurons."""
    network = Network.create_random(
        n_neurons=n_neurons,
        box_size=(10.0, 10.0, 10.0),
        radius=2.0,
        initial_weight=0.1,
        excitatory_fraction=excitatory_fraction,
        seed=seed,
    )
    assert len(network.neuron_types) == n_neurons


@given(
    seed=st.integers(min_value=0, max_value=10000),
)
def test_weight_signs_match_neuron_types(seed):
    """Excitatory neurons should have non-negative weights, inhibitory non-positive."""
    network = Network.create_random(
        n_neurons=50,
        box_size=(10.0, 10.0, 10.0),
        radius=3.0,
        initial_weight=0.1,
        excitatory_fraction=0.5,
        seed=seed,
    )
    for i in range(network.n_neurons):
        connected = network.link_matrix[i]
        if np.any(connected):
            weights = network.weight_matrix[i, connected]
            if network.neuron_types[i]:  # Excitatory
                assert np.all(weights >= 0)
            else:  # Inhibitory
                assert np.all(weights <= 0)
```

**Step 2: Add hebbian property test**

```python
# Add to test/property/test_hebbian_props.py

@given(
    seed=st.integers(min_value=0, max_value=10000),
)
def test_weight_bounds_preserved_after_learning(seed):
    """Weights should stay within type-specific bounds after learning."""
    rng = np.random.default_rng(seed)
    n = 20
    
    neuron_types = rng.random(n) < 0.5
    weights = np.where(neuron_types[:, None], 0.1, -0.1) * rng.random((n, n))
    link_matrix = rng.random((n, n)) < 0.3
    np.fill_diagonal(link_matrix, False)
    
    learner = HebbianLearner(
        learning_rate=0.1,
        forgetting_rate=0.1,
        weight_min=0.0,
        weight_max=0.3,
        weight_min_inh=-0.3,
        weight_max_inh=0.0,
    )
    
    firing_prev = rng.random(n) < 0.3
    firing_curr = rng.random(n) < 0.3
    
    new_weights = learner.apply(weights, link_matrix, firing_prev, firing_curr, neuron_types)
    
    for i in range(n):
        if neuron_types[i]:
            assert np.all(new_weights[i] >= 0.0)
            assert np.all(new_weights[i] <= 0.3)
        else:
            assert np.all(new_weights[i] >= -0.3)
            assert np.all(new_weights[i] <= 0.0)
```

**Step 3: Run property tests**

```bash
pytest test/property/test_network_props.py test/property/test_hebbian_props.py -v
```

Expected: PASS

**Step 4: Commit**

```bash
git add test/property/test_network_props.py test/property/test_hebbian_props.py
git commit -m "test: add property tests for excitatory/inhibitory neurons"
```

---

#### Task 13.7: Final Verification

**Step 1: Run all tests**

```bash
pytest --quiet
```

Expected: All tests pass

**Step 2: Run simulation and verify E/I dynamics**

```bash
python -c "
from pathlib import Path
from src.config.loader import load_config
from src.core.network import Network
import numpy as np

config = load_config(Path('config/default.toml'))
network = Network.create_random(
    n_neurons=config.network.n_neurons,
    box_size=config.network.box_size,
    radius=config.network.radius,
    initial_weight=config.network.initial_weight,
    excitatory_fraction=config.network.excitatory_fraction,
    seed=config.seed,
)

n_exc = np.sum(network.neuron_types)
n_inh = network.n_neurons - n_exc
print(f'Excitatory: {n_exc} ({n_exc/network.n_neurons:.1%})')
print(f'Inhibitory: {n_inh} ({n_inh/network.n_neurons:.1%})')

exc_weights = network.weight_matrix[network.neuron_types]
inh_weights = network.weight_matrix[~network.neuron_types]
print(f'Excitatory weights: [{exc_weights[exc_weights!=0].min():.4f}, {exc_weights[exc_weights!=0].max():.4f}]')
print(f'Inhibitory weights: [{inh_weights[inh_weights!=0].min():.4f}, {inh_weights[inh_weights!=0].max():.4f}]')
"
```

Expected output showing ~80% excitatory with positive weights, ~20% inhibitory with negative weights

**Step 3: Update memory/plan.md status**

Mark Phase 13 as complete.

**Step 4: Commit**

```bash
git add memory/plan.md
git commit -m "docs: mark Phase 13 (excitatory/inhibitory neurons) complete"
```

---

## Open Questions (SOC Design)

- [ ] What leak_rate and reset_potential values produce critical dynamics?
- [ ] Should we add external input/stimulation to trigger avalanches?
- [ ] Do we need inhibitory neurons for full SOC (currently all excitatory)? → **See Phase 13**

---

## References

- css-theory.md: Hebbian rules, Oja rule, saturation constraints
- Beggs & Plenz (2003): Neuronal avalanches in neocortex
- Oja (1982): Simplified neuron model with normalization
