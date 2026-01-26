# Branch Comparison Report: `main` vs `features`

## Context

Both branches diverged from commit `183d296`. Since then:
- **main**: 23 commits (backend abstraction, E/I neurons, parameter sweep, tests)
- **features**: 5 commits (modular learning rules, directed networks, interactive UI)
- **52 files** differ; a merge attempt produces **9 conflicts** in core files

---

## 1. Network Model

The branches use completely different network generation strategies.

### main: `Network.create_random()`
- **Undirected** spatial graph
- Neurons placed in arbitrary `box_size` (e.g. `[10, 10, 10]`)
- Two neurons are connected if their Euclidean distance <= `radius`
- All edges get the **same weight** (`initial_weight`), signed by neuron type
- No guarantee of connectivity (depends on radius and density)
- Parameters: `n_neurons`, `box_size`, `radius`, `initial_weight`, `excitatory_fraction`

### features: `Network.create_beta_weighted_directed()`
- **Directed** graph
- Starts with a directed cycle (guarantees every neuron is reachable)
- Additional edges added by expanding a radius and sampling candidates
- Each edge gets an **independent weight** from a `Beta(a, b)` distribution
- Guaranteed strong connectivity via the cycle backbone
- Parameters: `n_neurons`, `k_prop`, `a`, `b`, `inhibitory_proportion`

### Impact
These produce fundamentally different topologies. The features model is richer (variable weights, directed, guaranteed connectivity) but harder to compare with spatial models from the literature.

---

## 2. Excitatory / Inhibitory Neurons

Both branches support E/I differentiation, but with inverted conventions.

| | main | features |
|---|---|---|
| Field name | `neuron_types` | `inhibitory_nodes` |
| Convention | `True` = excitatory | `True` = inhibitory |
| Config param | `excitatory_fraction` (default 0.8) | `inhibitory_proportion` (default 0.0) |
| Inhibitory weight bounds | `[-0.3, 0.0]` (configurable) | `[-1.0, -0.0001]` (hardcoded) |
| Excitatory weight bounds | `[weight_min, weight_max]` (configurable) | `[0.0001, 1.0]` (hardcoded) |
| Bounding implementation | Vectorized per-row `np.clip` | O(N^2) scalar double for-loop |

### Impact
Main's approach is more flexible (configurable bounds) and performant (vectorized). Features' approach prevents weights from ever reaching exactly zero (the 0.0001 floor), which is a deliberate design choice to preserve connection influence.

---

## 3. Learning Rules

This is where the branches diverge most in architecture.

### main: Monolithic `HebbianLearner`
- Single class with STDP + baseline decay + Oja rule all inline
- All rules always active (no toggles)
- Stateless (no memory between calls)
- Uses backend abstraction for JAX compatibility

### features: Modular `WeightUpdater` + standalone functions
- **`stdp.py`**: LTP/LTD as a standalone function
- **`oja.py`**: Oja's self-normalizing rule as a standalone function
- **`homeostatic.py`**: NEW rule not on main — adjusts incoming weights based on rolling spike counts to stabilize firing rates
- **`weight_update.py`**: Orchestrator that composes all three with `enable_stdp`, `enable_oja`, `enable_homeostatic` toggles
- Stateful (maintains `spike_history` deque for homeostatic scaling)
- Pure NumPy (no backend abstraction)

### The STDP and Oja algorithms are identical
The actual math for STDP (LTP/LTD via outer products) and Oja (multiplicative decay by postsynaptic activity squared) is the same on both branches. The difference is purely organizational.

### Homeostatic scaling (features only)
For each neuron `j`, count spikes over a rolling window:
- If `spike_count[j] < min_threshold`: boost all incoming weights to `j`
- If `spike_count[j] > max_threshold`: reduce all incoming weights to `j`
- Otherwise: no change

This is a biologically-motivated stabilization mechanism that prevents runaway excitation or complete silence.

### Impact
Features' modular architecture is a clear design improvement — it separates concerns, allows runtime toggling, and adds homeostatic scaling. But the O(N^2) bounding loop and lack of backend abstraction are performance regressions.

---

## 4. Neuron State & Initialization

| | main | features |
|---|---|---|
| Init parameter | `initial_firing_fraction` (float, 0 to 1) | `firing_count` (int, exact count) |
| Behavior | Each neuron fires independently with probability p | Exactly N neurons selected at random |
| Mutation style | Functional (new arrays each step, JAX-safe) | In-place (`*=`, `+=`, `np.copyto`) |
| `reinitialize_firing()` | Present (used after activity dies) | Absent |

### Impact
Features gives exact control over initial conditions. Main gives probabilistic initialization that's compatible with GPU backends. The missing `reinitialize_firing` on features means there's no clean way to restart activity after all neurons go silent (the `AvalancheController` works around this differently).

---

## 5. Avalanche Detection

| | main | features |
|---|---|---|
| Burn-in | First 10 avalanches discarded | None (all recorded) |
| Branching ratio scope | Global average across ALL avalanches | Most recent avalanche only |
| Ratio accumulation | `_all_ratios` list persists across avalanches | Not present |
| `_firing_history` cleanup | Cleared after each avalanche | **NOT cleared** (bug: leaks between avalanches) |

### Impact
These compute fundamentally different statistics. Main's approach (global average with burn-in) is more statistically robust. Features' approach (latest avalanche only) is more responsive but noisier and has a memory leak bug.

---

## 6. Visualization

### main: Pygame + optional matplotlib
- Static text panel (time, firing count, avg weight, seed)
- Optional `MatplotlibAnalyticsView` with incremental rendering and throttling
- Separate `AvalancheAnalyticsView` (6-subplot dashboard)
- Standalone scripts: `demo_avalanche_analytics.py`, `soc_parameter_sweep.py`
- `main.py` fully wired for both headless and visualization modes

### features: Pygame with interactive UI
- **17 interactive widgets**: sliders for all learning params, checkboxes for rule toggles, text inputs for network params
- All parameters adjustable at runtime without code changes
- Embedded `AvalancheController` with stimulus injection (press A, type count)
- Blue-teal-yellow edge gradient (vs main's grayscale)
- Matplotlib rendering is slower (full `clear()`/`plot()` every frame, no throttling)
- `main.py` visualization mode is a **stub** ("not yet implemented")

### Impact
Features has a much richer interactive experience for exploration. Main has better analytical tooling (parameter sweep, avalanche analytics, performant matplotlib). Features broke the `main.py` entry point.

---

## 7. Configuration

### main: Full spatial model config (12+ params)
```toml
n_neurons = 300
box_size = [10.0, 10.0, 10.0]
radius = 2.0
initial_weight = 0.06
weight_min = 0.0
weight_max = 0.3
initial_firing_fraction = 0.15
excitatory_fraction = 0.8
weight_min_inh = -0.3
weight_max_inh = 0.0
# + learning and LIF params
```

### features: Minimal config (4 network params)
```toml
n_neurons = 200
firing_count = 0
# + learning and LIF params
# Beta/directed network params are hardcoded in visualization.py
```

### Impact
Features moved most parameters to the interactive UI (sliders), which is great for exploration but means headless mode can't configure the network model from config alone.

---

## 8. Backend Abstraction (main only)

Main has `backend.py` and `backend_jax.py` providing:
- `ArrayBackend` Protocol with 14 methods
- `NumPyBackend` (default) and `JAXBackend` (GPU) implementations
- Factory function `get_backend(prefer_gpu=True/False)`
- Used throughout `Network`, `NeuronState`, `HebbianLearner`, `Simulation`

Features removed this entirely and hardcodes NumPy everywhere.

### Impact
Main can potentially run on GPU via JAX. Features is simpler but CPU-only. The backend abstraction adds complexity to every core file on main but enables scalability.

---

## 9. Test Coverage

| | main | features | delta |
|---|---|---|---|
| Backend tests | ~62 (unit + property + integration) | 0 | -62 |
| Config loader tests | ~28 (unit + property) | 0 | -28 |
| E/I property tests | ~22 | 0 | -22 |
| Homeostatic tests | 0 | 3 | +3 |
| Oja tests | 0 | 3 | +3 |
| STDP standalone tests | 0 | 4 | +4 |
| WeightUpdater tests | 0 | 6 | +6 |
| Inhibitory node tests | 0 | 8 | +8 |
| **Total difference** | | | **~88 fewer tests on features** |

Additionally, `test/unit/test_hebbian.py` on features has **broken references** — an incomplete refactor left references to `HebbianLearner` and `learner` where `WeightUpdater` and `updater` should be.

---

## 10. Bugs on `features`

1. **`simulation.py` reset bug**: Calls `NeuronState.create(..., initial_firing_fraction=...)` but features' `NeuronState.create` expects `firing_count` (int). Passing a float like 0.1 results in `firing_count=0` — no neurons fire after reset.

2. **`_firing_history` leak**: `AvalancheDetector._close_avalanche()` does not clear `_firing_history`, so data from one avalanche leaks into the branching ratio calculation of the next.

3. **Broken test file**: `test/unit/test_hebbian.py` has mixed references to both `HebbianLearner` and `WeightUpdater` — would fail on import.

4. **Stale script**: `scripts/demo_matplotlib.py` references config fields (`box_size`, `radius`, `initial_weight`) that no longer exist on features — would crash.

5. **`average_weight` dilution**: Computes mean over the entire weight matrix (including zeros where no connections exist), systematically underreporting actual connection strength.

---

## 11. Merge Feasibility

A `git merge features` produces **9 conflict files**:

```
config/default.toml
main.py
scripts/demo_matplotlib.py
src/core/network.py
src/core/neuron_state.py
src/core/simulation.py
src/learning/hebbian.py
src/visualization/visualization.py (add/add conflict)
```

Every core file conflicts. This is not a textual merge problem — it's a **semantic reconciliation** of two divergent architectures with different:
- Network generation models
- Boolean conventions (inverted E/I flags)
- Learning rule architectures
- State management paradigms (functional vs in-place)
- Configuration models

**A direct merge is not recommended.**

---

## 12. Summary: What Each Branch Does Better

### main is better at:
- Backend abstraction (GPU/JAX path)
- Test coverage (112 more tests)
- Correct `average_weight` (connected weights only)
- Avalanche statistics (burn-in, global branching ratio)
- Simulation reset (preserves LIF params)
- Matplotlib performance (incremental rendering, throttling)
- Working `main.py` entry point
- Configurable E/I weight bounds

### features is better at:
- Modular learning rules (separated STDP/Oja/homeostatic)
- Runtime-toggleable learning rules
- Homeostatic scaling (entirely new capability)
- Network topology (directed, Beta-distributed, guaranteed connectivity)
- Interactive Pygame UI (17 widgets, all params adjustable live)
- Embedded avalanche controller with stimulus injection
- Exact firing count initialization
- Visual edge rendering (color gradient)

---

## 13. Recommended Integration Strategy

Instead of merging, **cherry-pick features onto main**:

1. **Port modular learning rules** (`stdp.py`, `oja.py`, `homeostatic.py`, `weight_update.py`) onto main, adapting them to use the backend abstraction and `neuron_types` convention
2. **Add `create_beta_weighted_directed`** as an additional factory method on `Network` (keep `create_random` too)
3. **Port the interactive UI widgets** (sliders, checkboxes, text inputs) into main's visualization
4. **Port `AvalancheController`** into main's visualization alongside the existing analytics views
5. **Fix the bugs** listed in section 10
6. **Keep main's** avalanche detection (burn-in, global ratio), backend abstraction, test suite, config system, and `average_weight` calculation
7. **Vectorize** the O(N^2) weight bounding loop from `WeightUpdater`
8. **Write tests** for the newly ported modules
