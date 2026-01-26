
## [2026-01-22 19:30] Fix persistent saturation after LIF implementation
User reports: "check the execution for python main.py --config config/default.toml --headless --steps 1000 -v, there is still saturation"

**Root cause identified:** `main.py` was not passing `leak_rate`, `reset_potential`, `decay_alpha`, and `oja_alpha` from config to simulation components. This meant:
- LIF dynamics disabled (leak_rate=0, reset_potential=0) - neurons never stopped firing
- Weight decay disabled (decay_alpha=0, oja_alpha=0) - no stabilization

**Fixes applied:**
1. `main.py`: Added missing params to `NeuronState.create()` and `HebbianLearner()`
2. `neuron_state.py`: Initialize membrane potential at threshold for initially firing neurons
3. `config/default.toml`: Tuned parameters for balanced dynamics
4. `test_lif_simulation.py`: Updated test for realistic transient dynamics

## [2026-01-22 18:09] Investigate firing saturation bug in headless simulation
User reports: "I can see a lot of trailing lines with the same neuron firing at libitum, that shouldn't really be the case based on the docs/css-theory.md"

**Root cause identified:** `main.py` does not wire up the `HebbianLearner` to the `Simulation`. Without learning, weights remain constant, and once enough neurons start firing they never stop because:
1. Firing neurons activate their neighbors
2. Neighbors exceed threshold and fire
3. System saturates to ~87% firing (263/300)
4. No LTD to reduce weights when patterns break STDP rules

**Fix:** Import and instantiate `HebbianLearner` in `main.py` and pass it to `Simulation`.

## [2026-01-22 18:15] Phase 7: TUI Logger + Output Recorder
User selected to work on Phase 7 with TDD. Modules: `visualization/tui_logger.py`, `output/recorder.py`

## [2026-01-22 18:21] Phase 6: Matplotlib Analytics
User selected Phase 6. Note: Visualization tests skipped per plan - tested by running manually.
## [2026-01-23 12:18] Query
I want the visualization to be rotational with mouse-drag orbiting and uniform node sizes (orthographic).

## [2026-01-23 13:12] Query
Add zoom capability to the pygame visualization.

## [2026-01-23 13:15] Query
Increase window size and stack control text at bottom-left.

