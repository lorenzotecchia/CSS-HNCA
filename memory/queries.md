
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
