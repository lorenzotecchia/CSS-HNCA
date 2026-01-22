# Commit History

**Last Updated**: 2026-01-22

## Recent Commits (Newest First)

| Hash | Message | Phase |
|------|---------|-------|
| `6e2fd20` | feat: implement Phase 2 config loader and CLI (TDD complete) | Phase 2 |
| `f27005a` | feat: implement Phase 5 Hebbian learning with STDP (TDD complete) | Phase 5 |
| `f4733ec` | feat: implement Phase 1 core skeleton (TDD complete) | Phase 1 |
| `b9286cb` | Document Network Generation Algorithm in css-theory.md | Docs |
| `3e4957c` | rules fixes, math formalism | Docs |
| `cb7586f` | mathematical formalism for the change state algorithm | Docs |
| `bc23665` | feat: account for strict tdd in planning | Planning |
| `b9bcd51` | docs: add architecture design for neural cellular automata | Planning |
| `35c257d` | feat: added skeleton for theory | Docs |
| `7026991` | chore: updated the readme | Setup |
| `d725f42` | feat: added makefile to enhance development | Setup |
| `f940b8b` | Initial commit | Setup |

## Phase Completion Summary

- **Phase 1 (Core Skeleton)**: COMPLETE - `f4733ec`
- **Phase 2 (Config Loader & CLI)**: COMPLETE - `6e2fd20`
- **Phase 3 (Event Bus)**: NOT STARTED
- **Phase 4 (Pygame Visualization)**: NOT STARTED
- **Phase 5 (Hebbian Learning + STDP)**: COMPLETE - `f27005a`
- **Phase 6 (Matplotlib Analytics)**: NOT STARTED
- **Phase 7 (TUI Logger + Output Recorder)**: NOT STARTED
- **Phase 8 (Backend Abstraction/JAX)**: NOT STARTED
- **Phase 9 (Oja Rule + Advanced Plasticity)**: NOT STARTED

## [2026-01-22] 9cd4317
**Message:** fix: wire HebbianLearner in main.py CLI
**Changes:** Import HebbianLearner and instantiate it with config values in `create_simulation_from_config()`, then pass to Simulation constructor. Previously learning was never applied because learner was None.

## [2026-01-22] e843616
**Message:** feat(phase7): TUI Logger + Output Recorder
**Changes:** Implemented TUILogger for terminal output and Recorder for CSV/NPZ persistence. 35 new tests (unit, integration, property). 197 total tests passing.

## [2026-01-22] 2574349
**Message:** feat(phase6): Matplotlib Analytics visualization
**Changes:** MatplotlibAnalyticsView with firing/weight line plots, histogram, and optional heatmap. Demo script included.
