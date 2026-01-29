# CSS-HNCA
Complex System Simulation for Hebbian Neural Cellular Automaton


---

## Usage of the make file
Quick start the project using `make venv` to create a virtual environment in python3.12. Then activate the virtual environment with `source .venv/bin/activate` finally use `make install` to install dependencies. Use `make help` to inspect all the other commands available with make.

---

## Parallel Development with Git Worktrees

The project has 9 implementation phases (see `memory/plan.md`). You can work on multiple phases simultaneously using **git worktrees** with Copilot CLI.

### Why Worktrees?
- Work on Phase 3 (Event bus) while someone else works on Phase 4 (Pygame visualization)
- Each worktree is a separate directory with its own working state
- All worktrees share the same git repository

### Setting Up a Worktree for a Phase

1. **Start Copilot CLI** in the main repo and ask:
   ```
   Create a worktree for phase-3-events
   ```

2. Copilot will:
   - Detect/create the `.worktrees/` directory
   - Verify it's git-ignored
   - Run `git worktree add .worktrees/phase-3-events -b feature/phase-3-events`
   - Install dependencies (`make install`)
   - Run baseline tests

3. **Navigate to the worktree**:
   ```bash
   cd .worktrees/phase-3-events
   ```

4. **Start a new Copilot CLI session** in that directory and work on that phase.

### Example: Working on Multiple Phases

```bash
# Terminal 1 - Phase 3: Event bus
cd .worktrees/phase-3-events
copilot-cli
# "Implement the EventBus class from the plan"

# Terminal 2 - Phase 4: Pygame visualization  
cd .worktrees/phase-4-pygame
copilot-cli
# "Implement PygameNetworkView from the plan"

# Terminal 3 - Phase 5: Hebbian learning
cd .worktrees/phase-5-hebbian
copilot-cli
# "Implement Hebbian learning with STDP"
```

### Suggested Phase → Worktree Mapping

| Phase | Branch Name | Description |
|-------|-------------|-------------|
| 1 | `feature/phase-1-core` | Core skeleton (Network, NeuronState, Simulation) |
| 2 | `feature/phase-2-config` | Config loader and CLI |
| 3 | `feature/phase-3-events` | Event bus |
| 4 | `feature/phase-4-pygame` | Pygame visualization |
| 5 | `feature/phase-5-hebbian` | Hebbian learning + STDP |
| 6 | `feature/phase-6-matplotlib` | Matplotlib analytics |
| 7 | `feature/phase-7-output` | TUI logger + output recorder |
| 8 | `feature/phase-8-backend` | Backend abstraction (JAX) |
| 9 | `feature/phase-9-oja` | Oja rule + advanced plasticity |

### Managing Worktrees

```bash
# List all worktrees
git worktree list

# Remove a worktree after merging
git worktree remove .worktrees/phase-3-events

# Prune stale worktree references
git worktree prune
```

### Tips
- **Dependencies:** Each worktree needs its own venv. Run `make venv && source .venv/bin/activate && make install` in each.
- **Independent phases:** Phases 3, 4, 6, 7 can be developed in parallel (they only depend on core).
- **Sequential phases:** Phase 1 must complete before most others. Phase 5 (learning) integrates with Phase 1 (core).

---

## Development Commands

### Testing

```bash
# Run all tests (212 tests)
make test

# Run tests in parallel (faster)
make test-fast

# Run with coverage report
make test-cov

# Run specific test categories
python -m pytest test/unit/ -v          # Unit tests only
python -m pytest test/integration/ -v   # Integration tests only
python -m pytest test/property/ -v      # Property-based tests only

# Run tests for a specific module
python -m pytest test/unit/test_event_bus.py -v
```

### Linting

```bash
make lint          # Run flake8 linter
```

### Merging Worktree Changes

After completing work in a worktree:

```bash
# 1. Verify tests pass in the worktree
make test

# 2. Commit your changes
git add -A && git commit -m "feat: implement Phase X (TDD complete)"

# 3. Go to main worktree and merge
cd /path/to/main/CSS-HNCA
git merge feature/phase-X-branch --no-edit

# 4. Verify tests pass on main
make test

# 5. Update memory files (commits.md, tests.md, plan.md)
```

---

## Simulation Outputs

All simulation outputs are gitignored and stored locally. The `output/` directory contains results from local runs and parameter sweeps.

### Snellius Supercomputer Runs

Large-scale parameter sweeps were run on the Snellius supercomputer (SURF). Results are stored locally but not tracked in git.

| Directory | Job ID | Configs | Samples/Config | Parameters Swept | Total Runs |
|-----------|--------|---------|----------------|------------------|------------|
| `output_snellius/` | 18777853 | 54 | 3,000 LHS | firing_fraction (3), leak/reset (3), k_prop (3), decay/oja (2) | 162,000 |
| `output_snellius_v2/` | 18818779 | 1,080 | 500 LHS | excitatory_fraction (5), firing_fraction (3), leak/reset (2), k_prop (3), decay_alpha (4), oja_alpha (3) | 540,000 |

**Sweep v1** (`snellius_sweep.py`):
- 500 neurons, diagonal LHS sampling (learning_rate == forgetting_rate)
- Grid: 3×3×3×2 = 54 configurations
- Fixed parameters: excitatory_fraction=1.0, threshold=0.5, beta=(2,6)

**Sweep v2** (`snellius_sweep_v2.py`):
- 500 neurons, 2D LHS sampling (independent learning_rate, forgetting_rate)
- Grid: 5×3×2×3×4×3 = 1,080 configurations
- Features incremental checkpointing and resume capability
- Zoomed-in around avalanche-producing regimes from v1

### Analysis Scripts

| Script | Purpose |
|--------|---------|
| `scripts/plot_snellius_sweep.py` | Analyze v1 results |
| `scripts/plot_snellius_sweep_v2.py` | Analyze v2 results |
| `scripts/merge_results.py` | Aggregate multi-run results |

### Generated Plots

Results are visualized in `output/plots_snellius/` and `output/plots_snellius_v2/`:
- Parameter grid heatmaps
- Learning rate vs metrics by k_prop
- Decay/Oja interaction effects
- Excitatory fraction influence
