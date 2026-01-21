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
