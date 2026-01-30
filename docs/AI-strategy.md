# AI Strategy and Reflection

## 1. Introduction

This document describes the GenAI-assisted development strategy employed for the CSS-HNCA (Hebbian Neural Cellular Automaton) project. The approach prioritizes structured prompting, test-driven development (TDD), and systematic documentation to ensure reproducibility and maintainability.

## 2. Tooling and Model Selection

### Primary Tools
- **GitHub Copilot CLI** with **Claude Opus 4.5** as the underlying model & *Claude Code Pro* with various models
- Claude Opus 4.5 was selected for its demonstrated effectiveness in complex coding tasks, as documented in the [Claude Opus 4.5 System Card](https://assets.anthropic.com/m/64823ba7485345a7/Claude-Opus-4-5-System-Card.pdf)

### Skills Framework
The project leverages the **Open Skills Standard** developed by Claude Code, an open specification for describing AI agent capabilities. Reference: [Agent Skills Specification](https://agentskills.io/specification).

Along with these basic skills, we also decided to use a framework of [superpowers](https://claude.com/plugins/superpowers) which further enrich the concept of skills to make the latter interoperate with each other. This way we ensured a greater degree of generalization and effectiveness. We refer to the [repository](https://github.com/obra/superpowers) where the claude code plugin was developed.

## 3. Development Methodology

### Waterfall-Style Phased Approach
Development followed a structured multi-step plan: While Claude Code supports parallel agent dispatching for concurrent tasks, this functionality was intentionally not exploited in favor of a sequential, "waterfall" pattern. This choice prioritized methodical validation at each phase before proceeding to the next.

### Test-Driven Development (TDD)
Each feature was implemented following TDD principles:
1. Define expected behavior through unit tests
2. Implement functionality to pass tests
3. Refactor and validate

**Final test coverage:** 212 tests passing

## 4. Skills-Based Workflow

The project uses `.github/skills/` to guide LLM behavior and maintain workflow consistency across sessions. The `neural-cellular-automata-project-workflow` skill defines a memory file structure for persistent state management.

### Memory File Structure (as defined in workflow skill)

```
memory/
|-- commits.md   # Commit history with feature descriptions
|-- plan.md      # Phase tracking and task management
|-- queries.md   # Representative prompts for reproducibility
+-- tests.md     # Test suite documentation by feature
```

### Purpose of Each File

| File | Purpose |
|------|---------|
| `plan.md` | Central planning document with task tracking, design decisions, and class specifications |
| `queries.md` | Timestamped log of key prompts to ensure reproducibility |
| `commits.md` | Commit summaries with features implemented and metrics |
| `tests.md` | Comprehensive test documentation organized by test suite |

> [!NOTE]
> These file have been eliminated for clarity, previous commits have these file updated to the last working version of the code and can be inspected to ensure repeatability.

## 5. Skills Structure

The following skills were defined to guide LLM behavior:

```
.github/skills/
|-- neural-cellular-automata-project-workflow/  # Project-specific workflow management
|   +-- SKILL.md
|-- brain-storming/           # Creative design exploration
|   +-- SKILL.md
|-- coding/                   # Implementation guidelines (code review agent)
|   +-- SKILL.md
|-- debugger/                 # Systematic debugging approach
|   +-- SKILL.md
|-- planner/                  # Task decomposition and tracking
|   +-- SKILL.md
|-- testing/                  # TDD and test creation
|   +-- SKILL.md
|-- using-git-worktrees/      # Feature isolation strategies
|   +-- SKILL.md
|-- verification-before-completion/  # Validation requirements
|   +-- SKILL.md
|-- dispatching-parallel-agents/     # Multi-agent coordination
|   +-- SKILL.md
|-- executing-plans/          # Plan execution orchestration
|   +-- SKILL.md
|-- finishing-a-development-branch/  # Branch completion workflow
|   +-- SKILL.md
|-- subagent-driven-development/     # Multi-agent development
|   +-- SKILL.md
|-- requesting-code-review/   # Code review workflow
|   +-- SKILL.md
+-- using-superpowers/        # Claude special capabilities
    +-- SKILL.md
```

### Skill Descriptions

| Skill | Purpose |
|-------|---------|
| **neural-cellular-automata-project-workflow** | Manages memory files for plans, queries, tests, and commits |
| **brain-storming** | Guides design exploration before implementation |
| **coding** | Senior code reviewer agent for quality assurance |
| **debugger** | Systematic approach for bug identification and resolution |
| **planner** | Task decomposition and progress tracking |
| **testing** | TDD workflow and test creation guidelines |
| **using-git-worktrees** | Isolated feature development via git worktrees |
| **verification-before-completion** | Ensures validation before claiming work complete |
| **dispatching-parallel-agents** | Coordinates concurrent work across multiple agents |
| **executing-plans** | Converts plans into executed development work |
| **finishing-a-development-branch** | Handles final steps before merge |
| **subagent-driven-development** | Multi-agent development coordination |
| **requesting-code-review** | Initiates formal code review process |
| **using-superpowers** | Leverages Claude's advanced features |

## 6. Prompting Strategy

### Effective Prompt Patterns

1. **Context-rich initial prompts**: Provided full project context including existing code structure, design decisions, and constraints
2. **Decision trees**: Presented options for design decisions (e.g., backend abstraction, visualization approach) for informed selection
3. **Incremental complexity**: Built features from simple to complex (e.g., Network → NeuronState → HebbianLearner → Simulation)
4. **Test-first requests**: Asked for tests before implementation

### Example Prompt Sequence

```
[2026-01-XX XX:XX] Project Setup
→ "Let's start with the appropriate skills and work on the plan..."

[2026-01-XX XX:XX] Backend Selection
→ "Let's implement a backend abstraction layer supporting NumPy and JAX..."

[2026-01-XX XX:XX] Learning Rule Design
→ "Implement STDP with configurable LTP/LTD rates and weight decay mechanisms..."
```

## 7. Key Outcomes

### Quantitative Results

| Metric | Value |
|--------|-------|
| Total Tests | 212 |
| Passing Tests | 212 (100%) |
| Core Classes | 11 |
| Source Files | 22 |
| Test Files | 28 |
| Parameter Sweep Configs | 50+ |

### Architecture Components

| Component | Description |
|-----------|-------------|
| **Network** | 3D spatial topology with synaptic weights |
| **NeuronState** | LIF dynamics with membrane potential |
| **HebbianLearner** | STDP with LTP/LTD and weight decay |
| **Simulation** | Orchestration of simulation loop |
| **EventBus** | Typed pub/sub for decoupled communication |
| **AvalancheDetector** | SOC metrics (power-law, branching ratio) |
| **ArrayBackend** | CPU/GPU backend abstraction |

### Experiment Findings

| Configuration | Metric | Insight |
|---------------|--------|---------|
| learning_rate × forgetting_rate | Criticality | Explored 50+ parameter combinations |
| Avalanche analysis | Power-law slope | Detected scale-free distributions |
| Weight decay tuning | Network stability | Balanced LTP/LTD for homeostasis |

### Generated Plots

Experiment results were visualized using scripts in `scripts/`. The following outputs are available in `output/`:

| Plot | Description |
|------|-------------|
| `learning_forgetting_sweep_heatmap.png` | Criticality heatmap for learning × forgetting rate |
| `learning_forgetting_sweep_plot.png` | Parameter sweep line plots |
| `heatmap.png` | Weight matrix visualization |
| `plots_snellius/` | HPC cluster experiment results |
| `plots_snellius_v2/` | Updated HPC simulation outputs |

## 8. Reflection

### What Worked Well

The most effective prompting strategy was **providing structured decision trees** rather than open-ended questions. When I asked the LLM to present options (e.g., "NumPy vs JAX backend?", "matplotlib vs pygame visualization?"), it produced well-reasoned trade-offs that I could evaluate against my conceptual understanding. This led to informed decisions with documented rationale.

**Skill-based guidance** proved invaluable for multi-session work. By encoding project conventions in `.github/skills/` files, the LLM automatically adhered to TDD workflows, code review practices, and commit message conventions without repeated instructions.

The **TDD approach** caught several issues early. For example, when implementing the STDP learning rule, unit tests revealed edge cases in weight updates near boundaries. Property-based tests using Hypothesis helped verify invariants across a wide range of inputs.

The **backend abstraction** allowed seamless development on CPU while enabling GPU acceleration for production runs—the Protocol-based interface made this transition transparent to the simulation code.

### Challenges Encountered

The main challenge was **parameter tuning for self-organized criticality**. Initial defaults produced either subcritical (dying activity) or supercritical (runaway firing) dynamics. This required extensive parameter sweeps on HPC infrastructure to find the critical regime.

Another issue was **balancing learning and forgetting rates**—the STDP rule needed careful tuning of LTP/LTD ratios and weight decay parameters to achieve stable yet plastic network dynamics.

### What I Would Do Differently

Next time, I would establish **quantitative acceptance criteria upfront** (e.g., "power-law slope between -1.2 and -1.8") rather than relying on qualitative inspection of avalanche distributions. This would enable fully automated validation of criticality.

### Lessons Learned

1. Explicit skill definitions significantly improve LLM consistency across sessions
2. Test-first prompting produces more reliable implementations than code-first approaches
3. Property-based testing (Hypothesis) excels at finding edge cases in numerical code
4. Backend abstraction enables development/production environment separation
5. Decision trees help capture design rationale and enable informed trade-off analysis
6. HPC infrastructure is essential for exploring high-dimensional parameter spaces in neural simulations
