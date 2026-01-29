# Repository Cleanup & Documentation Design

**Date:** 2026-01-29
**Purpose:** Prepare CSS-HNCA for public release and academic submission

## Goals

1. Professional documentation suitable for researchers and users
2. Clear code organization with explained components
3. Clean git history without unprofessional commit messages
4. Reproducible results with sample output

---

## 1. Documentation Structure

### README.md (Quick Reference)
- Project title + badges (tests, license)
- 2-3 paragraph scientific introduction (HNCA, SOC, motivation)
- Quick install instructions
- Basic usage examples (3-4 commands)
- Key parameters table
- Links to detailed docs

### docs/ Folder (Academic Style)
```
docs/
├── introduction.md      # Scientific background, motivation, research questions
├── methods.md           # Network model, STDP, LIF dynamics, weight decay
├── architecture.md      # Code structure, components, data flow diagrams
├── configuration.md     # All parameters with descriptions and valid ranges
├── usage.md             # CLI options, headless mode, HPC deployment
└── results.md           # Output interpretation, sample analysis, visualization
```

### Content Migration
- `memory/plan.md` design decisions → `docs/architecture.md`
- `docs/css-theory.md` → merge into `docs/methods.md`
- Delete `memory/` folder after migration

---

## 2. Scripts & Examples Organization

### examples/ (Demos)
```
examples/
├── README.md                    # Overview with expected output
├── basic_simulation.py          # Minimal working example
├── realtime_visualization.py    # matplotlib demo (from demo_matplotlib.py)
└── avalanche_analysis.py        # SOC metrics demo (from demo_avalanche_analytics.py)
```

### scripts/ (Analysis Tools)
```
scripts/
├── README.md                    # Tool descriptions and usage
├── parameter_sweep.py           # SOC sweep (from soc_parameter_sweep.py)
├── learning_rate_sweep.py       # LHS plasticity sweep (from miao.py)
├── plot_timeseries.py           # CSV plotting (from plot_csv.py)
└── plot_weight_heatmap.py       # Weight matrix viz (from plot_heatmap.py)
```

### File Actions
| Current Name | Action |
|--------------|--------|
| `demo_matplotlib.py` | Move to `examples/realtime_visualization.py` |
| `demo_avalanche_analytics.py` | Move to `examples/avalanche_analysis.py` |
| `soc_parameter_sweep.py` | Move to `scripts/parameter_sweep.py` |
| `miao.py` | Rename to `scripts/learning_rate_sweep.py` |
| `plot_csv.py` | Rename to `scripts/plot_timeseries.py` |
| `plot_heatmap.py` | Rename to `scripts/plot_weight_heatmap.py` |
| `avg_weight.py` | Review - integrate or delete |

---

## 3. Output Handling

### Structure
```
output/
├── .gitkeep
├── README.md              # Explains format, why ignored, how to reproduce
└── sample/
    ├── README.md          # Column descriptions
    └── timeseries_example.csv
```

### Gitignore Additions
```gitignore
output/*.csv
output/*.npz
output/**/*.csv
output/**/*.npz
!output/sample/
```

### Deletions
- `output_snellius/` - HPC-specific archives not needed for release

---

## 4. Git Cleanup

### Approach
1. Create branch `release/v1.0` from `main`
2. Interactive rebase to squash/reword unprofessional commits only:
   - `6407b04` ":-)
   - `a13e235` "scemo chi legge"
   - `514183e` "miao"
3. Keep all other commits with original messages
4. Tag as `v1.0.0`

### Files to Remove Before Release
- `text_presentation.md` (untracked scratch file)
- `memory/` folder (after content migration)

---

## 5. Implementation Order

1. **Documentation first** - Write all docs before restructuring
2. **Migrate memory content** - Extract valuable parts to proper docs
3. **Reorganize scripts** - Create examples/, rename files
4. **Setup output** - Sample files, gitignore, README
5. **Git cleanup** - Create release branch, rebase, tag
6. **Final review** - Verify all tests pass, docs render correctly

---

## Acceptance Criteria

- [ ] README provides quick start in under 2 minutes
- [ ] All docs cover their topics completely
- [ ] Examples run without modification
- [ ] Scripts have clear usage instructions
- [ ] Sample output demonstrates expected format
- [ ] No unprofessional commit messages in release branch
- [ ] All 212 tests still pass
- [ ] Repository is suitable for academic citation
