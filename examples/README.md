# Examples

Demonstration scripts showing how to use CSS-HNCA.

## basic_simulation.py

Minimal example running a headless simulation and printing metrics.

```bash
python examples/basic_simulation.py
```

**Output:**
```
Running 1000 steps...
Step 100: firing=42, avg_weight=0.0612
Step 200: firing=38, avg_weight=0.0608
...
Final: 1000 steps, avg firing=35.2, final avg_weight=0.0589
```

## realtime_visualization.py

Real-time matplotlib plots showing firing count, weight evolution, and weight distribution.

```bash
python examples/realtime_visualization.py
```

Requires display. Shows three subplots updated each simulation step.

## avalanche_analysis.py

Demonstrates avalanche detection and SOC metrics computation.

```bash
python examples/avalanche_analysis.py
```

**Output:**
- Avalanche size distribution
- Branching ratio over time
- Power-law fit statistics
