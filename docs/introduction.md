# Introduction

## Overview

CSS-HNCA (Complex System Simulation for Hebbian Neural Cellular Automaton) is a computational neuroscience framework for simulating self-organized criticality (SOC) in neural networks. The simulation models spiking neurons with spike-timing-dependent plasticity (STDP) to study how neural systems naturally evolve toward critical dynamics.

## Scientific Background

### Self-Organized Criticality

Self-organized criticality (SOC) describes systems that naturally evolve toward a critical state where activity propagates as scale-free avalanches. In neural systems, criticality is hypothesized to optimize information processing, dynamic range, and computational capacity.

Key signatures of criticality:
- **Power-law avalanche distributions**: Avalanche sizes follow P(s) ~ s^(-τ) with τ ≈ 1.5
- **Branching ratio ≈ 1.0**: On average, each firing neuron triggers exactly one other neuron
- **Long-range temporal correlations**: Activity exhibits 1/f noise characteristics

### Hebbian Learning and STDP

Spike-timing-dependent plasticity (STDP) is a biologically observed learning rule where synaptic strength changes based on the relative timing of pre- and postsynaptic spikes:

- **Long-term potentiation (LTP)**: If neuron A fires before neuron B, the synapse A→B strengthens
- **Long-term depression (LTD)**: If neuron B fires before neuron A, the synapse A→B weakens

This temporal asymmetry allows networks to learn causal relationships and develop structured connectivity.

## Research Applications

This framework enables investigation of:
- Parameter regimes that produce critical dynamics
- Interactions between plasticity rules and network topology
- Stability mechanisms (weight decay, Oja rule) in plastic networks
- Avalanche statistics as biomarkers for criticality

## References

1. Beggs, J.M. & Plenz, D. (2003). Neuronal avalanches in neocortical circuits. *Journal of Neuroscience*, 23(35), 11167-11177.
2. Bi, G.Q. & Poo, M.M. (1998). Synaptic modifications in cultured hippocampal neurons. *Journal of Neuroscience*, 18(24), 10464-10472.
3. Oja, E. (1982). Simplified neuron model as a principal component analyzer. *Journal of Mathematical Biology*, 15(3), 267-273.
