# Methods

## Network Model

### Spatial Organization

Neurons are positioned in a 3D volume with coordinates randomly sampled from a uniform distribution within a bounding box. Connectivity is distance-dependent: neurons within radius r of each other may form synaptic connections.

### Connectivity Structure

The network uses directed, weighted connections. The link matrix L ∈ {0,1}^(n×n) encodes which connections exist, and the weight matrix W ∈ ℝ^(n×n) stores synaptic strengths.

For connection A→B:
- L[A,B] = 1 if distance(A,B) < radius
- W[A,B] = synaptic weight from A to B

### Neuron Types

Neurons are classified as excitatory or inhibitory:
- **Excitatory neurons**: Positive outgoing weights, bounded by [w_min, w_max]
- **Inhibitory neurons**: Negative outgoing weights, bounded by [w_min_inh, w_max_inh]

The excitatory fraction parameter controls the proportion of excitatory neurons.

## Neuron Dynamics

Neurons can be in either one of two states: firing and not-firing. 
The state s_j of a neuron j at time t is: 
``` 
s_j(t) = 0 if j is not-firing at time t
s_j(t) = 1 if j is firing at time t
```
When the network is generated, all neurons are at state 0. At the start of the simulation, a fraction of firing neurons is set to state 1.
If the activity dies out (all neurons are at state 0), a new input of firing neurons is given to the same network, preserving its weights and connections.

The vector state s_j(t) is updated deterministically throughout the simulation using the following rules.

### Leaky Integrate-and-Fire (LIF) Model

Each neuron maintains a membrane potential V that integrates incoming activity and decays over time:

```
V_i(t+1) = (1 - λ) · V_i(t) + Σ_j(w_ji · s_j(t)) - V_reset · fired_i(t)
```

Where:
- λ = leak rate (decay toward resting potential)
- V_reset = reset amount after firing
- s_j(t) = firing state of neuron j at time t

A neuron fires when its membrane potential exceeds the threshold:
```
s_i(t) = 1 if V_i(t) ≥ γ, else 0
```

### State Update

In matrix form, the input to each neuron is computed as:
```
v(t) = W^T · s(t-1)
```

## Plasticity Rules

### Spike-Timing-Dependent Plasticity (STDP)

For a synapse A→B with weight w_AB:

```
w_AB(t+1) = w_AB(t) + l   if A fires at t-1 AND B fires at t  (LTP)
w_AB(t+1) = w_AB(t) - f   if A fires at t AND B fires at t-1  (LTD)
w_AB(t+1) = w_AB(t)       otherwise
```

Where l = learning rate and f = forgetting rate.

### Weight Decay Mechanisms

To prevent unbounded weight growth, two decay mechanisms are implemented:

**Baseline Decay:**
```
W(t+1) = (1 - α) · W(t)
```

**Oja Rule (activity-dependent):**
```
w_AB(t+1) -= α_oja · activity_B² · w_AB
```

The Oja rule provides competitive normalization where synapses compete based on postsynaptic activity.

### Weight Bounds

Weights are constrained to prevent sign changes:
- Excitatory: 0 ≤ w ≤ w_max
- Inhibitory: w_min_inh ≤ w ≤ 0

## Avalanche Detection

### Definition

An avalanche is a burst of activity above a quiet threshold:

1. **Quiet state**: firing_count < quiet_threshold × n_neurons
2. **Avalanche start**: firing rises above quiet threshold
3. **Avalanche end**: firing returns below quiet threshold

### Metrics

- **Size**: Total number of spikes during avalanche
- **Duration**: Number of time steps
- **Peak**: Maximum firing count

### Criticality Indicators

- **Power-law slope**: Log-log regression of size distribution (critical: τ ≈ -1.5)
- **Branching ratio**: mean(firing(t+1) / firing(t)) during avalanches (critical: σ ≈ 1.0)
