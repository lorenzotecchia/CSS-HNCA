"""Hebbian learning with STDP (Spike-Timing Dependent Plasticity).

Implements the STDP learning rule from the theory:
- LTP: w_AB += l if A(t)=1 and B(t+1)=1 (causal correlation)
- LTD: w_AB -= f if B(t)=1 and A(t+1)=1 (anti-causal correlation)

Plus weight decay mechanisms:
- Baseline decay: W *= (1 - α)
- Oja rule: W -= α_oja * activity² * W

And homeostatic scaling:
- Adjusts incoming weights based on spiking frequency over a time window.
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from numpy import ndarray

from src.core.backend import ArrayBackend, get_backend


@dataclass
class HebbianLearner:
    """Implements STDP learning rule with weight decay and homeostatic scaling.

    Attributes:
        learning_rate: l - amount to increase weight on LTP
        forgetting_rate: f - amount to decrease weight on LTD
        weight_min: Minimum allowed weight for excitatory neurons
        weight_max: Maximum allowed weight for excitatory neurons
        decay_alpha: Baseline weight decay rate (default 0.0)
        oja_alpha: Oja rule decay coefficient (default 0.0)
        weight_min_inh: Minimum allowed weight for inhibitory neurons
        weight_max_inh: Maximum allowed weight for inhibitory neurons
        enable_homeostatic: Whether to apply homeostatic scaling (default False)
        spike_timespan: Time window for spike counting in homeostatic scaling
        min_spike_amount: Minimum spikes required in timespan
        max_spike_amount: Maximum spikes allowed in timespan
        weight_change_constant: Amount to adjust weights in homeostatic scaling
        spike_history: Deque of recent firing states (internal)
        backend: Array computation backend
    """

    learning_rate: float
    forgetting_rate: float
    weight_min: float
    weight_max: float
    decay_alpha: float = 0.0
    oja_alpha: float = 0.0
    weight_min_inh: float = -0.3
    weight_max_inh: float = 0.0
    enable_homeostatic: bool = False
    spike_timespan: int = 30
    min_spike_amount: int = 5
    max_spike_amount: int = 10
    weight_change_constant: float = 0.01
    spike_history: deque = field(default_factory=deque)
    backend: ArrayBackend = field(default_factory=get_backend)

    def __post_init__(self):
        """Initialize spike_history with correct maxlen."""
        self.spike_history = deque(self.spike_history, maxlen=self.spike_timespan)

    def apply(
        self,
        weights: ndarray,
        link_matrix: ndarray,
        firing_prev: ndarray,
        firing_current: ndarray,
        inhibitory_nodes: ndarray | None = None,
    ) -> ndarray:
        """Apply STDP learning rule with weight decay and homeostatic scaling.

        STDP Rule for connection A -> B (weight W[A, B]):
        - LTP: W[A,B] += l if A fired at t-1 (prev) and B fires at t (current)
        - LTD: W[A,B] -= f if B fired at t-1 (prev) and A fires at t (current)

        Weight Decay:
        - Baseline: W *= (1 - decay_alpha)
        - Oja: W[A,B] -= oja_alpha * activity_B² * W[A,B]

        Homeostatic Scaling (if enabled):
        - For each neuron B, count spikes in last spike_timespan steps
        - If count < min_spike_amount, increase all incoming weights
        - If count > max_spike_amount, decrease all incoming weights

        Args:
            weights: Current weight matrix (N, N), W[i,j] = weight from i to j
            link_matrix: Structural connectivity (N, N) bool
            firing_prev: Firing state at t-1 (N,)
            firing_current: Firing state at t (N,)
            inhibitory_nodes: Boolean array (N,) - True=inhibitory.
                              If None, all neurons treated as excitatory.

        Returns:
            Updated weight matrix with bounds enforced
        """
        # Derive neuron_types from inhibitory_nodes
        if inhibitory_nodes is not None:
            neuron_types = ~inhibitory_nodes  # True = excitatory
        else:
            neuron_types = None

        # Start with a copy of current weights
        new_weights = weights.copy()

        # 1. Baseline weight decay: W *= (1 - α)
        if self.decay_alpha > 0:
            new_weights *= 1 - self.decay_alpha

        # Convert boolean arrays to float for broadcasting
        prev = firing_prev.astype(float)
        curr = firing_current.astype(float)

        # 2. STDP: LTP and LTD
        ltp_mask = np.outer(prev, curr)
        ltd_mask = np.outer(curr, prev)

        ltp_update = self.learning_rate * ltp_mask * link_matrix
        ltd_update = self.forgetting_rate * ltd_mask * link_matrix

        new_weights = new_weights + ltp_update - ltd_update

        # 3. Oja rule
        if self.oja_alpha > 0:
            activity_sq = curr**2
            oja_decay = self.oja_alpha * activity_sq
            new_weights -= new_weights * oja_decay[np.newaxis, :]

        # 4. Enforce per-type bounds (vectorized)
        if neuron_types is not None:
            min_bounds = np.where(neuron_types, self.weight_min, self.weight_min_inh)
            max_bounds = np.where(neuron_types, self.weight_max, self.weight_max_inh)
            min_bounds_matrix = min_bounds[:, np.newaxis]
            max_bounds_matrix = max_bounds[:, np.newaxis]
            new_weights = np.clip(new_weights, min_bounds_matrix, max_bounds_matrix)
        else:
            new_weights = np.clip(new_weights, self.weight_min, self.weight_max)

        # Ensure no weights where no links
        new_weights = self.backend.where(link_matrix, new_weights, 0.0)

        # Update spike history for homeostatic scaling
        self.spike_history.append(firing_current.copy())

        # 5. Homeostatic scaling
        if self.enable_homeostatic and len(self.spike_history) == self.spike_timespan:
            history_array = np.array(self.spike_history)
            spike_counts = np.sum(history_array, axis=0)

            for j in range(weights.shape[1]):
                if spike_counts[j] < self.min_spike_amount:
                    new_weights[:, j] += self.weight_change_constant * link_matrix[:, j]
                elif spike_counts[j] > self.max_spike_amount:
                    new_weights[:, j] -= self.weight_change_constant * link_matrix[:, j]

            # Re-enforce bounds after homeostatic adjustment
            if neuron_types is not None:
                new_weights = np.clip(new_weights, min_bounds_matrix, max_bounds_matrix)
            else:
                new_weights = np.clip(new_weights, self.weight_min, self.weight_max)
            new_weights = np.where(link_matrix, new_weights, 0.0)

        return new_weights
