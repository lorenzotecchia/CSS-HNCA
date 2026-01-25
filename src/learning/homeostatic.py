"""Homeostatic scaling for spiking frequency regulation.

Adjusts incoming weights based on spiking frequency over a time window.
- If a neuron spikes too little, increase all incoming weights
- If a neuron spikes too much, decrease all incoming weights
"""

from collections import deque

import numpy as np
from numpy import ndarray


def apply_homeostatic(
    weights: ndarray,
    link_matrix: ndarray,
    firing_current: ndarray,
    spike_history: deque,
    spike_timespan: int,
    min_spike_amount: int,
    max_spike_amount: int,
    weight_change_constant: float,
) -> ndarray:
    """Apply homeostatic scaling.

    For each neuron B, count spikes in last spike_timespan steps.
    - If count < min_spike_amount, increase all incoming weights by weight_change_constant
    - If count > max_spike_amount, decrease all incoming weights by weight_change_constant

    Args:
        weights: Current weight matrix (N, N), W[i,j] = weight from i to j
        link_matrix: Structural connectivity (N, N) bool
        firing_current: Firing state at t (N,)
        spike_history: Deque of recent firing states
        spike_timespan: Time window for spike counting
        min_spike_amount: Minimum spikes required
        max_spike_amount: Maximum spikes allowed
        weight_change_constant: Amount to adjust weights

    Returns:
        Updated weight matrix after homeostatic scaling
    """
    # Update spike history
    spike_history.append(firing_current.copy())

    if len(spike_history) < spike_timespan:
        return weights.copy()

    # Compute spike counts for each neuron over the last spike_timespan steps
    history_array = np.array(spike_history)  # (timespan, N)
    spike_counts = np.sum(history_array, axis=0)  # (N,)

    new_weights = weights.copy()

    for j in range(weights.shape[1]):
        if spike_counts[j] < min_spike_amount:
            # Increase all incoming weights to j
            new_weights[:, j] += weight_change_constant * link_matrix[:, j]
        elif spike_counts[j] > max_spike_amount:
            # Decrease all incoming weights to j
            new_weights[:, j] -= weight_change_constant * link_matrix[:, j]

    return new_weights