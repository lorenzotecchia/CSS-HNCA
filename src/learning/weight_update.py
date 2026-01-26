"""Weight update orchestrator for neural plasticity rules.

Combines STDP, Oja's rule, and homeostatic scaling with options to enable/disable each.
Handles inhibitory connections (negative weights) correctly.
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from numpy import ndarray

from .homeostatic import apply_homeostatic
from .oja import apply_oja
from .stdp import apply_stdp


@dataclass
class WeightUpdater:
    """Orchestrates weight updates with multiple plasticity rules.

    Attributes:
        enable_stdp: Whether to apply STDP (default True)
        enable_oja: Whether to apply Oja's rule (default False)
        enable_homeostatic: Whether to apply homeostatic scaling (default False)
        learning_rate: STDP LTP rate
        forgetting_rate: STDP LTD rate
        decay_alpha: Baseline weight decay rate (default 0.0)
        oja_alpha: Oja rule coefficient
        spike_timespan: Homeostatic time window
        min_spike_amount: Homeostatic min spikes
        max_spike_amount: Homeostatic max spikes
        weight_change_constant: Homeostatic adjustment amount
        weight_min: Minimum weight for excitatory neurons
        weight_max: Maximum weight for excitatory neurons
        weight_min_inh: Minimum weight for inhibitory neurons
        weight_max_inh: Maximum weight for inhibitory neurons
        spike_history: Internal spike history for homeostatic
    """

    enable_stdp: bool = True
    enable_oja: bool = False
    enable_homeostatic: bool = False
    learning_rate: float = 0.01
    forgetting_rate: float = 0.005
    decay_alpha: float = 0.0
    oja_alpha: float = 0.0
    spike_timespan: int = 100
    min_spike_amount: int = 5
    max_spike_amount: int = 15
    weight_change_constant: float = 0.01
    weight_min: float = 0.0
    weight_max: float = 1.0
    weight_min_inh: float = -1.0
    weight_max_inh: float = 0.0
    spike_history: deque = field(default_factory=deque)

    def __post_init__(self):
        """Initialize spike_history with correct maxlen."""
        self.spike_history = deque(self.spike_history, maxlen=self.spike_timespan)

    def apply(
        self,
        weights: ndarray,
        link_matrix: ndarray,
        firing_prev: ndarray,
        firing_current: ndarray,
        inhibitory_nodes: ndarray,
    ) -> ndarray:
        """Apply enabled plasticity rules to update weights.

        Handles inhibitory connections by adjusting updates based on inhibitory_nodes.

        Args:
            weights: Current weight matrix (N, N)
            link_matrix: Structural connectivity (N, N) bool
            firing_prev: Firing state at t-1 (N,)
            firing_current: Firing state at t (N,)
            inhibitory_nodes: Boolean array indicating inhibitory neurons (N,)

        Returns:
            Updated weight matrix with bounds enforced
        """
        new_weights = weights.copy()

        # Baseline weight decay
        if self.decay_alpha > 0:
            new_weights *= (1 - self.decay_alpha)

        # Apply STDP if enabled
        if self.enable_stdp:
            new_weights = apply_stdp(
                new_weights, link_matrix, firing_prev, firing_current,
                self.learning_rate, self.forgetting_rate
            )

        # Apply Oja's rule if enabled
        if self.enable_oja:
            new_weights = apply_oja(
                new_weights, link_matrix, firing_current, self.oja_alpha
            )

        # Apply homeostatic scaling if enabled
        if self.enable_homeostatic:
            new_weights = apply_homeostatic(
                new_weights, link_matrix, firing_current, self.spike_history,
                self.spike_timespan, self.min_spike_amount, self.max_spike_amount,
                self.weight_change_constant
            )

        # Vectorized per-type weight clamping
        neuron_types = ~inhibitory_nodes  # True = excitatory
        min_bounds = np.where(neuron_types, self.weight_min, self.weight_min_inh)[:, np.newaxis]
        max_bounds = np.where(neuron_types, self.weight_max, self.weight_max_inh)[:, np.newaxis]
        new_weights = np.clip(new_weights, min_bounds, max_bounds)

        # Ensure no weights where no links
        new_weights = np.where(link_matrix, new_weights, 0.0)

        return new_weights