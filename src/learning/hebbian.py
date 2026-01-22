"""Hebbian learning with STDP (Spike-Timing Dependent Plasticity).

Implements the STDP learning rule from the theory:
- LTP: w_AB += l if A(t)=1 and B(t+1)=1 (causal correlation)
- LTD: w_AB -= f if B(t)=1 and A(t+1)=1 (anti-causal correlation)
"""

from dataclasses import dataclass

import numpy as np
from numpy import ndarray


@dataclass
class HebbianLearner:
    """Implements STDP learning rule.

    Attributes:
        learning_rate: l - amount to increase weight on LTP
        forgetting_rate: f - amount to decrease weight on LTD
        weight_min: Minimum allowed weight
        weight_max: Maximum allowed weight
    """

    learning_rate: float
    forgetting_rate: float
    weight_min: float
    weight_max: float

    def apply(
        self,
        weights: ndarray,
        link_matrix: ndarray,
        firing_prev: ndarray,
        firing_current: ndarray,
    ) -> ndarray:
        """Apply STDP learning rule to update weights.

        STDP Rule for connection A -> B (weight W[A, B]):
        - LTP: W[A,B] += l if A fired at t-1 (prev) and B fires at t (current)
        - LTD: W[A,B] -= f if B fired at t-1 (prev) and A fires at t (current)

        Args:
            weights: Current weight matrix (N, N), W[i,j] = weight from i to j
            link_matrix: Structural connectivity (N, N) bool
            firing_prev: Firing state at t-1 (N,)
            firing_current: Firing state at t (N,)

        Returns:
            Updated weight matrix with bounds enforced
        """
        # Start with a copy of current weights
        new_weights = weights.copy()

        # Convert boolean arrays to float for broadcasting
        prev = firing_prev.astype(float)
        curr = firing_current.astype(float)

        # LTP: A(t-1)=1 and B(t)=1 -> W[A,B] += l
        # For each connection A->B (W[A,B]), LTP applies when:
        #   firing_prev[A]=1 and firing_current[B]=1
        # Using outer product: ltp_mask[i,j] = prev[i] * curr[j]
        ltp_mask = np.outer(prev, curr)

        # LTD: B(t-1)=1 and A(t)=1 -> W[A,B] -= f
        # For each connection A->B (W[A,B]), LTD applies when:
        #   firing_prev[B]=1 and firing_current[A]=1
        # Using outer product: ltd_mask[i,j] = curr[i] * prev[j]
        ltd_mask = np.outer(curr, prev)

        # Apply learning only where structural links exist
        ltp_update = self.learning_rate * ltp_mask * link_matrix
        ltd_update = self.forgetting_rate * ltd_mask * link_matrix

        # Update weights: +LTP, -LTD
        new_weights = new_weights + ltp_update - ltd_update

        # Enforce bounds
        new_weights = np.clip(new_weights, self.weight_min, self.weight_max)

        # Ensure no weights where no links (may have been introduced by numerical issues)
        new_weights = np.where(link_matrix, new_weights, 0.0)

        return new_weights
