"""Hebbian learning with STDP (Spike-Timing Dependent Plasticity).

Implements the STDP learning rule from the theory:
- LTP: w_AB += l if A(t)=1 and B(t+1)=1 (causal correlation)
- LTD: w_AB -= f if B(t)=1 and A(t+1)=1 (anti-causal correlation)

Plus weight decay mechanisms:
- Baseline decay: W *= (1 - α)
- Oja rule: W -= α_oja * activity² * W
"""

from dataclasses import dataclass, field

import numpy as np
from numpy import ndarray


@dataclass
class HebbianLearner:
    """Implements STDP learning rule with weight decay.

    Attributes:
        learning_rate: l - amount to increase weight on LTP
        forgetting_rate: f - amount to decrease weight on LTD
        weight_min: Minimum allowed weight
        weight_max: Maximum allowed weight
        decay_alpha: Baseline weight decay rate (default 0.0)
        oja_alpha: Oja rule decay coefficient (default 0.0)
    """

    learning_rate: float
    forgetting_rate: float
    weight_min: float
    weight_max: float
    decay_alpha: float = 0.0
    oja_alpha: float = 0.0

    def apply(
        self,
        weights: ndarray,
        link_matrix: ndarray,
        firing_prev: ndarray,
        firing_current: ndarray,
    ) -> ndarray:
        """Apply STDP learning rule with weight decay.

        STDP Rule for connection A -> B (weight W[A, B]):
        - LTP: W[A,B] += l if A fired at t-1 (prev) and B fires at t (current)
        - LTD: W[A,B] -= f if B fired at t-1 (prev) and A fires at t (current)

        Weight Decay:
        - Baseline: W *= (1 - decay_alpha)
        - Oja: W[A,B] -= oja_alpha * activity_B² * W[A,B]

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

        # 1. Baseline weight decay: W *= (1 - α)
        if self.decay_alpha > 0:
            new_weights *= (1 - self.decay_alpha)

        # Convert boolean arrays to float for broadcasting
        prev = firing_prev.astype(float)
        curr = firing_current.astype(float)

        # 2. STDP: LTP and LTD
        # LTP: A(t-1)=1 and B(t)=1 -> W[A,B] += l
        ltp_mask = np.outer(prev, curr)
        # LTD: B(t-1)=1 and A(t)=1 -> W[A,B] -= f
        ltd_mask = np.outer(curr, prev)

        # Apply learning only where structural links exist
        ltp_update = self.learning_rate * ltp_mask * link_matrix
        ltd_update = self.forgetting_rate * ltd_mask * link_matrix

        new_weights = new_weights + ltp_update - ltd_update

        # 3. Oja rule: W[A,B] -= oja_alpha * activity_B² * W[A,B]
        # activity_B is the postsynaptic activity (firing_current[B])
        if self.oja_alpha > 0:
            # For each connection A->B, decay by oja_alpha * curr[B]² * W[A,B]
            # Broadcast: curr² is (N,), we want to apply to each column
            activity_sq = curr ** 2  # (N,)
            oja_decay = self.oja_alpha * activity_sq  # (N,)
            # Apply to each column j: new_weights[:, j] -= oja_decay[j] * new_weights[:, j]
            new_weights -= new_weights * oja_decay[np.newaxis, :]

        # Enforce bounds
        new_weights = np.clip(new_weights, self.weight_min, self.weight_max)

        # Ensure no weights where no links
        new_weights = np.where(link_matrix, new_weights, 0.0)

        return new_weights
