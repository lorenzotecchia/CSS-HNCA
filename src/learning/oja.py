"""Oja's rule for weight normalization.

Implements the Oja rule: W[A,B] -= α_oja * activity_B² * W[A,B]
"""

import numpy as np
from numpy import ndarray


def apply_oja(
    weights: ndarray,
    link_matrix: ndarray,
    firing_current: ndarray,
    oja_alpha: float,
) -> ndarray:
    """Apply Oja's rule for weight normalization.

    Oja rule: W[A,B] -= oja_alpha * activity_B² * W[A,B]
    activity_B is the postsynaptic activity (firing_current[B])

    Args:
        weights: Current weight matrix (N, N), W[i,j] = weight from i to j
        link_matrix: Structural connectivity (N, N) bool
        firing_current: Firing state at t (N,)
        oja_alpha: Oja rule decay coefficient

    Returns:
        Updated weight matrix after Oja rule
    """
    if oja_alpha <= 0:
        return weights.copy()

    # Convert to float for broadcasting
    curr = firing_current.astype(float)

    # activity_B is curr[B]
    activity_sq = curr ** 2  # (N,)

    # For each connection A->B, decay by oja_alpha * curr[B]² * W[A,B]
    # Broadcast: apply to each column j
    oja_decay = oja_alpha * activity_sq  # (N,)
    new_weights = weights - weights * oja_decay[np.newaxis, :]

    return new_weights