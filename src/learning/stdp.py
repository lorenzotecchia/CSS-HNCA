"""STDP (Spike-Timing Dependent Plasticity) learning rule.

Implements the STDP rule:
- LTP: w_AB += l if A(t)=1 and B(t+1)=1 (causal correlation)
- LTD: w_AB -= f if B(t)=1 and A(t+1)=1 (anti-causal correlation)
"""

import numpy as np
from numpy import ndarray


def apply_stdp(
    weights: ndarray,
    link_matrix: ndarray,
    firing_prev: ndarray,
    firing_current: ndarray,
    learning_rate: float,
    forgetting_rate: float,
) -> ndarray:
    """Apply STDP learning rule.

    STDP Rule for connection A -> B (weight W[A, B]):
    - LTP: W[A,B] += l if A fired at t-1 (prev) and B fires at t (current)
    - LTD: W[A,B] -= f if B fired at t-1 (prev) and A fires at t (current)

    Args:
        weights: Current weight matrix (N, N), W[i,j] = weight from i to j
        link_matrix: Structural connectivity (N, N) bool
        firing_prev: Firing state at t-1 (N,)
        firing_current: Firing state at t (N,)
        learning_rate: l - amount to increase weight on LTP
        forgetting_rate: f - amount to decrease weight on LTD

    Returns:
        Updated weight matrix after STDP
    """
    # Convert boolean arrays to float for broadcasting
    prev = firing_prev.astype(float)
    curr = firing_current.astype(float)

    # LTP: A(t-1)=1 and B(t)=1 -> W[A,B] += l
    ltp_mask = np.outer(prev, curr)
    # LTD: B(t-1)=1 and A(t)=1 -> W[A,B] -= f
    ltd_mask = np.outer(curr, prev)

    # Apply learning only where structural links exist
    ltp_update = learning_rate * ltp_mask * link_matrix
    ltd_update = forgetting_rate * ltd_mask * link_matrix

    new_weights = weights + ltp_update - ltd_update

    return new_weights