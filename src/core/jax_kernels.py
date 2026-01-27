"""JAX-accelerated simulation kernels.

JIT-compiled functions for GPU-accelerated neural network simulation.
Uses float32 for Metal GPU compatibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

# Lazy JAX import for optional dependency
_jax = None
_jnp = None


def _ensure_jax():
    """Lazily import JAX modules."""
    global _jax, _jnp
    if _jax is None:
        try:
            import jax
            import jax.numpy as jnp
            _jax = jax
            _jnp = jnp
        except ImportError:
            raise ImportError(
                "JAX is required for GPU acceleration. "
                "Install with: pip install css-hnca[gpu]"
            )
    return _jax, _jnp


def is_jax_available() -> bool:
    """Check if JAX is available."""
    try:
        _ensure_jax()
        return True
    except ImportError:
        return False


def get_jax_device_info() -> str:
    """Get information about available JAX devices."""
    jax, _ = _ensure_jax()
    devices = jax.devices()
    device_strs = [f"{d.platform}:{d.device_kind}" for d in devices]
    return ", ".join(device_strs)


def create_jit_simulation_step():
    """Create JIT-compiled simulation step function.
    
    Returns:
        Tuple of (step_fn, apply_stdp_fn) that are JIT-compiled.
    """
    jax, jnp = _ensure_jax()

    @jax.jit
    def compute_input_and_fire(
        weight_matrix: jnp.ndarray,
        firing: jnp.ndarray,
        membrane_potential: jnp.ndarray,
        threshold: float,
        leak_rate: float,
        reset_potential: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute input signal and update firing state (JIT-compiled).
        
        Args:
            weight_matrix: (N, N) weight matrix
            firing: (N,) current firing state as float (0.0 or 1.0)
            membrane_potential: (N,) membrane potentials
            threshold: firing threshold
            leak_rate: LIF leak rate
            reset_potential: reset amount after firing
            
        Returns:
            Tuple of (new_firing, new_membrane_potential, input_signal)
        """
        # Compute input: v = W^T @ s
        input_signal = jnp.matmul(weight_matrix.T, firing)
        
        # LIF dynamics
        # 1. Leak
        potential = membrane_potential * (1.0 - leak_rate)
        # 2. Integrate
        potential = potential + input_signal
        # 3. Fire
        new_firing = (potential >= threshold).astype(jnp.float32)
        # 4. Reset
        potential = potential - reset_potential * new_firing
        # 5. Clamp
        potential = jnp.maximum(potential, 0.0)
        
        return new_firing, potential, input_signal

    @jax.jit
    def apply_stdp_jit(
        weights: jnp.ndarray,
        link_matrix: jnp.ndarray,
        firing_prev: jnp.ndarray,
        firing_current: jnp.ndarray,
        learning_rate: float,
        forgetting_rate: float,
        decay_alpha: float,
    ) -> jnp.ndarray:
        """Apply STDP learning rule (JIT-compiled).
        
        Args:
            weights: (N, N) weight matrix
            link_matrix: (N, N) boolean connectivity as float (0.0 or 1.0)
            firing_prev: (N,) previous firing state as float
            firing_current: (N,) current firing state as float
            learning_rate: LTP rate
            forgetting_rate: LTD rate
            decay_alpha: baseline weight decay
            
        Returns:
            Updated weight matrix
        """
        # Baseline decay
        new_weights = weights * (1.0 - decay_alpha)
        
        # LTP: A(t-1)=1 and B(t)=1 -> W[A,B] += l
        ltp_mask = jnp.outer(firing_prev, firing_current)
        # LTD: B(t-1)=1 and A(t)=1 -> W[A,B] -= f
        ltd_mask = jnp.outer(firing_current, firing_prev)
        
        # Apply only where links exist
        ltp_update = learning_rate * ltp_mask * link_matrix
        ltd_update = forgetting_rate * ltd_mask * link_matrix
        
        new_weights = new_weights + ltp_update - ltd_update
        
        return new_weights

    @jax.jit
    def clamp_weights(
        weights: jnp.ndarray,
        link_matrix: jnp.ndarray,
        inhibitory_mask: jnp.ndarray,
        weight_min: float,
        weight_max: float,
        weight_min_inh: float,
        weight_max_inh: float,
    ) -> jnp.ndarray:
        """Clamp weights per neuron type (JIT-compiled).
        
        Args:
            weights: (N, N) weight matrix
            link_matrix: (N, N) connectivity as float
            inhibitory_mask: (N,) boolean as float (1.0 for inhibitory)
            weight_min/max: bounds for excitatory
            weight_min_inh/max_inh: bounds for inhibitory
            
        Returns:
            Clamped weight matrix
        """
        # Per-row bounds based on presynaptic neuron type
        min_bounds = jnp.where(
            inhibitory_mask[:, None] > 0.5,
            weight_min_inh,
            weight_min
        )
        max_bounds = jnp.where(
            inhibitory_mask[:, None] > 0.5,
            weight_max_inh,
            weight_max
        )
        
        clamped = jnp.clip(weights, min_bounds, max_bounds)
        # Zero out non-links
        return jnp.where(link_matrix > 0.5, clamped, 0.0)

    return compute_input_and_fire, apply_stdp_jit, clamp_weights


class JAXSimulationState:
    """Holds JAX arrays for GPU-accelerated simulation."""
    
    def __init__(
        self,
        weight_matrix: np.ndarray,
        link_matrix: np.ndarray,
        firing: np.ndarray,
        membrane_potential: np.ndarray,
        inhibitory_nodes: np.ndarray,
        threshold: float,
        leak_rate: float,
        reset_potential: float,
        learning_rate: float,
        forgetting_rate: float,
        decay_alpha: float,
        weight_min: float,
        weight_max: float,
        weight_min_inh: float,
        weight_max_inh: float,
    ):
        """Initialize JAX state from NumPy arrays."""
        jax, jnp = _ensure_jax()
        
        # Convert to float32 for Metal compatibility
        self.weight_matrix = jnp.array(weight_matrix, dtype=jnp.float32)
        self.link_matrix = jnp.array(link_matrix, dtype=jnp.float32)
        self.firing = jnp.array(firing, dtype=jnp.float32)
        self.firing_prev = jnp.zeros_like(self.firing)
        self.membrane_potential = jnp.array(membrane_potential, dtype=jnp.float32)
        self.inhibitory_mask = jnp.array(inhibitory_nodes, dtype=jnp.float32)
        
        # Scalar parameters
        self.threshold = float(threshold)
        self.leak_rate = float(leak_rate)
        self.reset_potential = float(reset_potential)
        self.learning_rate = float(learning_rate)
        self.forgetting_rate = float(forgetting_rate)
        self.decay_alpha = float(decay_alpha)
        self.weight_min = float(weight_min)
        self.weight_max = float(weight_max)
        self.weight_min_inh = float(weight_min_inh)
        self.weight_max_inh = float(weight_max_inh)
        
        # Create JIT-compiled kernels
        self._compute_input_and_fire, self._apply_stdp, self._clamp_weights = \
            create_jit_simulation_step()
    
    def step(self) -> None:
        """Execute one simulation step on GPU."""
        # Save previous firing state
        self.firing_prev = self.firing
        
        # Compute input and update firing
        self.firing, self.membrane_potential, _ = self._compute_input_and_fire(
            self.weight_matrix,
            self.firing_prev,
            self.membrane_potential,
            self.threshold,
            self.leak_rate,
            self.reset_potential,
        )
        
        # Apply STDP
        self.weight_matrix = self._apply_stdp(
            self.weight_matrix,
            self.link_matrix,
            self.firing_prev,
            self.firing,
            self.learning_rate,
            self.forgetting_rate,
            self.decay_alpha,
        )
        
        # Clamp weights
        self.weight_matrix = self._clamp_weights(
            self.weight_matrix,
            self.link_matrix,
            self.inhibitory_mask,
            self.weight_min,
            self.weight_max,
            self.weight_min_inh,
            self.weight_max_inh,
        )
    
    def get_firing_count(self) -> int:
        """Get count of firing neurons."""
        _, jnp = _ensure_jax()
        return int(jnp.sum(self.firing))
    
    def get_average_weight(self) -> float:
        """Get average weight of connected neurons."""
        _, jnp = _ensure_jax()
        connected = self.weight_matrix * self.link_matrix
        n_links = jnp.sum(self.link_matrix)
        if n_links == 0:
            return 0.0
        return float(jnp.sum(connected) / n_links)
    
    def sync_to_numpy(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sync state back to NumPy arrays.
        
        Returns:
            Tuple of (weight_matrix, firing, membrane_potential) as NumPy arrays.
        """
        return (
            np.array(self.weight_matrix),
            np.array(self.firing) > 0.5,  # Convert back to bool
            np.array(self.membrane_potential),
        )
