"""NeuronState data structure for neural cellular automata.

Represents the firing state of neurons at each time step.
"""

from dataclasses import dataclass

import numpy as np
from numpy import ndarray

from src.core.backend import ArrayBackend, get_backend


@dataclass
class NeuronState:
    """Firing state of neurons in the network.

    Attributes:
        firing: Shape (N,) - current firing state (bool)
        firing_prev: Shape (N,) - previous time step firing state (bool)
        threshold: Firing threshold γ
        membrane_potential: Shape (N,) - LIF membrane potential (float)
        leak_rate: LIF leak rate λ (potential decays by this fraction each step)
        reset_potential: Amount subtracted from potential after firing
    """

    firing: ndarray
    firing_prev: ndarray
    threshold: float
    membrane_potential: ndarray
    leak_rate: float
    reset_potential: float

    @classmethod
    def create(
        cls,
        n_neurons: int,
        threshold: float,
        initial_firing_fraction: float,
        seed: int | None = None,
        leak_rate: float = 0.0,
        reset_potential: float = 0.0,
        backend: ArrayBackend | None = None,
    ) -> "NeuronState":
        """Create initial neuron state with random firing pattern.

        Args:
            n_neurons: Number of neurons
            threshold: Firing threshold γ (must be >= 0)
            initial_firing_fraction: Fraction of neurons firing at t=0 (0 to 1)
            seed: Random seed for reproducibility
            leak_rate: LIF leak rate λ in [0, 1] (potential decay fraction)
            reset_potential: Amount subtracted from potential after firing (>= 0)
            backend: Array backend for computation (default: NumPy)

        Returns:
            NeuronState with random initial firing based on fraction

        Raises:
            ValueError: If threshold < 0 or initial_firing_fraction not in [0, 1]
                       or leak_rate not in [0, 1] or reset_potential < 0
        """
        if threshold < 0:
            raise ValueError(f"Threshold must be >= 0, got {threshold}")
        if not 0 <= initial_firing_fraction <= 1:
            raise ValueError(
                f"initial_firing_fraction must be in [0, 1], got {initial_firing_fraction}"
            )
        if not 0 <= leak_rate <= 1:
            raise ValueError(f"leak_rate must be in [0, 1], got {leak_rate}")
        if reset_potential < 0:
            raise ValueError(f"reset_potential must be >= 0, got {reset_potential}")

        if backend is None:
            backend = get_backend()

        # Create initial firing state based on fraction
        firing = backend.random_bool(initial_firing_fraction, (n_neurons,), seed)

        # Previous state starts all False
        firing_prev = backend.zeros((n_neurons,), dtype=np.bool_)

        # Membrane potential: set to threshold for initially firing neurons
        # This ensures they can contribute input before the first update
        # Use where() instead of in-place assignment for JAX compatibility
        zeros = backend.zeros((n_neurons,), dtype=np.float64)
        threshold_arr = zeros + threshold
        membrane_potential = backend.where(firing, threshold_arr, zeros)

        return cls(
            firing=backend.to_numpy(firing),
            firing_prev=backend.to_numpy(firing_prev),
            threshold=threshold,
            membrane_potential=backend.to_numpy(membrane_potential),
            leak_rate=leak_rate,
            reset_potential=reset_potential,
        )

    def update_firing(self, input_signal: ndarray) -> None:
        """Update firing state using Leaky Integrate-and-Fire dynamics.

        LIF Model:
            V(t+1) = (1 - λ) * V(t) + input - V_reset * fired(t)
        
        Neurons fire if their membrane potential >= threshold.
        Previous firing state is preserved before update.

        Args:
            input_signal: Shape (N,) - weighted sum of inputs for each neuron
        """
        # Preserve current state as previous
        np.copyto(self.firing_prev, self.firing)

        # LIF dynamics:
        # 1. Leak: decay toward zero
        self.membrane_potential *= (1 - self.leak_rate)
        
        # 2. Integrate: add input signal
        self.membrane_potential += input_signal
        
        # 3. Fire: check threshold
        np.copyto(self.firing, self.membrane_potential >= self.threshold)
        
        # 4. Reset: subtract reset_potential for neurons that fired
        self.membrane_potential -= self.reset_potential * self.firing.astype(float)
        
        # Clamp potential to non-negative (biological constraint)
        np.maximum(self.membrane_potential, 0.0, out=self.membrane_potential)
