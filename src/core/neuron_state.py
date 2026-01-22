"""NeuronState data structure for neural cellular automata.

Represents the firing state of neurons at each time step.
"""

from dataclasses import dataclass

import numpy as np
from numpy import ndarray


@dataclass
class NeuronState:
    """Firing state of neurons in the network.

    Attributes:
        firing: Shape (N,) - current firing state (bool)
        firing_prev: Shape (N,) - previous time step firing state (bool)
        threshold: Firing threshold γ
    """

    firing: ndarray
    firing_prev: ndarray
    threshold: float

    @classmethod
    def create(
        cls,
        n_neurons: int,
        threshold: float,
        initial_firing_fraction: float,
        seed: int | None = None,
    ) -> "NeuronState":
        """Create initial neuron state with random firing pattern.

        Args:
            n_neurons: Number of neurons
            threshold: Firing threshold γ (must be >= 0)
            initial_firing_fraction: Fraction of neurons firing at t=0 (0 to 1)
            seed: Random seed for reproducibility

        Returns:
            NeuronState with random initial firing based on fraction

        Raises:
            ValueError: If threshold < 0 or initial_firing_fraction not in [0, 1]
        """
        if threshold < 0:
            raise ValueError(f"Threshold must be >= 0, got {threshold}")
        if not 0 <= initial_firing_fraction <= 1:
            raise ValueError(
                f"initial_firing_fraction must be in [0, 1], got {initial_firing_fraction}"
            )

        rng = np.random.default_rng(seed)

        # Create initial firing state based on fraction
        firing = rng.random(n_neurons) < initial_firing_fraction

        # Previous state starts all False
        firing_prev = np.zeros(n_neurons, dtype=np.bool_)

        return cls(
            firing=firing,
            firing_prev=firing_prev,
            threshold=threshold,
        )

    def update_firing(self, input_signal: ndarray) -> None:
        """Update firing state based on input signal.

        Neurons fire if their input signal >= threshold.
        Previous firing state is preserved before update.

        Args:
            input_signal: Shape (N,) - weighted sum of inputs for each neuron
        """
        # Preserve current state as previous
        np.copyto(self.firing_prev, self.firing)

        # Update firing based on threshold comparison
        np.copyto(self.firing, input_signal >= self.threshold)
