"""Network data structure for neural cellular automata.

Represents the structural and synaptic connectivity of the neural network.
"""

from dataclasses import dataclass

import numpy as np
from numpy import ndarray

from src.core.backend import ArrayBackend, get_backend


@dataclass
class Network:
    """Neural network with 3D positions and connectivity.

    Attributes:
        positions: Shape (N, 3) - 3D coordinates of neurons
        link_matrix: Shape (N, N) - directed structural connectivity (bool)
        weight_matrix: Shape (N, N) - synaptic weights (float)
        neuron_types: Shape (N,) - neuron type (True=excitatory, False=inhibitory)
        n_neurons: Number of neurons in the network
        radius: Connectivity radius
        box_size: Dimensions of the 3D volume
    """

    positions: ndarray
    link_matrix: ndarray
    weight_matrix: ndarray
    neuron_types: ndarray  # True = excitatory, False = inhibitory
    n_neurons: int
    radius: float
    box_size: tuple[float, float, float]

    @classmethod
    def create_random(
        cls,
        n_neurons: int,
        box_size: tuple[float, float, float],
        radius: float,
        initial_weight: float,
        seed: int | None = None,
        backend: ArrayBackend | None = None,
        excitatory_fraction: float = 0.8,
    ) -> "Network":
        """Create a network with random neuron positions.

        Args:
            n_neurons: Number of neurons to create
            box_size: (x, y, z) dimensions of the 3D volume
            radius: Maximum distance for structural connectivity
            initial_weight: Initial synaptic weight for connected neurons
            seed: Random seed for reproducibility
            backend: Array computation backend (defaults to NumPyBackend)
            excitatory_fraction: Fraction of neurons that are excitatory (default 0.8)

        Returns:
            Network with randomly positioned neurons and distance-based connectivity
        """
        if not 0.0 <= excitatory_fraction <= 1.0:
            raise ValueError("excitatory_fraction must be in [0.0, 1.0]")

        if backend is None:
            backend = get_backend()

        # Generate random 3D positions within box bounds
        # Derive different seeds for each dimension to maintain independence
        if seed is not None:
            seed_x, seed_y, seed_z = seed, seed + 1, seed + 2
        else:
            seed_x = seed_y = seed_z = None

        x_pos = backend.random_uniform(0, box_size[0], (n_neurons,), seed_x)
        y_pos = backend.random_uniform(0, box_size[1], (n_neurons,), seed_y)
        z_pos = backend.random_uniform(0, box_size[2], (n_neurons,), seed_z)

        # Stack into (N, 3) array - use numpy stack since result needs to be numpy for dataclass
        positions = np.column_stack([backend.to_numpy(x_pos), backend.to_numpy(y_pos), backend.to_numpy(z_pos)])

        # Distance computation - keep as numpy (initialization code, not hot path)
        # This is acceptable since Network creation is one-time cost
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))

        link_matrix = (distances <= radius) & (distances > 0)

        # Assign neuron types: True = excitatory, False = inhibitory
        rng = np.random.default_rng(seed)
        neuron_types = rng.random(n_neurons) < excitatory_fraction

        # Weight matrix: sign depends on presynaptic neuron type
        weight_signs = np.where(neuron_types, 1.0, -1.0)  # (N,)
        weight_matrix = np.where(link_matrix, initial_weight * weight_signs[:, np.newaxis], 0.0)

        return cls(
            positions=positions,
            link_matrix=link_matrix,
            weight_matrix=weight_matrix,
            neuron_types=neuron_types,
            n_neurons=n_neurons,
            radius=radius,
            box_size=box_size,
        )
