"""Network data structure for neural cellular automata.

Represents the structural and synaptic connectivity of the neural network.
"""

from dataclasses import dataclass

import numpy as np
from numpy import ndarray


@dataclass
class Network:
    """Neural network with 3D positions and connectivity.

    Attributes:
        positions: Shape (N, 3) - 3D coordinates of neurons
        link_matrix: Shape (N, N) - directed structural connectivity (bool)
        weight_matrix: Shape (N, N) - synaptic weights (float)
        n_neurons: Number of neurons in the network
        radius: Connectivity radius
        box_size: Dimensions of the 3D volume
    """

    positions: ndarray
    link_matrix: ndarray
    weight_matrix: ndarray
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
    ) -> "Network":
        """Create a network with random neuron positions.

        Args:
            n_neurons: Number of neurons to create
            box_size: (x, y, z) dimensions of the 3D volume
            radius: Maximum distance for structural connectivity
            initial_weight: Initial synaptic weight for connected neurons
            seed: Random seed for reproducibility

        Returns:
            Network with randomly positioned neurons and distance-based connectivity
        """
        rng = np.random.default_rng(seed)

        # Generate random 3D positions within box bounds
        positions = np.zeros((n_neurons, 3), dtype=np.float64)
        positions[:, 0] = rng.uniform(0, box_size[0], n_neurons)
        positions[:, 1] = rng.uniform(0, box_size[1], n_neurons)
        positions[:, 2] = rng.uniform(0, box_size[2], n_neurons)

        # Compute pairwise distances
        # Using broadcasting: diff[i,j,k] = positions[i,k] - positions[j,k]
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))

        # Link matrix: True if distance <= radius and not self-connection
        link_matrix = (distances <= radius) & (distances > 0)

        # Weight matrix: initial_weight where links exist, 0 otherwise
        weight_matrix = np.where(link_matrix, initial_weight, 0.0)

        return cls(
            positions=positions,
            link_matrix=link_matrix,
            weight_matrix=weight_matrix,
            n_neurons=n_neurons,
            radius=radius,
            box_size=box_size,
        )
