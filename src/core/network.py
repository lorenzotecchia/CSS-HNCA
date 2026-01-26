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
        inhibitory_nodes: Shape (N,) - boolean array indicating inhibitory neurons
    """

    positions: ndarray
    link_matrix: ndarray
    weight_matrix: ndarray
    n_neurons: int
    radius: float
    box_size: tuple[float, float, float]
    inhibitory_nodes: ndarray

    @classmethod
    def create_beta_weighted_directed(
        cls,
        n_neurons: int,
        k_prop: float,
        a: float = 2.0,
        b: float = 6.0,
        inhibitory_proportion: float = 0.0,
        seed: int | None = None,
    ) -> "Network":
        """Create a directed network with beta-distributed weights.

        Initializes with a directed cycle, then adds edges based on 3D proximity,
        with weights sampled from beta(a, b) distribution.

        Args:
            n_neurons: Number of neurons (N >= 3)
            k_prop: Average degree proportion (2/N <= k_prop <= 1-1/N)
            a: Beta distribution parameter (a > 0)
            b: Beta distribution parameter (b > 0)
            inhibitory_proportion: Proportion of neurons that are inhibitory (0.0 to 1.0)
            seed: Random seed for reproducibility

        Returns:
            Network instance with directed connectivity and beta weights
        """
        # Input validation
        if n_neurons < 3:
            raise ValueError("n_neurons must be >= 3")
        if not (2 / n_neurons <= k_prop <= 1 - 1 / n_neurons):
            raise ValueError(f"k_prop must be in [{2/n_neurons:.3f}, {1-1/n_neurons:.3f}]")
        if a <= 0 or b <= 0:
            raise ValueError("a and b must be > 0")
        if not (0.0 <= inhibitory_proportion <= 1.0):
            raise ValueError("inhibitory_proportion must be in [0.0, 1.0]")

        rng = np.random.default_rng(seed)

        # 1. Initialize directed cycle
        link_matrix = np.zeros((n_neurons, n_neurons), dtype=bool)
        for i in range(n_neurons):
            link_matrix[i, (i + 1) % n_neurons] = True

        # 2. Random 3D positions in [0,1]x[0,1]x[0,1]
        positions = rng.uniform(0, 1, (n_neurons, 3))

        # 3. Find possible additional edges within increasing radius
        target_additional_edges = round(k_prop * n_neurons**2) - n_neurons
        possible_edges = []
        r = 0.01
        max_r = np.sqrt(3)

        while len(possible_edges) < target_additional_edges and r <= max_r:
            possible_edges = []
            for i in range(n_neurons):
                for j in range(n_neurons):
                    if i == j or link_matrix[i, j]:
                        continue
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist <= r:
                        possible_edges.append((i, j))
            r += 0.01

        # 4. Sample edges to add
        if len(possible_edges) >= target_additional_edges:
            sampled_indices = rng.choice(
                len(possible_edges), size=target_additional_edges, replace=False
            )
            for idx in sampled_indices:
                i, j = possible_edges[idx]
                link_matrix[i, j] = True
        else:
            # Add all possible edges if not enough
            for i, j in possible_edges:
                link_matrix[i, j] = True

        # 5. Assign beta-distributed weights to all edges
        weight_matrix = np.zeros((n_neurons, n_neurons))
        for i in range(n_neurons):
            for j in range(n_neurons):
                if link_matrix[i, j]:
                    weight_matrix[i, j] = rng.beta(a, b)

        # Select inhibitory nodes
        inhibitory_nodes = rng.random(n_neurons) < inhibitory_proportion

        # Assign negative weights to inhibitory out-degrees
        for i in range(n_neurons):
            if inhibitory_nodes[i]:
                weight_matrix[i, :] = -np.abs(weight_matrix[i, :])

        # Clamp weights to bounds
        for i in range(n_neurons):
            for j in range(n_neurons):
                if link_matrix[i, j]:
                    if inhibitory_nodes[i]:
                        weight_matrix[i, j] = np.clip(weight_matrix[i, j], -1.0, -0.0001)
                    else:
                        weight_matrix[i, j] = np.clip(weight_matrix[i, j], 0.0001, 1.0)

        return cls(
            positions=positions,
            link_matrix=link_matrix,
            weight_matrix=weight_matrix,
            n_neurons=n_neurons,
            radius=r,
            box_size=(1.0, 1.0, 1.0),
            inhibitory_nodes=inhibitory_nodes,
        )
