"""Simulation orchestrator for neural cellular automata.

Orchestrates the network evolution with state updates and learning.
"""

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from src.core.network import Network
from src.core.neuron_state import NeuronState


class SimulationState(Enum):
    """State machine for simulation control."""

    STOPPED = auto()
    RUNNING = auto()
    PAUSED = auto()


@dataclass
class Simulation:
    """Orchestrates network simulation with learning.

    Attributes:
        network: The neural network structure
        state: Current neuron firing state
        learning_rate: l parameter for LTP
        forgetting_rate: f parameter for LTD
        time_step: Current simulation time
        sim_state: Control state (STOPPED, RUNNING, PAUSED)
    """

    network: Network
    state: NeuronState
    learning_rate: float
    forgetting_rate: float
    time_step: int = field(default=0, init=False)
    sim_state: SimulationState = field(default=SimulationState.STOPPED, init=False)

    # Store initial config for reset
    _initial_weight: float = field(default=0.1, init=False, repr=False)
    _initial_firing_fraction: float = field(default=0.1, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate that network and state dimensions match."""
        if self.network.n_neurons != self.state.firing.shape[0]:
            raise ValueError(
                f"Network has {self.network.n_neurons} neurons but state has "
                f"{self.state.firing.shape[0]} neurons"
            )
        # Store initial config for potential reset
        # Infer from current state (approximate)
        if np.any(self.network.link_matrix):
            linked_weights = self.network.weight_matrix[self.network.link_matrix]
            if len(linked_weights) > 0:
                self._initial_weight = float(np.mean(linked_weights))
        self._initial_firing_fraction = float(np.mean(self.state.firing))

    def step(self) -> None:
        """Advance simulation by one time step.

        Computes: v(t) = W^T · s(t-1)
        Updates firing state based on threshold.
        """
        # Compute input to each neuron: v = W^T · s
        # W[i,j] = weight from i to j
        # v[j] = sum over i of W[i,j] * s[i] = (W^T · s)[j]
        input_signal = self.network.weight_matrix.T @ self.state.firing.astype(float)

        # Update neuron state
        self.state.update_firing(input_signal)

        # Increment time
        self.time_step += 1

    def reset(self, seed: int | None = None) -> None:
        """Reset simulation to initial state.

        Args:
            seed: If provided, regenerate network and state with this seed
        """
        self.time_step = 0
        self.sim_state = SimulationState.STOPPED

        if seed is not None:
            # Regenerate network with new seed
            new_network = Network.create_random(
                n_neurons=self.network.n_neurons,
                box_size=self.network.box_size,
                radius=self.network.radius,
                initial_weight=self._initial_weight,
                seed=seed,
            )
            # Update in place
            self.network.positions[:] = new_network.positions
            self.network.link_matrix[:] = new_network.link_matrix
            self.network.weight_matrix[:] = new_network.weight_matrix

            # Regenerate neuron state
            new_state = NeuronState.create(
                n_neurons=self.network.n_neurons,
                threshold=self.state.threshold,
                initial_firing_fraction=self._initial_firing_fraction,
                seed=seed,
            )
            self.state.firing[:] = new_state.firing
            self.state.firing_prev[:] = new_state.firing_prev

    def start(self) -> None:
        """Start or resume simulation."""
        self.sim_state = SimulationState.RUNNING

    def pause(self) -> None:
        """Pause simulation (only if running)."""
        if self.sim_state == SimulationState.RUNNING:
            self.sim_state = SimulationState.PAUSED

    @property
    def firing_count(self) -> int:
        """Return count of currently firing neurons."""
        return int(np.sum(self.state.firing))

    @property
    def average_weight(self) -> float:
        """Return average weight of all connections."""
        return float(np.mean(self.network.weight_matrix))
