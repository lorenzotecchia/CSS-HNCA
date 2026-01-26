"""Simulation orchestrator for neural cellular automata.

Orchestrates the network evolution with state updates and learning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

from src.core.backend import ArrayBackend, get_backend
from src.core.network import Network
from src.core.neuron_state import NeuronState

if TYPE_CHECKING:
    from src.events.bus import EventBus
    from src.learning.weight_update import WeightUpdater


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
    learner: WeightUpdater | None = None
    event_bus: EventBus | None = None
    backend: ArrayBackend = field(default_factory=get_backend)
    time_step: int = field(default=0, init=False)
    sim_state: SimulationState = field(default=SimulationState.STOPPED, init=False)

    # Store initial config for reset
    _initial_firing_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate that network and state dimensions match."""
        if self.network.n_neurons != self.state.firing.shape[0]:
            raise ValueError(
                f"Network has {self.network.n_neurons} neurons but state has "
                f"{self.state.firing.shape[0]} neurons"
            )
        # Store initial config for potential reset
        self._initial_firing_count = int(np.sum(self.state.firing))

    def step(self) -> None:
        """Advance simulation by one time step.

        Computes: v(t) = W^T · s(t-1)
        Updates firing state based on threshold.
        Applies Hebbian learning if learner is configured.
        """
        # Store previous firing state for STDP (before update)
        firing_prev = self.state.firing.copy()

        # Compute input to each neuron: v = W^T · s
        # W[i,j] = weight from i to j
        # v[j] = sum over i of W[i,j] * s[i] = (W^T · s)[j]
        input_signal = self.network.weight_matrix.T @ self.state.firing.astype(float)

        # Update neuron state
        self.state.update_firing(input_signal)

        # Apply Hebbian learning if learner is configured
        if self.learner is not None:
            new_weights = self.learner.apply(
                weights=self.network.weight_matrix,
                link_matrix=self.network.link_matrix,
                firing_prev=firing_prev,
                firing_current=self.state.firing,
                inhibitory_nodes=self.network.inhibitory_nodes,
            )
            self.network.weight_matrix[:] = new_weights

        # Increment time
        self.time_step += 1

        # Emit step event
        if self.event_bus is not None:
            from src.events.bus import StepEvent

            self.event_bus.emit(
                StepEvent(
                    time_step=self.time_step,
                    firing_count=self.firing_count,
                    avg_weight=self.average_weight,
                )
            )

    def reset(self, seed: int | None = None) -> None:
        """Reset simulation to initial state.

        Args:
            seed: If provided, regenerate network and state with this seed
        """
        self.time_step = 0
        self.sim_state = SimulationState.STOPPED

        if seed is not None:
            # Regenerate neuron state (preserving LIF parameters)
            new_state = NeuronState.create(
                n_neurons=self.network.n_neurons,
                threshold=self.state.threshold,
                firing_count=self._initial_firing_count,
                seed=seed,
                leak_rate=self.state.leak_rate,
                reset_potential=self.state.reset_potential,
            )
            self.state.firing[:] = new_state.firing
            self.state.firing_prev[:] = new_state.firing_prev
            self.state.membrane_potential[:] = new_state.membrane_potential

        # Emit reset event
        if self.event_bus is not None:
            from src.events.bus import ResetEvent

            self.event_bus.emit(ResetEvent(seed=seed))

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
        """Return average weight of connected neurons only."""
        connected_weights = self.network.weight_matrix[self.network.link_matrix]
        if len(connected_weights) == 0:
            return 0.0
        return float(np.mean(connected_weights))
