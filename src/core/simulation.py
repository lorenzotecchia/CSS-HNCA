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
    from src.core.jax_kernels import JAXSimulationState
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
        use_gpu: Whether to use JAX GPU acceleration
    """

    network: Network
    state: NeuronState
    learning_rate: float
    forgetting_rate: float
    learner: WeightUpdater | None = None
    event_bus: EventBus | None = None
    backend: ArrayBackend = field(default_factory=get_backend)
    use_gpu: bool = False
    time_step: int = field(default=0, init=False)
    sim_state: SimulationState = field(default=SimulationState.STOPPED, init=False)

    # Store initial config for reset
    _initial_firing_count: int = field(default=0, init=False, repr=False)
    _jax_state: JAXSimulationState | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate that network and state dimensions match."""
        if self.network.n_neurons != self.state.firing.shape[0]:
            raise ValueError(
                f"Network has {self.network.n_neurons} neurons but state has "
                f"{self.state.firing.shape[0]} neurons"
            )
        # Store initial config for potential reset
        self._initial_firing_count = int(np.sum(self.state.firing))
        
        # Initialize JAX state if GPU acceleration is enabled
        if self.use_gpu:
            self._init_jax_state()

    def _init_jax_state(self) -> None:
        """Initialize JAX GPU state from current NumPy state."""
        from src.core.jax_kernels import JAXSimulationState, is_jax_available
        
        if not is_jax_available():
            import warnings
            warnings.warn(
                "JAX not available, falling back to NumPy. "
                "Install with: pip install css-hnca[gpu]"
            )
            self.use_gpu = False
            return
        
        # Get weight bounds from learner if available
        if self.learner is not None:
            weight_min = self.learner.weight_min
            weight_max = self.learner.weight_max
            weight_min_inh = self.learner.weight_min_inh
            weight_max_inh = self.learner.weight_max_inh
            decay_alpha = self.learner.decay_alpha
        else:
            weight_min, weight_max = 0.0, 1.0
            weight_min_inh, weight_max_inh = -1.0, 0.0
            decay_alpha = 0.0
        
        self._jax_state = JAXSimulationState(
            weight_matrix=self.network.weight_matrix,
            link_matrix=self.network.link_matrix,
            firing=self.state.firing.astype(np.float32),
            membrane_potential=self.state.membrane_potential,
            inhibitory_nodes=self.network.inhibitory_nodes,
            threshold=self.state.threshold,
            leak_rate=self.state.leak_rate,
            reset_potential=self.state.reset_potential,
            learning_rate=self.learning_rate,
            forgetting_rate=self.forgetting_rate,
            decay_alpha=decay_alpha,
            weight_min=weight_min,
            weight_max=weight_max,
            weight_min_inh=weight_min_inh,
            weight_max_inh=weight_max_inh,
        )

    def step(self) -> None:
        """Advance simulation by one time step.

        Computes: v(t) = W^T · s(t-1)
        Updates firing state based on threshold.
        Applies Hebbian learning if learner is configured.
        """
        if self.use_gpu and self._jax_state is not None:
            self._step_gpu()
        else:
            self._step_cpu()

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

    def _step_gpu(self) -> None:
        """Execute step on GPU using JAX."""
        self._jax_state.step()

    def _step_cpu(self) -> None:
        """Execute step on CPU using NumPy."""
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

    def sync_from_gpu(self) -> None:
        """Sync state from GPU back to CPU arrays.
        
        Call this after GPU simulation to update NumPy arrays
        for visualization or analysis.
        """
        if self._jax_state is not None:
            weights, firing, potential = self._jax_state.sync_to_numpy()
            self.network.weight_matrix[:] = weights
            self.state.firing[:] = firing
            self.state.membrane_potential[:] = potential

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

        # Reinitialize JAX state if using GPU
        if self.use_gpu:
            self._init_jax_state()

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
        if self.use_gpu and self._jax_state is not None:
            return self._jax_state.get_firing_count()
        return int(np.sum(self.state.firing))

    @property
    def average_weight(self) -> float:
        """Return average weight of connected neurons only."""
        if self.use_gpu and self._jax_state is not None:
            return self._jax_state.get_average_weight()
        connected_weights = self.network.weight_matrix[self.network.link_matrix]
        if len(connected_weights) == 0:
            return 0.0
        return float(np.mean(connected_weights))
