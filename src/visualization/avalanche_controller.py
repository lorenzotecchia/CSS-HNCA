"""Avalanche control logic for the pygame visualization."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.core.simulation import Simulation
from src.events.avalanche import AvalancheDetector


@dataclass
class AvalancheController:
    simulation: Simulation
    n_neurons: int
    stimulus_count: int = 1
    pause_timesteps: int = 1

    target: int | None = None
    seen: int = 0

    detector: AvalancheDetector = field(init=False)
    _last_count: int = field(default=0, init=False, repr=False)
    _pause_remaining: int = field(default=0, init=False, repr=False)
    _pending_stimulus: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.quiet_threshold = 1.0 / max(1, self.n_neurons)
        self.reset_detector()

    def reset_detector(self, reset_seen: bool = False) -> None:
        self.detector = AvalancheDetector(
            n_neurons=self.n_neurons, quiet_threshold=self.quiet_threshold
        )
        self._last_count = 0
        self._pause_remaining = 0.0
        self._pending_stimulus = False
        if reset_seen:
            self.seen = 0

    def rebind(self, simulation: Simulation, n_neurons: int, stimulus_count: int | None = None, reset_seen: bool = False) -> None:
        self.simulation = simulation
        self.n_neurons = n_neurons
        if stimulus_count is not None:
            self.stimulus_count = stimulus_count
        self.quiet_threshold = 1.0 / max(1, self.n_neurons)
        self.reset_detector(reset_seen=reset_seen)

    def apply_stimulus(self) -> None:
        if self.n_neurons <= 0:
            return
        count = min(self.stimulus_count, self.n_neurons)
        indices = np.random.choice(self.n_neurons, size=count, replace=False)
        self.simulation.state.firing[indices] = True
        mp = self.simulation.state.membrane_potential
        mp[indices] = np.maximum(mp[indices], self.simulation.state.threshold)

    def set_target(self, target: int) -> None:
        if target <= 0:
            raise ValueError("Avalanche count must be > 0.")
        self.target = target
        self.seen = 0
        self.reset_detector()
        self.apply_stimulus()

    def record_step(self, time_step: int, firing_count: int) -> bool:
        """Return whether the run should keep running."""
        self.detector.record_step(time_step, firing_count)
        if self.target is None:
            self._last_count = len(self.detector.avalanches)
            return True
        current_count = len(self.detector.avalanches)
        if current_count <= self._last_count:
            # No new avalanche detected yet
            pass
        else:
            self.seen += current_count - self._last_count
            self._last_count = current_count
            if self.seen >= self.target:
                return False  # Stop the run if target reached

        # New logic: Trigger stimulus pause when firing count drops to 0 (avalanche ends)
        if firing_count == 0 and self.target is not None and self.seen < self.target and not self._pending_stimulus:
            self._pause_remaining = self.pause_timesteps  # Set to 1 timestep
            self._pending_stimulus = True

        return True

    def update(self, dt: float) -> None:
        if not self._pending_stimulus:
            return
        if self._pause_remaining > 0:
            self._pause_remaining -= 1
            if self._pause_remaining > 0:
                return
        self._pending_stimulus = False
        self.apply_stimulus()

    def status_line(self) -> str:
        if self.target is None:
            return "avalanches = off"
        return f"avalanches = {self.seen}/{self.target}"
