"""Output Recorder for simulation data persistence.

Records time series to CSV and weight matrix snapshots to NPZ.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

import numpy as np
from numpy import ndarray


@dataclass
class Recorder:
    """Records simulation metrics and snapshots.

    Attributes:
        output_dir: Directory for output files
        snapshot_interval: Steps between NPZ snapshots
    """

    output_dir: Path
    snapshot_interval: int = 1000
    _csv_file: TextIO | None = field(default=None, init=False, repr=False)
    _csv_initialized: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        """Create output directory if it doesn't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_csv_initialized(self) -> None:
        """Initialize CSV file with header if not already done."""
        if self._csv_initialized:
            return

        csv_path = self.output_dir / "timeseries.csv"
        self._csv_file = open(csv_path, "w")
        self._csv_file.write("time_step,firing_count,avg_weight\n")
        self._csv_file.flush()
        self._csv_initialized = True

    def record_step(
        self,
        time_step: int,
        firing_count: int,
        avg_weight: float,
        weight_matrix: ndarray | None = None,
    ) -> None:
        """Record metrics for a simulation step.

        Writes to CSV and saves NPZ snapshot at intervals.

        Args:
            time_step: Current simulation time step
            firing_count: Number of neurons currently firing
            avg_weight: Average synaptic weight
            weight_matrix: Weight matrix for snapshot (optional)
        """
        self._ensure_csv_initialized()

        # Write CSV row
        assert self._csv_file is not None
        self._csv_file.write(f"{time_step},{firing_count},{avg_weight}\n")
        self._csv_file.flush()

        # Save snapshot at intervals if weight matrix provided
        if weight_matrix is not None and time_step % self.snapshot_interval == 0:
            self.save_snapshot(time_step, weight_matrix)

    def save_snapshot(self, time_step: int, weight_matrix: ndarray) -> None:
        """Save weight matrix snapshot to NPZ file.

        Args:
            time_step: Current simulation time step
            weight_matrix: Weight matrix to save
        """
        filename = f"snapshot_{time_step:06d}.npz"
        npz_path = self.output_dir / filename
        np.savez(npz_path, weight_matrix=weight_matrix, time_step=time_step)

    def close(self) -> None:
        """Close CSV file and flush buffers."""
        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None
