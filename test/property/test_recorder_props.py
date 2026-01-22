"""Property-based tests for Output Recorder.

Tests invariants: recorded steps match simulation steps.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from src.output.recorder import Recorder


class TestRecorderStepCountProperty:
    """Property: recorded step count matches calls."""

    @given(n_steps=st.integers(min_value=1, max_value=100))
    @settings(max_examples=20)
    def test_csv_row_count_matches_steps(self, n_steps: int) -> None:
        """CSV should have exactly n_steps data rows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder(output_dir=Path(tmpdir))

            for i in range(1, n_steps + 1):
                recorder.record_step(
                    time_step=i,
                    firing_count=i * 10,
                    avg_weight=i * 0.01,
                )

            recorder.close()

            csv_path = Path(tmpdir) / "timeseries.csv"
            with open(csv_path) as f:
                lines = f.readlines()

            # Header + n_steps data rows
            assert len(lines) == n_steps + 1


class TestRecorderTimeStepOrdering:
    """Property: time steps are recorded in order."""

    @given(n_steps=st.integers(min_value=2, max_value=50))
    @settings(max_examples=20)
    def test_time_steps_are_sequential(self, n_steps: int) -> None:
        """Time steps in CSV should be sequential."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder(output_dir=Path(tmpdir))

            for i in range(1, n_steps + 1):
                recorder.record_step(time_step=i, firing_count=10, avg_weight=0.1)

            recorder.close()

            csv_path = Path(tmpdir) / "timeseries.csv"
            with open(csv_path) as f:
                lines = f.readlines()[1:]  # Skip header

            time_steps = [int(line.split(",")[0]) for line in lines]
            expected = list(range(1, n_steps + 1))
            assert time_steps == expected


class TestRecorderSnapshotProperty:
    """Property: snapshots saved at correct intervals."""

    @given(
        n_steps=st.integers(min_value=10, max_value=200),
        interval=st.integers(min_value=5, max_value=50),
    )
    @settings(max_examples=20)
    def test_snapshot_count_matches_interval(
        self, n_steps: int, interval: int
    ) -> None:
        """Number of snapshots should equal n_steps // interval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder(output_dir=Path(tmpdir), snapshot_interval=interval)
            weights = np.random.rand(10, 10)

            for i in range(1, n_steps + 1):
                recorder.record_step(
                    time_step=i,
                    firing_count=10,
                    avg_weight=0.1,
                    weight_matrix=weights,
                )

            recorder.close()

            npz_files = list(Path(tmpdir).glob("snapshot_*.npz"))
            expected_count = n_steps // interval
            assert len(npz_files) == expected_count


class TestRecorderDataPreservation:
    """Property: data is preserved correctly."""

    @given(
        firing_count=st.integers(min_value=0, max_value=1000),
        avg_weight=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=30)
    def test_recorded_values_match_input(
        self, firing_count: int, avg_weight: float
    ) -> None:
        """Recorded values should match input values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder(output_dir=Path(tmpdir))
            recorder.record_step(
                time_step=1,
                firing_count=firing_count,
                avg_weight=avg_weight,
            )
            recorder.close()

            csv_path = Path(tmpdir) / "timeseries.csv"
            with open(csv_path) as f:
                lines = f.readlines()

            data_line = lines[1].strip()
            parts = data_line.split(",")

            assert int(parts[1]) == firing_count
            assert float(parts[2]) == pytest.approx(avg_weight, rel=1e-10)


class TestRecorderWeightMatrixPreservation:
    """Property: weight matrices are preserved in snapshots."""

    @given(
        n_neurons=st.integers(min_value=2, max_value=20),
    )
    @settings(max_examples=15)
    def test_snapshot_preserves_weight_matrix(self, n_neurons: int) -> None:
        """Snapshot should preserve weight matrix exactly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder(output_dir=Path(tmpdir))
            weights = np.random.rand(n_neurons, n_neurons)

            recorder.save_snapshot(time_step=1, weight_matrix=weights)

            npz_path = Path(tmpdir) / "snapshot_000001.npz"
            data = np.load(npz_path)

            np.testing.assert_array_equal(data["weight_matrix"], weights)
