"""Integration tests for headless simulation run.

Tests full simulation with TUI logger and recorder.
"""

import io
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.core.network import Network
from src.core.neuron_state import NeuronState
from src.core.simulation import Simulation
from src.learning.hebbian import HebbianLearner
from src.output.recorder import Recorder
from src.visualization.tui_logger import TUILogger


class TestHeadlessSimulationWithTUI:
    """Test headless simulation with TUI logging."""

    def test_tui_logs_all_steps(self) -> None:
        """TUI should log each simulation step."""
        network = Network.create_random(
            n_neurons=10,
            box_size=(5.0, 5.0, 5.0),
            radius=2.0,
            initial_weight=0.1,
            seed=42,
        )
        state = NeuronState.create(
            n_neurons=10,
            threshold=0.5,
            initial_firing_fraction=0.2,
            seed=42,
        )
        sim = Simulation(
            network=network,
            state=state,
            learning_rate=0.01,
            forgetting_rate=0.005,
        )

        buffer = io.StringIO()
        logger = TUILogger(stream=buffer)

        # Run 10 steps
        for _ in range(10):
            sim.step()
            logger.log_step(
                time_step=sim.time_step,
                firing_count=sim.firing_count,
                avg_weight=sim.average_weight,
            )

        lines = buffer.getvalue().strip().split("\n")
        assert len(lines) == 10
        assert "[t=00001]" in lines[0]
        assert "[t=00010]" in lines[9]


class TestHeadlessSimulationWithRecorder:
    """Test headless simulation with recorder."""

    def test_recorder_captures_all_steps(self) -> None:
        """Recorder should capture metrics for all steps."""
        network = Network.create_random(
            n_neurons=10,
            box_size=(5.0, 5.0, 5.0),
            radius=2.0,
            initial_weight=0.1,
            seed=42,
        )
        state = NeuronState.create(
            n_neurons=10,
            threshold=0.5,
            initial_firing_fraction=0.2,
            seed=42,
        )
        sim = Simulation(
            network=network,
            state=state,
            learning_rate=0.01,
            forgetting_rate=0.005,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder(output_dir=Path(tmpdir))

            # Run 20 steps
            for _ in range(20):
                sim.step()
                recorder.record_step(
                    time_step=sim.time_step,
                    firing_count=sim.firing_count,
                    avg_weight=sim.average_weight,
                )

            recorder.close()

            # Verify CSV
            csv_path = Path(tmpdir) / "timeseries.csv"
            with open(csv_path) as f:
                lines = f.readlines()

            assert len(lines) == 21  # Header + 20 data rows

    def test_recorder_saves_snapshots_at_interval(self) -> None:
        """Recorder should save NPZ snapshots at configured interval."""
        network = Network.create_random(
            n_neurons=10,
            box_size=(5.0, 5.0, 5.0),
            radius=2.0,
            initial_weight=0.1,
            seed=42,
        )
        state = NeuronState.create(
            n_neurons=10,
            threshold=0.5,
            initial_firing_fraction=0.2,
            seed=42,
        )
        learner = HebbianLearner(
            learning_rate=0.01,
            forgetting_rate=0.005,
            weight_min=0.0,
            weight_max=1.0,
        )
        sim = Simulation(
            network=network,
            state=state,
            learning_rate=0.01,
            forgetting_rate=0.005,
            learner=learner,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder(output_dir=Path(tmpdir), snapshot_interval=50)

            # Run 100 steps
            for _ in range(100):
                sim.step()
                recorder.record_step(
                    time_step=sim.time_step,
                    firing_count=sim.firing_count,
                    avg_weight=sim.average_weight,
                    weight_matrix=sim.network.weight_matrix,
                )

            recorder.close()

            # Should have snapshots at 50 and 100
            npz_50 = Path(tmpdir) / "snapshot_000050.npz"
            npz_100 = Path(tmpdir) / "snapshot_000100.npz"
            assert npz_50.exists()
            assert npz_100.exists()

            # Verify snapshot content
            data = np.load(npz_100)
            assert data["time_step"] == 100
            assert data["weight_matrix"].shape == (10, 10)


class TestHeadlessSimulationWithBoth:
    """Test headless simulation with both TUI and recorder."""

    def test_combined_tui_and_recorder(self) -> None:
        """Simulation should work with both TUI and recorder simultaneously."""
        network = Network.create_random(
            n_neurons=10,
            box_size=(5.0, 5.0, 5.0),
            radius=2.0,
            initial_weight=0.1,
            seed=42,
        )
        state = NeuronState.create(
            n_neurons=10,
            threshold=0.5,
            initial_firing_fraction=0.2,
            seed=42,
        )
        sim = Simulation(
            network=network,
            state=state,
            learning_rate=0.01,
            forgetting_rate=0.005,
        )

        buffer = io.StringIO()
        logger = TUILogger(stream=buffer)

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder(output_dir=Path(tmpdir))

            for _ in range(5):
                sim.step()
                logger.log_step(
                    time_step=sim.time_step,
                    firing_count=sim.firing_count,
                    avg_weight=sim.average_weight,
                )
                recorder.record_step(
                    time_step=sim.time_step,
                    firing_count=sim.firing_count,
                    avg_weight=sim.average_weight,
                )

            recorder.close()

            # Both should have 5 entries
            log_lines = buffer.getvalue().strip().split("\n")
            assert len(log_lines) == 5

            csv_path = Path(tmpdir) / "timeseries.csv"
            with open(csv_path) as f:
                csv_lines = f.readlines()
            assert len(csv_lines) == 6  # Header + 5 data rows


class TestRecorderDataIntegrity:
    """Test that recorded data matches simulation state."""

    def test_recorded_firing_count_matches_simulation(self) -> None:
        """Recorded firing count should match actual simulation state."""
        network = Network.create_random(
            n_neurons=20,
            box_size=(5.0, 5.0, 5.0),
            radius=2.0,
            initial_weight=0.1,
            seed=123,
        )
        state = NeuronState.create(
            n_neurons=20,
            threshold=0.5,
            initial_firing_fraction=0.3,
            seed=123,
        )
        sim = Simulation(
            network=network,
            state=state,
            learning_rate=0.01,
            forgetting_rate=0.005,
        )

        recorded_counts = []

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder(output_dir=Path(tmpdir))

            for _ in range(10):
                sim.step()
                firing_count = sim.firing_count
                recorded_counts.append(firing_count)
                recorder.record_step(
                    time_step=sim.time_step,
                    firing_count=firing_count,
                    avg_weight=sim.average_weight,
                )

            recorder.close()

            # Parse CSV and verify counts match
            csv_path = Path(tmpdir) / "timeseries.csv"
            with open(csv_path) as f:
                lines = f.readlines()[1:]  # Skip header

            csv_counts = [int(line.split(",")[1]) for line in lines]
            assert csv_counts == recorded_counts
