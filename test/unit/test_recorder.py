"""Unit tests for Output Recorder.

Tests CSV format and NPZ snapshot structure.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest


class TestRecorderCreation:
    """Test Recorder initialization."""

    def test_creates_with_output_directory(self) -> None:
        """Recorder should accept output directory path."""
        from src.output.recorder import Recorder

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder(output_dir=Path(tmpdir))
            assert recorder.output_dir == Path(tmpdir)

    def test_creates_output_directory_if_not_exists(self) -> None:
        """Recorder should create output directory if it doesn't exist."""
        from src.output.recorder import Recorder

        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "new_output"
            recorder = Recorder(output_dir=new_dir)
            assert new_dir.exists()

    def test_default_snapshot_interval(self) -> None:
        """Recorder should have default snapshot interval of 1000."""
        from src.output.recorder import Recorder

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder(output_dir=Path(tmpdir))
            assert recorder.snapshot_interval == 1000

    def test_custom_snapshot_interval(self) -> None:
        """Recorder should accept custom snapshot interval."""
        from src.output.recorder import Recorder

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder(output_dir=Path(tmpdir), snapshot_interval=500)
            assert recorder.snapshot_interval == 500


class TestRecorderCSV:
    """Test CSV recording functionality."""

    def test_record_step_creates_csv_file(self) -> None:
        """First record_step should create timeseries.csv."""
        from src.output.recorder import Recorder

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder(output_dir=Path(tmpdir))
            recorder.record_step(time_step=1, firing_count=10, avg_weight=0.1)

            csv_path = Path(tmpdir) / "timeseries.csv"
            assert csv_path.exists()

    def test_csv_has_header(self) -> None:
        """CSV should have header: time_step,firing_count,avg_weight."""
        from src.output.recorder import Recorder

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder(output_dir=Path(tmpdir))
            recorder.record_step(time_step=1, firing_count=10, avg_weight=0.1)

            csv_path = Path(tmpdir) / "timeseries.csv"
            with open(csv_path) as f:
                header = f.readline().strip()
            assert header == "time_step,firing_count,avg_weight"

    def test_csv_records_data_correctly(self) -> None:
        """CSV should record step data in correct format."""
        from src.output.recorder import Recorder

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder(output_dir=Path(tmpdir))
            recorder.record_step(time_step=42, firing_count=23, avg_weight=0.1523)

            csv_path = Path(tmpdir) / "timeseries.csv"
            with open(csv_path) as f:
                lines = f.readlines()
            
            assert len(lines) == 2  # Header + 1 data row
            data_line = lines[1].strip()
            assert data_line == "42,23,0.1523"

    def test_csv_appends_multiple_steps(self) -> None:
        """Multiple record_step calls should append to CSV."""
        from src.output.recorder import Recorder

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder(output_dir=Path(tmpdir))
            recorder.record_step(1, 10, 0.1)
            recorder.record_step(2, 20, 0.2)
            recorder.record_step(3, 30, 0.3)

            csv_path = Path(tmpdir) / "timeseries.csv"
            with open(csv_path) as f:
                lines = f.readlines()
            
            assert len(lines) == 4  # Header + 3 data rows


class TestRecorderNPZ:
    """Test NPZ snapshot functionality."""

    def test_save_snapshot_creates_npz_file(self) -> None:
        """save_snapshot should create NPZ file with correct name."""
        from src.output.recorder import Recorder

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder(output_dir=Path(tmpdir))
            weights = np.array([[0.1, 0.2], [0.3, 0.4]])
            
            recorder.save_snapshot(time_step=1000, weight_matrix=weights)

            npz_path = Path(tmpdir) / "snapshot_001000.npz"
            assert npz_path.exists()

    def test_snapshot_contains_weight_matrix(self) -> None:
        """NPZ snapshot should contain weight_matrix array."""
        from src.output.recorder import Recorder

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder(output_dir=Path(tmpdir))
            weights = np.array([[0.1, 0.2], [0.3, 0.4]])
            
            recorder.save_snapshot(time_step=1000, weight_matrix=weights)

            npz_path = Path(tmpdir) / "snapshot_001000.npz"
            data = np.load(npz_path)
            assert "weight_matrix" in data.files
            np.testing.assert_array_equal(data["weight_matrix"], weights)

    def test_snapshot_contains_time_step(self) -> None:
        """NPZ snapshot should contain time_step."""
        from src.output.recorder import Recorder

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder(output_dir=Path(tmpdir))
            weights = np.array([[0.1, 0.2], [0.3, 0.4]])
            
            recorder.save_snapshot(time_step=1000, weight_matrix=weights)

            npz_path = Path(tmpdir) / "snapshot_001000.npz"
            data = np.load(npz_path)
            assert "time_step" in data.files
            assert data["time_step"] == 1000

    def test_snapshot_filename_zero_padded(self) -> None:
        """Snapshot filename should be zero-padded to 6 digits."""
        from src.output.recorder import Recorder

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder(output_dir=Path(tmpdir))
            weights = np.array([[0.1, 0.2], [0.3, 0.4]])
            
            recorder.save_snapshot(time_step=42, weight_matrix=weights)

            npz_path = Path(tmpdir) / "snapshot_000042.npz"
            assert npz_path.exists()


class TestRecorderAutoSnapshot:
    """Test automatic snapshot saving at intervals."""

    def test_record_step_saves_snapshot_at_interval(self) -> None:
        """record_step should save snapshot when time_step % interval == 0."""
        from src.output.recorder import Recorder

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder(output_dir=Path(tmpdir), snapshot_interval=100)
            weights = np.array([[0.1, 0.2], [0.3, 0.4]])
            
            # Record steps up to 100
            for i in range(1, 101):
                recorder.record_step(
                    time_step=i, 
                    firing_count=10, 
                    avg_weight=0.1,
                    weight_matrix=weights if i == 100 else None
                )

            npz_path = Path(tmpdir) / "snapshot_000100.npz"
            assert npz_path.exists()

    def test_record_step_no_snapshot_before_interval(self) -> None:
        """record_step should not save snapshot before interval."""
        from src.output.recorder import Recorder

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder(output_dir=Path(tmpdir), snapshot_interval=100)
            
            recorder.record_step(time_step=50, firing_count=10, avg_weight=0.1)

            npz_files = list(Path(tmpdir).glob("snapshot_*.npz"))
            assert len(npz_files) == 0


class TestRecorderClose:
    """Test Recorder cleanup."""

    def test_close_flushes_csv(self) -> None:
        """close() should flush and close CSV file."""
        from src.output.recorder import Recorder

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = Recorder(output_dir=Path(tmpdir))
            recorder.record_step(1, 10, 0.1)
            recorder.close()

            # File should be readable after close
            csv_path = Path(tmpdir) / "timeseries.csv"
            with open(csv_path) as f:
                content = f.read()
            assert "1,10,0.1" in content
