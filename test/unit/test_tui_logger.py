"""Unit tests for TUI Logger.

Tests output format and step event handling.
"""

import io
import sys
from unittest.mock import patch

import pytest


class TestTUILoggerCreation:
    """Test TUILogger initialization."""

    def test_creates_with_default_stream(self) -> None:
        """TUILogger should default to sys.stdout."""
        from src.visualization.tui_logger import TUILogger

        logger = TUILogger()
        assert logger.stream == sys.stdout

    def test_creates_with_custom_stream(self) -> None:
        """TUILogger should accept custom output stream."""
        from src.visualization.tui_logger import TUILogger

        buffer = io.StringIO()
        logger = TUILogger(stream=buffer)
        assert logger.stream == buffer


class TestTUILoggerOutput:
    """Test TUILogger output formatting."""

    def test_log_step_formats_correctly(self) -> None:
        """log_step should format as [t=XXXXX] firing: N | avg_weight: X.XXXX."""
        from src.visualization.tui_logger import TUILogger

        buffer = io.StringIO()
        logger = TUILogger(stream=buffer)

        logger.log_step(time_step=42, firing_count=23, avg_weight=0.1523)

        output = buffer.getvalue()
        assert output == "[t=00042] firing: 23 | avg_weight: 0.1523\n"

    def test_log_step_pads_time_step_to_5_digits(self) -> None:
        """Time step should be zero-padded to 5 digits."""
        from src.visualization.tui_logger import TUILogger

        buffer = io.StringIO()
        logger = TUILogger(stream=buffer)

        logger.log_step(time_step=1, firing_count=100, avg_weight=0.5)

        output = buffer.getvalue()
        assert "[t=00001]" in output

    def test_log_step_handles_large_time_steps(self) -> None:
        """Large time steps should not truncate."""
        from src.visualization.tui_logger import TUILogger

        buffer = io.StringIO()
        logger = TUILogger(stream=buffer)

        logger.log_step(time_step=123456, firing_count=50, avg_weight=0.25)

        output = buffer.getvalue()
        assert "[t=123456]" in output

    def test_log_step_formats_weight_to_4_decimals(self) -> None:
        """Average weight should have 4 decimal places."""
        from src.visualization.tui_logger import TUILogger

        buffer = io.StringIO()
        logger = TUILogger(stream=buffer)

        logger.log_step(time_step=1, firing_count=10, avg_weight=0.123456789)

        output = buffer.getvalue()
        assert "avg_weight: 0.1235" in output  # Rounded

    def test_log_step_handles_zero_values(self) -> None:
        """Should handle zero firing count and weight."""
        from src.visualization.tui_logger import TUILogger

        buffer = io.StringIO()
        logger = TUILogger(stream=buffer)

        logger.log_step(time_step=0, firing_count=0, avg_weight=0.0)

        output = buffer.getvalue()
        assert "[t=00000] firing: 0 | avg_weight: 0.0000\n" == output


class TestTUILoggerMultipleSteps:
    """Test TUILogger with multiple log calls."""

    def test_logs_multiple_steps_sequentially(self) -> None:
        """Multiple log_step calls should append lines."""
        from src.visualization.tui_logger import TUILogger

        buffer = io.StringIO()
        logger = TUILogger(stream=buffer)

        logger.log_step(1, 10, 0.1)
        logger.log_step(2, 20, 0.2)
        logger.log_step(3, 30, 0.3)

        lines = buffer.getvalue().strip().split("\n")
        assert len(lines) == 3
        assert "[t=00001]" in lines[0]
        assert "[t=00002]" in lines[1]
        assert "[t=00003]" in lines[2]


class TestTUILoggerVerbosity:
    """Test TUILogger verbosity control."""

    def test_silent_mode_produces_no_output(self) -> None:
        """When verbose=False, log_step should not output."""
        from src.visualization.tui_logger import TUILogger

        buffer = io.StringIO()
        logger = TUILogger(stream=buffer, verbose=False)

        logger.log_step(1, 10, 0.1)

        assert buffer.getvalue() == ""

    def test_verbose_mode_produces_output(self) -> None:
        """When verbose=True (default), log_step should output."""
        from src.visualization.tui_logger import TUILogger

        buffer = io.StringIO()
        logger = TUILogger(stream=buffer, verbose=True)

        logger.log_step(1, 10, 0.1)

        assert buffer.getvalue() != ""
