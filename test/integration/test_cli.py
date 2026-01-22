"""Integration tests for CLI configuration loading.

RED phase: These tests should fail until CLI is implemented.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from src.config.loader import load_config, SimulationConfig
from src.core.network import Network
from src.core.neuron_state import NeuronState
from src.core.simulation import Simulation


class TestCLIConfigLoading:
    """Tests for CLI loading configuration and creating simulation."""

    @pytest.fixture
    def valid_config_file(self):
        """Create a temporary valid config file."""
        toml_content = """
seed = 42

[network]
n_neurons = 50
box_size = [10.0, 10.0, 10.0]
radius = 2.5
initial_weight = 0.1
weight_min = 0.0
weight_max = 1.0
initial_firing_fraction = 0.1

[learning]
threshold = 0.5
learning_rate = 0.01
forgetting_rate = 0.005
decay_alpha = 0.001

[visualization]
pygame_enabled = false
matplotlib_enabled = false
window_width = 800
window_height = 600
fps = 30
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False
        ) as f:
            f.write(toml_content)
            f.flush()
            yield Path(f.name)

    def test_config_creates_valid_network(self, valid_config_file):
        """Loaded config should create a valid Network."""
        config = load_config(valid_config_file)

        network = Network.create_random(
            n_neurons=config.network.n_neurons,
            box_size=config.network.box_size,
            radius=config.network.radius,
            initial_weight=config.network.initial_weight,
            seed=config.seed,
        )

        assert network.n_neurons == 50
        assert network.box_size == (10.0, 10.0, 10.0)
        assert network.radius == 2.5

    def test_config_creates_valid_neuron_state(self, valid_config_file):
        """Loaded config should create a valid NeuronState."""
        config = load_config(valid_config_file)

        state = NeuronState.create(
            n_neurons=config.network.n_neurons,
            threshold=config.learning.threshold,
            initial_firing_fraction=config.network.initial_firing_fraction,
            seed=config.seed,
        )

        assert state.firing.shape[0] == 50
        assert state.threshold == 0.5

    def test_config_creates_valid_simulation(self, valid_config_file):
        """Loaded config should create a working Simulation."""
        config = load_config(valid_config_file)

        network = Network.create_random(
            n_neurons=config.network.n_neurons,
            box_size=config.network.box_size,
            radius=config.network.radius,
            initial_weight=config.network.initial_weight,
            seed=config.seed,
        )
        state = NeuronState.create(
            n_neurons=config.network.n_neurons,
            threshold=config.learning.threshold,
            initial_firing_fraction=config.network.initial_firing_fraction,
            seed=config.seed,
        )
        simulation = Simulation(
            network=network,
            state=state,
            learning_rate=config.learning.learning_rate,
            forgetting_rate=config.learning.forgetting_rate,
        )

        # Simulation should be functional
        simulation.start()
        simulation.step()
        assert simulation.time_step == 1


class TestCLIExecution:
    """Tests for CLI command execution."""

    @pytest.fixture
    def valid_config_file(self):
        """Create a temporary valid config file."""
        toml_content = """
seed = 42

[network]
n_neurons = 50
box_size = [10.0, 10.0, 10.0]
radius = 2.5
initial_weight = 0.1
weight_min = 0.0
weight_max = 1.0
initial_firing_fraction = 0.1

[learning]
threshold = 0.5
learning_rate = 0.01
forgetting_rate = 0.005
decay_alpha = 0.001

[visualization]
pygame_enabled = false
matplotlib_enabled = false
window_width = 800
window_height = 600
fps = 30
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False
        ) as f:
            f.write(toml_content)
            f.flush()
            yield Path(f.name)

    def test_cli_help_flag(self):
        """CLI should respond to --help flag."""
        result = subprocess.run(
            [sys.executable, "main.py", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "usage" in result.stdout.lower() or "help" in result.stdout.lower()

    def test_cli_with_config_file(self, valid_config_file):
        """CLI should accept -c flag for config file."""
        result = subprocess.run(
            [sys.executable, "main.py", "-c", str(valid_config_file), "--headless", "--steps", "5"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0

    def test_cli_headless_mode(self, valid_config_file):
        """CLI should run in headless mode without display."""
        result = subprocess.run(
            [sys.executable, "main.py", "-c", str(valid_config_file), "--headless", "--steps", "10"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0

    def test_cli_verbose_flag(self, valid_config_file):
        """CLI should accept -v flag for verbose output."""
        result = subprocess.run(
            [sys.executable, "main.py", "-c", str(valid_config_file), "--headless", "--steps", "3", "-v"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        # Verbose mode should produce more output
        assert len(result.stdout) > 0 or len(result.stderr) > 0

    def test_cli_invalid_config_path(self):
        """CLI should error with invalid config path."""
        result = subprocess.run(
            [sys.executable, "main.py", "-c", "/nonexistent/config.toml"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode != 0


class TestCLIDefaultConfig:
    """Tests for CLI default configuration behavior."""

    def test_cli_uses_default_config_when_none_specified(self):
        """CLI should use default config when no -c flag provided."""
        # This test assumes config/default.toml exists
        result = subprocess.run(
            [sys.executable, "main.py", "--headless", "--steps", "1"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Should work if default config exists, or provide helpful error
        # We check it doesn't crash with an unexpected error
        assert result.returncode == 0 or "config" in result.stderr.lower()
