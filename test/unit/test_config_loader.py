"""Unit tests for configuration loader.

RED phase: These tests should fail until config loader is implemented.
"""

import tempfile
from pathlib import Path

import pytest

from src.config.loader import (
    NetworkConfig,
    LearningConfig,
    VisualizationConfig,
    SimulationConfig,
    load_config,
    ConfigValidationError,
)


class TestNetworkConfig:
    """Tests for NetworkConfig dataclass."""

    def test_network_config_creation(self):
        """NetworkConfig should be created with all required fields."""
        config = NetworkConfig(
            n_neurons=300,
            box_size=(10.0, 10.0, 10.0),
            radius=2.5,
            initial_weight=0.1,
            weight_min=0.0,
            weight_max=1.0,
            initial_firing_fraction=0.1,
            leak_rate=0.1,
            reset_potential=1.0,
        )
        assert config.n_neurons == 300
        assert config.box_size == (10.0, 10.0, 10.0)
        assert config.radius == 2.5
        assert config.initial_weight == 0.1
        assert config.weight_min == 0.0
        assert config.weight_max == 1.0
        assert config.initial_firing_fraction == 0.1
        assert config.leak_rate == 0.1
        assert config.reset_potential == 1.0

    def test_network_config_is_frozen(self):
        """NetworkConfig should be immutable (frozen)."""
        config = NetworkConfig(
            n_neurons=300,
            box_size=(10.0, 10.0, 10.0),
            radius=2.5,
            initial_weight=0.1,
            weight_min=0.0,
            weight_max=1.0,
            initial_firing_fraction=0.1,
            leak_rate=0.1,
            reset_potential=1.0,
        )
        with pytest.raises(AttributeError):
            config.n_neurons = 500


class TestLearningConfig:
    """Tests for LearningConfig dataclass."""

    def test_learning_config_creation(self):
        """LearningConfig should be created with all required fields."""
        config = LearningConfig(
            threshold=0.5,
            learning_rate=0.01,
            forgetting_rate=0.005,
            decay_alpha=0.001,
            oja_alpha=0.001,
        )
        assert config.threshold == 0.5
        assert config.learning_rate == 0.01
        assert config.forgetting_rate == 0.005
        assert config.decay_alpha == 0.001
        assert config.oja_alpha == 0.001

    def test_learning_config_is_frozen(self):
        """LearningConfig should be immutable (frozen)."""
        config = LearningConfig(
            threshold=0.5,
            learning_rate=0.01,
            forgetting_rate=0.005,
            decay_alpha=0.001,
            oja_alpha=0.001,
        )
        with pytest.raises(AttributeError):
            config.threshold = 0.8


class TestVisualizationConfig:
    """Tests for VisualizationConfig dataclass."""

    def test_visualization_config_creation(self):
        """VisualizationConfig should be created with all required fields."""
        config = VisualizationConfig(
            pygame_enabled=True,
            matplotlib_enabled=True,
            window_width=800,
            window_height=600,
            fps=30,
        )
        assert config.pygame_enabled is True
        assert config.matplotlib_enabled is True
        assert config.window_width == 800
        assert config.window_height == 600
        assert config.fps == 30

    def test_visualization_config_is_frozen(self):
        """VisualizationConfig should be immutable (frozen)."""
        config = VisualizationConfig(
            pygame_enabled=True,
            matplotlib_enabled=True,
            window_width=800,
            window_height=600,
            fps=30,
        )
        with pytest.raises(AttributeError):
            config.fps = 60


class TestSimulationConfig:
    """Tests for SimulationConfig dataclass."""

    def test_simulation_config_creation(self):
        """SimulationConfig should contain all sub-configs."""
        network = NetworkConfig(
            n_neurons=300,
            box_size=(10.0, 10.0, 10.0),
            radius=2.5,
            initial_weight=0.1,
            weight_min=0.0,
            weight_max=1.0,
            initial_firing_fraction=0.1,
            leak_rate=0.1,
            reset_potential=1.0,
        )
        learning = LearningConfig(
            threshold=0.5,
            learning_rate=0.01,
            forgetting_rate=0.005,
            decay_alpha=0.001,
            oja_alpha=0.001,
        )
        visualization = VisualizationConfig(
            pygame_enabled=True,
            matplotlib_enabled=True,
            window_width=800,
            window_height=600,
            fps=30,
        )
        config = SimulationConfig(
            network=network,
            learning=learning,
            visualization=visualization,
            seed=42,
        )
        assert config.network == network
        assert config.learning == learning
        assert config.visualization == visualization
        assert config.seed == 42

    def test_simulation_config_seed_can_be_none(self):
        """SimulationConfig should allow None seed for random initialization."""
        network = NetworkConfig(
            n_neurons=300,
            box_size=(10.0, 10.0, 10.0),
            radius=2.5,
            initial_weight=0.1,
            weight_min=0.0,
            weight_max=1.0,
            initial_firing_fraction=0.1,
            leak_rate=0.1,
            reset_potential=1.0,
        )
        learning = LearningConfig(
            threshold=0.5,
            learning_rate=0.01,
            forgetting_rate=0.005,
            decay_alpha=0.001,
            oja_alpha=0.001,
        )
        visualization = VisualizationConfig(
            pygame_enabled=True,
            matplotlib_enabled=True,
            window_width=800,
            window_height=600,
            fps=30,
        )
        config = SimulationConfig(
            network=network,
            learning=learning,
            visualization=visualization,
            seed=None,
        )
        assert config.seed is None


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_from_valid_toml(self):
        """load_config should parse a valid TOML file."""
        toml_content = """
seed = 42

[network]
n_neurons = 300
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
pygame_enabled = true
matplotlib_enabled = true
window_width = 800
window_height = 600
fps = 30
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False
        ) as f:
            f.write(toml_content)
            f.flush()
            config = load_config(Path(f.name))

        assert isinstance(config, SimulationConfig)
        assert config.seed == 42
        assert config.network.n_neurons == 300
        assert config.learning.threshold == 0.5
        assert config.visualization.fps == 30

    def test_load_config_file_not_found(self):
        """load_config should raise error for non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_config(Path("/nonexistent/path/config.toml"))

    def test_load_config_invalid_toml_syntax(self):
        """load_config should raise error for invalid TOML syntax."""
        invalid_toml = "this is not valid toml [[[["
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False
        ) as f:
            f.write(invalid_toml)
            f.flush()
            with pytest.raises(Exception):  # TOML parse error
                load_config(Path(f.name))

    def test_load_config_missing_required_section(self):
        """load_config should raise error when required section is missing."""
        incomplete_toml = """
seed = 42

[network]
n_neurons = 300
box_size = [10.0, 10.0, 10.0]
radius = 2.5
initial_weight = 0.1
weight_min = 0.0
weight_max = 1.0
initial_firing_fraction = 0.1
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False
        ) as f:
            f.write(incomplete_toml)
            f.flush()
            with pytest.raises(ConfigValidationError):
                load_config(Path(f.name))

    def test_load_config_missing_required_field(self):
        """load_config should raise error when required field is missing."""
        incomplete_toml = """
seed = 42

[network]
n_neurons = 300
# missing box_size and other required fields

[learning]
threshold = 0.5
learning_rate = 0.01
forgetting_rate = 0.005
decay_alpha = 0.001

[visualization]
pygame_enabled = true
matplotlib_enabled = true
window_width = 800
window_height = 600
fps = 30
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False
        ) as f:
            f.write(incomplete_toml)
            f.flush()
            with pytest.raises(ConfigValidationError):
                load_config(Path(f.name))

    def test_load_config_seed_optional(self):
        """load_config should work when seed is not specified (defaults to None)."""
        toml_content = """
[network]
n_neurons = 300
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
pygame_enabled = true
matplotlib_enabled = true
window_width = 800
window_height = 600
fps = 30
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False
        ) as f:
            f.write(toml_content)
            f.flush()
            config = load_config(Path(f.name))

        assert config.seed is None


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_n_neurons_must_be_positive(self):
        """n_neurons must be a positive integer."""
        toml_content = """
seed = 42

[network]
n_neurons = 0
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
pygame_enabled = true
matplotlib_enabled = true
window_width = 800
window_height = 600
fps = 30
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False
        ) as f:
            f.write(toml_content)
            f.flush()
            with pytest.raises(ConfigValidationError):
                load_config(Path(f.name))

    def test_radius_must_be_positive(self):
        """radius must be positive."""
        toml_content = """
seed = 42

[network]
n_neurons = 300
box_size = [10.0, 10.0, 10.0]
radius = -1.0
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
pygame_enabled = true
matplotlib_enabled = true
window_width = 800
window_height = 600
fps = 30
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False
        ) as f:
            f.write(toml_content)
            f.flush()
            with pytest.raises(ConfigValidationError):
                load_config(Path(f.name))

    def test_initial_firing_fraction_in_range(self):
        """initial_firing_fraction must be in [0, 1]."""
        toml_content = """
seed = 42

[network]
n_neurons = 300
box_size = [10.0, 10.0, 10.0]
radius = 2.5
initial_weight = 0.1
weight_min = 0.0
weight_max = 1.0
initial_firing_fraction = 1.5

[learning]
threshold = 0.5
learning_rate = 0.01
forgetting_rate = 0.005
decay_alpha = 0.001

[visualization]
pygame_enabled = true
matplotlib_enabled = true
window_width = 800
window_height = 600
fps = 30
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False
        ) as f:
            f.write(toml_content)
            f.flush()
            with pytest.raises(ConfigValidationError):
                load_config(Path(f.name))

    def test_weight_min_less_than_max(self):
        """weight_min must be <= weight_max."""
        toml_content = """
seed = 42

[network]
n_neurons = 300
box_size = [10.0, 10.0, 10.0]
radius = 2.5
initial_weight = 0.1
weight_min = 1.0
weight_max = 0.0
initial_firing_fraction = 0.1

[learning]
threshold = 0.5
learning_rate = 0.01
forgetting_rate = 0.005
decay_alpha = 0.001

[visualization]
pygame_enabled = true
matplotlib_enabled = true
window_width = 800
window_height = 600
fps = 30
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False
        ) as f:
            f.write(toml_content)
            f.flush()
            with pytest.raises(ConfigValidationError):
                load_config(Path(f.name))

    def test_threshold_must_be_non_negative(self):
        """threshold must be >= 0."""
        toml_content = """
seed = 42

[network]
n_neurons = 300
box_size = [10.0, 10.0, 10.0]
radius = 2.5
initial_weight = 0.1
weight_min = 0.0
weight_max = 1.0
initial_firing_fraction = 0.1

[learning]
threshold = -0.5
learning_rate = 0.01
forgetting_rate = 0.005
decay_alpha = 0.001

[visualization]
pygame_enabled = true
matplotlib_enabled = true
window_width = 800
window_height = 600
fps = 30
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False
        ) as f:
            f.write(toml_content)
            f.flush()
            with pytest.raises(ConfigValidationError):
                load_config(Path(f.name))
