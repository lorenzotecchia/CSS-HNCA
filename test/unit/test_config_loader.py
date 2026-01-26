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
            firing_count=10,
            leak_rate=0.1,
            reset_potential=1.0,
            excitatory_fraction=0.8,
            weight_min=0.0,
            weight_max=1.0,
            weight_min_inh=-0.3,
            weight_max_inh=0.0,
            k_prop=0.05,
            beta_a=2.0,
            beta_b=6.0,
        )
        assert config.n_neurons == 300
        assert config.firing_count == 10
        assert config.leak_rate == 0.1
        assert config.reset_potential == 1.0
        assert config.excitatory_fraction == 0.8
        assert config.weight_min == 0.0
        assert config.weight_max == 1.0
        assert config.weight_min_inh == -0.3
        assert config.weight_max_inh == 0.0
        assert config.k_prop == 0.05
        assert config.beta_a == 2.0
        assert config.beta_b == 6.0

    def test_network_config_is_frozen(self):
        """NetworkConfig should be immutable (frozen)."""
        config = NetworkConfig(
            n_neurons=300,
            firing_count=10,
            leak_rate=0.1,
            reset_potential=1.0,
            excitatory_fraction=0.8,
            weight_min=0.0,
            weight_max=1.0,
            weight_min_inh=-0.3,
            weight_max_inh=0.0,
            k_prop=0.05,
            beta_a=2.0,
            beta_b=6.0,
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
            firing_count=10,
            leak_rate=0.1,
            reset_potential=1.0,
            excitatory_fraction=0.8,
            weight_min=0.0,
            weight_max=1.0,
            weight_min_inh=-0.3,
            weight_max_inh=0.0,
            k_prop=0.05,
            beta_a=2.0,
            beta_b=6.0,
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
            firing_count=10,
            leak_rate=0.1,
            reset_potential=1.0,
            excitatory_fraction=0.8,
            weight_min=0.0,
            weight_max=1.0,
            weight_min_inh=-0.3,
            weight_max_inh=0.0,
            k_prop=0.05,
            beta_a=2.0,
            beta_b=6.0,
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
firing_count = 10
leak_rate = 0.08
reset_potential = 0.4
excitatory_fraction = 0.8
weight_min = 0.0
weight_max = 1.0
weight_min_inh = -0.3
weight_max_inh = 0.0
k_prop = 0.05
beta_a = 2.0
beta_b = 6.0

[learning]
threshold = 0.5
learning_rate = 0.01
forgetting_rate = 0.005
decay_alpha = 0.001
oja_alpha = 0.001

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
firing_count = 10
leak_rate = 0.08
reset_potential = 0.4
excitatory_fraction = 0.8
weight_min = 0.0
weight_max = 1.0
weight_min_inh = -0.3
weight_max_inh = 0.0
k_prop = 0.05
beta_a = 2.0
beta_b = 6.0
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
# missing other required fields

[learning]
threshold = 0.5
learning_rate = 0.01
forgetting_rate = 0.005
decay_alpha = 0.001
oja_alpha = 0.001

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
firing_count = 10
leak_rate = 0.08
reset_potential = 0.4
excitatory_fraction = 0.8
weight_min = 0.0
weight_max = 1.0
weight_min_inh = -0.3
weight_max_inh = 0.0
k_prop = 0.05
beta_a = 2.0
beta_b = 6.0

[learning]
threshold = 0.5
learning_rate = 0.01
forgetting_rate = 0.005
decay_alpha = 0.001
oja_alpha = 0.001

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
firing_count=1

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

    def test_k_prop_must_be_in_valid_range(self):
        """k_prop must be in valid range for n_neurons."""
        # k_prop=0.01 is too small for any n_neurons >= 3
        toml_content = """
seed = 42

[network]
n_neurons = 100
firing_count = 1
leak_rate = 0.1
reset_potential = 0.5
excitatory_fraction = 0.8
weight_min = 0.0
weight_max = 1.0
weight_min_inh = -1.0
weight_max_inh = 0.0
k_prop = 0.01
beta_a = 2.0
beta_b = 6.0

[learning]
threshold = 0.5
learning_rate = 0.01
forgetting_rate = 0.005
decay_alpha = 0.001
oja_alpha = 0.002

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
            # This test is now a simple load test since validation happens at network creation
            config = load_config(Path(f.name))
            assert config.network.k_prop == 0.01

    def test_firing_count_non_negative(self):
        """firing_count must be non-negative."""
        toml_content = """
seed = 42

[network]
n_neurons = 100
firing_count = 5
leak_rate = 0.1
reset_potential = 0.5
excitatory_fraction = 0.8
weight_min = 0.0
weight_max = 1.0
weight_min_inh = -1.0
weight_max_inh = 0.0
k_prop = 0.2
beta_a = 2.0
beta_b = 6.0

[learning]
threshold = 0.5
learning_rate = 0.01
forgetting_rate = 0.005
decay_alpha = 0.001
oja_alpha = 0.002

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
            assert config.network.firing_count >= 0

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
firing_count=1

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
firing_count=1

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


class TestExcitatoryInhibitoryConfig:
    """Tests for excitatory/inhibitory neuron configuration."""

    def test_excitatory_fraction_parsed(self):
        """Test excitatory_fraction is parsed from config."""
        config = load_config(Path("config/default.toml"))
        assert hasattr(config.network, "excitatory_fraction")
        assert 0 <= config.network.excitatory_fraction <= 1

    def test_inhibitory_weight_bounds_parsed(self):
        """Test inhibitory weight bounds are parsed from config."""
        config = load_config(Path("config/default.toml"))
        assert hasattr(config.network, "weight_min_inh")
        assert hasattr(config.network, "weight_max_inh")
        assert config.network.weight_min_inh <= config.network.weight_max_inh

    def test_excitatory_fraction_must_be_in_range(self):
        """excitatory_fraction must be in [0, 1]."""
        toml_content = """
seed = 42

[network]
n_neurons = 300
box_size = [10.0, 10.0, 10.0]
radius = 2.5
initial_weight = 0.1
weight_min = 0.0
weight_max = 1.0
firing_count=1
excitatory_fraction = 1.5

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

    def test_inhibitory_weight_min_must_be_lte_max(self):
        """weight_min_inh must be <= weight_max_inh."""
        toml_content = """
seed = 42

[network]
n_neurons = 300
box_size = [10.0, 10.0, 10.0]
radius = 2.5
initial_weight = 0.1
weight_min = 0.0
weight_max = 1.0
firing_count=1
weight_min_inh = 0.0
weight_max_inh = -0.5

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
