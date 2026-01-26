"""Property-based tests for configuration system.

RED phase: These tests should fail until config is implemented.
Uses Hypothesis for property-based testing.
"""

import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.config.loader import (
    NetworkConfig,
    LearningConfig,
    VisualizationConfig,
    SimulationConfig,
    load_config,
    ConfigValidationError,
)
from src.core.network import Network
from src.core.neuron_state import NeuronState
from src.core.simulation import Simulation


# Strategies for generating valid config values
valid_n_neurons = st.integers(min_value=1, max_value=1000)
small_n_neurons = st.integers(min_value=1, max_value=100)  # For tests that create networks
valid_box_dimension = st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False)
valid_radius = st.floats(min_value=0.01, max_value=50.0, allow_nan=False, allow_infinity=False)
valid_weight = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
valid_fraction = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
valid_learning_rate = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
valid_threshold = st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)
valid_dimension = st.integers(min_value=100, max_value=2000)
valid_fps = st.integers(min_value=1, max_value=120)
optional_seed = st.one_of(st.none(), st.integers(min_value=0, max_value=2**31 - 1))


@st.composite
def valid_box_size(draw):
    """Generate a valid 3D box size tuple."""
    return (
        draw(valid_box_dimension),
        draw(valid_box_dimension),
        draw(valid_box_dimension),
    )


@st.composite
def valid_network_config(draw, n_neurons_strategy=None):
    """Generate a valid NetworkConfig."""
    if n_neurons_strategy is None:
        n_neurons_strategy = st.integers(min_value=10, max_value=100)
    
    n_neurons = draw(n_neurons_strategy)
    # k_prop must be in [2/n, 1-1/n]
    k_min = 2 / n_neurons
    k_max = 1 - 1 / n_neurons
    k_prop = draw(st.floats(min_value=k_min, max_value=k_max, allow_nan=False))
    
    weight_min = draw(valid_weight)
    weight_max = draw(st.floats(min_value=weight_min, max_value=1.0, allow_nan=False, allow_infinity=False))
    excitatory_fraction = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    weight_min_inh = draw(st.floats(min_value=-1.0, max_value=0.0, allow_nan=False, allow_infinity=False))
    weight_max_inh = draw(st.floats(min_value=weight_min_inh, max_value=0.0, allow_nan=False, allow_infinity=False))
    firing_count = draw(st.integers(min_value=1, max_value=max(1, n_neurons // 2)))

    return NetworkConfig(
        n_neurons=n_neurons,
        firing_count=firing_count,
        leak_rate=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        reset_potential=draw(st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False)),
        excitatory_fraction=excitatory_fraction,
        weight_min=weight_min,
        weight_max=weight_max,
        weight_min_inh=weight_min_inh,
        weight_max_inh=weight_max_inh,
        k_prop=k_prop,
        beta_a=draw(st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)),
        beta_b=draw(st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)),
    )


@st.composite
def small_network_config(draw):
    """Generate a valid NetworkConfig with small network size (for tests that create networks)."""
    n_neurons = draw(st.integers(min_value=10, max_value=30))
    # k_prop must be in [2/n, 1-1/n]
    k_min = 2 / n_neurons
    k_max = 1 - 1 / n_neurons
    k_prop = draw(st.floats(min_value=k_min, max_value=k_max, allow_nan=False))
    
    weight_min = draw(valid_weight)
    weight_max = draw(st.floats(min_value=weight_min, max_value=1.0, allow_nan=False, allow_infinity=False))
    excitatory_fraction = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    weight_min_inh = draw(st.floats(min_value=-1.0, max_value=0.0, allow_nan=False, allow_infinity=False))
    weight_max_inh = draw(st.floats(min_value=weight_min_inh, max_value=0.0, allow_nan=False, allow_infinity=False))
    firing_count = draw(st.integers(min_value=1, max_value=max(1, n_neurons // 2)))

    return NetworkConfig(
        n_neurons=n_neurons,
        firing_count=firing_count,
        leak_rate=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        reset_potential=draw(st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False)),
        excitatory_fraction=excitatory_fraction,
        weight_min=weight_min,
        weight_max=weight_max,
        weight_min_inh=weight_min_inh,
        weight_max_inh=weight_max_inh,
        k_prop=k_prop,
        beta_a=draw(st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)),
        beta_b=draw(st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)),
    )


@st.composite
def valid_learning_config(draw):
    """Generate a valid LearningConfig."""
    return LearningConfig(
        threshold=draw(valid_threshold),
        learning_rate=draw(valid_learning_rate),
        forgetting_rate=draw(valid_learning_rate),
        decay_alpha=draw(valid_learning_rate),
        oja_alpha=draw(st.floats(min_value=0.0, max_value=0.1, allow_nan=False, allow_infinity=False)),
    )


@st.composite
def valid_visualization_config(draw):
    """Generate a valid VisualizationConfig."""
    return VisualizationConfig(
        pygame_enabled=draw(st.booleans()),
        matplotlib_enabled=draw(st.booleans()),
        window_width=draw(valid_dimension),
        window_height=draw(valid_dimension),
        fps=draw(valid_fps),
    )


@st.composite
def valid_simulation_config(draw):
    """Generate a valid SimulationConfig."""
    return SimulationConfig(
        network=draw(valid_network_config()),
        learning=draw(valid_learning_config()),
        visualization=draw(valid_visualization_config()),
        seed=draw(optional_seed),
    )


@st.composite
def small_simulation_config(draw):
    """Generate a valid SimulationConfig with small network (for tests that create networks)."""
    return SimulationConfig(
        network=draw(small_network_config()),
        learning=draw(valid_learning_config()),
        visualization=draw(valid_visualization_config()),
        seed=draw(optional_seed),
    )


class TestConfigProperties:
    """Property-based tests for config dataclasses."""

    @given(config=valid_network_config())
    @settings(max_examples=50)
    def test_network_config_weight_bounds_invariant(self, config):
        """Weight min should always be <= weight max."""
        assert config.weight_min <= config.weight_max

    @given(config=valid_network_config())
    @settings(max_examples=50)
    def test_network_config_k_prop_valid(self, config):
        """k_prop should be within valid range for n_neurons."""
        k_min = 2 / config.n_neurons
        k_max = 1 - 1 / config.n_neurons
        assert k_min <= config.k_prop <= k_max

    @given(config=valid_network_config())
    @settings(max_examples=50)
    def test_network_config_firing_count_valid(self, config):
        """Firing count should be in [0, n_neurons]."""
        assert 0 <= config.firing_count <= config.n_neurons

    @given(config=valid_learning_config())
    @settings(max_examples=50)
    def test_learning_config_threshold_non_negative(self, config):
        """Threshold should be non-negative."""
        assert config.threshold >= 0

    @given(config=valid_learning_config())
    @settings(max_examples=50)
    def test_learning_config_rates_non_negative(self, config):
        """Learning and forgetting rates should be non-negative."""
        assert config.learning_rate >= 0
        assert config.forgetting_rate >= 0
        assert config.decay_alpha >= 0


class TestConfigToSimulation:
    """Property tests: valid config always produces valid simulation."""

    @given(config=small_simulation_config())
    @settings(max_examples=20, deadline=None)
    def test_valid_config_produces_valid_network(self, config):
        """Any valid config should produce a valid Network."""
        network = Network.create_beta_weighted_directed(
            n_neurons=config.network.n_neurons,
            k_prop=config.network.k_prop,
            seed=config.seed,
        )

        assert network.n_neurons == config.network.n_neurons
        assert network.positions.shape == (config.network.n_neurons, 3)
        assert network.weight_matrix.shape == (config.network.n_neurons, config.network.n_neurons)

    @given(config=small_simulation_config())
    @settings(max_examples=20, deadline=None)
    def test_valid_config_produces_valid_neuron_state(self, config):
        """Any valid config should produce a valid NeuronState."""
        state = NeuronState.create(
            n_neurons=config.network.n_neurons,
            threshold=config.learning.threshold,
            firing_count=config.network.firing_count,
            seed=config.seed,
        )

        assert state.firing.shape[0] == config.network.n_neurons
        assert state.threshold == config.learning.threshold

    @given(config=small_simulation_config())
    @settings(max_examples=20, deadline=None)
    def test_valid_config_produces_runnable_simulation(self, config):
        """Any valid config should produce a simulation that can step."""
        network = Network.create_beta_weighted_directed(
            n_neurons=config.network.n_neurons,
            k_prop=config.network.k_prop,
            seed=config.seed,
        )
        state = NeuronState.create(
            n_neurons=config.network.n_neurons,
            threshold=config.learning.threshold,
            firing_count=config.network.firing_count,
            seed=config.seed,
        )
        simulation = Simulation(
            network=network,
            state=state,
            learning_rate=config.learning.learning_rate,
            forgetting_rate=config.learning.forgetting_rate,
        )

        # Should be able to step without errors
        initial_time = simulation.time_step
        simulation.step()
        assert simulation.time_step == initial_time + 1


class TestConfigRoundTrip:
    """Property tests for config serialization/deserialization."""

    @given(config=valid_simulation_config())
    @settings(max_examples=20, deadline=None)
    def test_config_toml_roundtrip(self, config):
        """Config should survive TOML round-trip with equivalent values."""
        # Serialize to TOML with new structure
        seed_line = f"seed = {config.seed}" if config.seed is not None else ""
        
        toml_content = f"""
{seed_line}

[network]
n_neurons = {config.network.n_neurons}
firing_count = {config.network.firing_count}
leak_rate = {config.network.leak_rate}
reset_potential = {config.network.reset_potential}
excitatory_fraction = {config.network.excitatory_fraction}
weight_min = {config.network.weight_min}
weight_max = {config.network.weight_max}
weight_min_inh = {config.network.weight_min_inh}
weight_max_inh = {config.network.weight_max_inh}
k_prop = {config.network.k_prop}
beta_a = {config.network.beta_a}
beta_b = {config.network.beta_b}

[learning]
threshold = {config.learning.threshold}
learning_rate = {config.learning.learning_rate}
forgetting_rate = {config.learning.forgetting_rate}
decay_alpha = {config.learning.decay_alpha}
oja_alpha = {config.learning.oja_alpha}

[visualization]
pygame_enabled = {str(config.visualization.pygame_enabled).lower()}
matplotlib_enabled = {str(config.visualization.matplotlib_enabled).lower()}
window_width = {config.visualization.window_width}
window_height = {config.visualization.window_height}
fps = {config.visualization.fps}
"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False
        ) as f:
            f.write(toml_content)
            f.flush()
            loaded = load_config(Path(f.name))

        # Check key values match
        assert loaded.network.n_neurons == config.network.n_neurons
        assert loaded.learning.threshold == pytest.approx(config.learning.threshold)
        assert loaded.visualization.fps == config.visualization.fps
