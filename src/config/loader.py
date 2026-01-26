"""Configuration loader for neural cellular automata.

Parses TOML configuration files into typed dataclasses.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigValidationError(ValueError):
    """Raised when configuration validation fails."""

    pass


@dataclass(frozen=True)
class NetworkConfig:
    """Configuration for network structure.

    Attributes:
        n_neurons: Number of neurons in the network
        firing_count: Number of neurons firing at t=0
        leak_rate: LIF leak rate λ (potential decay fraction per step)
        reset_potential: Amount subtracted from potential after firing
        excitatory_fraction: Fraction of neurons that are excitatory
        weight_min: Minimum weight for excitatory neurons
        weight_max: Maximum weight for excitatory neurons
        weight_min_inh: Minimum weight for inhibitory neurons
        weight_max_inh: Maximum weight for inhibitory neurons
        k_prop: Average degree proportion for beta-directed network
        beta_a: Beta distribution parameter a
        beta_b: Beta distribution parameter b
    """

    n_neurons: int
    firing_count: int
    leak_rate: float
    reset_potential: float
    excitatory_fraction: float
    weight_min: float
    weight_max: float
    weight_min_inh: float
    weight_max_inh: float
    k_prop: float
    beta_a: float
    beta_b: float


@dataclass(frozen=True)
class LearningConfig:
    """Configuration for learning parameters.

    Attributes:
        threshold: Firing threshold γ
        learning_rate: l parameter for LTP
        forgetting_rate: f parameter for LTD
        decay_alpha: Baseline weight decay rate
        oja_alpha: Oja rule decay coefficient
    """

    threshold: float
    learning_rate: float
    forgetting_rate: float
    decay_alpha: float
    oja_alpha: float


@dataclass(frozen=True)
class VisualizationConfig:
    """Configuration for visualization settings.

    Attributes:
        pygame_enabled: Whether to enable pygame 3D visualization
        matplotlib_enabled: Whether to enable matplotlib analytics
        window_width: Window width in pixels
        window_height: Window height in pixels
        fps: Target frames per second
    """

    pygame_enabled: bool
    matplotlib_enabled: bool
    window_width: int
    window_height: int
    fps: int


@dataclass(frozen=True)
class SimulationConfig:
    """Complete simulation configuration.

    Attributes:
        network: Network structure configuration
        learning: Learning parameters configuration
        visualization: Visualization settings
        seed: Random seed (None for random initialization)
    """

    network: NetworkConfig
    learning: LearningConfig
    visualization: VisualizationConfig
    seed: int | None


def _validate_network_config(data: dict[str, Any]) -> None:
    """Validate network configuration values."""
    if data.get("n_neurons", 0) <= 0:
        raise ConfigValidationError("n_neurons must be positive")

    firing_count = data.get("firing_count", -1)
    if not isinstance(firing_count, int) or firing_count < 0:
        raise ConfigValidationError("firing_count must be a non-negative integer")

    leak_rate = data.get("leak_rate", 0.0)
    if not 0 <= leak_rate <= 1:
        raise ConfigValidationError("leak_rate must be in [0, 1]")

    reset_potential = data.get("reset_potential", 0.0)
    if reset_potential < 0:
        raise ConfigValidationError("reset_potential must be >= 0")

    excitatory_fraction = data.get("excitatory_fraction", 0.8)
    if not 0 <= excitatory_fraction <= 1:
        raise ConfigValidationError("excitatory_fraction must be in [0, 1]")

    weight_min = data.get("weight_min", 0.0)
    weight_max = data.get("weight_max", 1.0)
    if weight_min > weight_max:
        raise ConfigValidationError("weight_min must be <= weight_max")

    weight_min_inh = data.get("weight_min_inh", -0.3)
    weight_max_inh = data.get("weight_max_inh", 0.0)
    if weight_min_inh > weight_max_inh:
        raise ConfigValidationError("weight_min_inh must be <= weight_max_inh")

    k_prop = data.get("k_prop", 0.05)
    if k_prop <= 0:
        raise ConfigValidationError("k_prop must be > 0")

    beta_a = data.get("beta_a", 2.0)
    if beta_a <= 0:
        raise ConfigValidationError("beta_a must be > 0")

    beta_b = data.get("beta_b", 6.0)
    if beta_b <= 0:
        raise ConfigValidationError("beta_b must be > 0")


def _validate_learning_config(data: dict[str, Any]) -> None:
    """Validate learning configuration values."""
    if data.get("threshold", -1) < 0:
        raise ConfigValidationError("threshold must be >= 0")


def _parse_network_config(data: dict[str, Any]) -> NetworkConfig:
    """Parse and validate network configuration section."""
    required_fields = [
        "n_neurons",
        "firing_count",
    ]

    for field in required_fields:
        if field not in data:
            raise ConfigValidationError(f"Missing required field: network.{field}")

    _validate_network_config(data)

    return NetworkConfig(
        n_neurons=int(data["n_neurons"]),
        firing_count=int(data["firing_count"]),
        leak_rate=float(data.get("leak_rate", 0.0)),
        reset_potential=float(data.get("reset_potential", 0.0)),
        excitatory_fraction=float(data.get("excitatory_fraction", 0.8)),
        weight_min=float(data.get("weight_min", 0.0)),
        weight_max=float(data.get("weight_max", 1.0)),
        weight_min_inh=float(data.get("weight_min_inh", -0.3)),
        weight_max_inh=float(data.get("weight_max_inh", 0.0)),
        k_prop=float(data.get("k_prop", 0.05)),
        beta_a=float(data.get("beta_a", 2.0)),
        beta_b=float(data.get("beta_b", 6.0)),
    )


def _parse_learning_config(data: dict[str, Any]) -> LearningConfig:
    """Parse and validate learning configuration section."""
    required_fields = ["threshold", "learning_rate", "forgetting_rate", "decay_alpha"]

    for field in required_fields:
        if field not in data:
            raise ConfigValidationError(f"Missing required field: learning.{field}")

    _validate_learning_config(data)

    return LearningConfig(
        threshold=float(data["threshold"]),
        learning_rate=float(data["learning_rate"]),
        forgetting_rate=float(data["forgetting_rate"]),
        decay_alpha=float(data["decay_alpha"]),
        oja_alpha=float(data.get("oja_alpha", 0.0)),
    )


def _parse_visualization_config(data: dict[str, Any]) -> VisualizationConfig:
    """Parse visualization configuration section."""
    required_fields = [
        "pygame_enabled",
        "matplotlib_enabled",
        "window_width",
        "window_height",
        "fps",
    ]

    for field in required_fields:
        if field not in data:
            raise ConfigValidationError(f"Missing required field: visualization.{field}")

    return VisualizationConfig(
        pygame_enabled=bool(data["pygame_enabled"]),
        matplotlib_enabled=bool(data["matplotlib_enabled"]),
        window_width=int(data["window_width"]),
        window_height=int(data["window_height"]),
        fps=int(data["fps"]),
    )


def load_config(path: Path) -> SimulationConfig:
    """Load and parse a TOML configuration file.

    Args:
        path: Path to the TOML configuration file

    Returns:
        Parsed SimulationConfig

    Raises:
        FileNotFoundError: If the config file doesn't exist
        ConfigValidationError: If config validation fails
        tomllib.TOMLDecodeError: If TOML syntax is invalid
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "rb") as f:
        data = tomllib.load(f)

    # Check required sections
    if "network" not in data:
        raise ConfigValidationError("Missing required section: [network]")
    if "learning" not in data:
        raise ConfigValidationError("Missing required section: [learning]")
    if "visualization" not in data:
        raise ConfigValidationError("Missing required section: [visualization]")

    network = _parse_network_config(data["network"])
    learning = _parse_learning_config(data["learning"])
    visualization = _parse_visualization_config(data["visualization"])

    # Seed is optional
    seed = data.get("seed")
    if seed is not None:
        seed = int(seed)

    return SimulationConfig(
        network=network,
        learning=learning,
        visualization=visualization,
        seed=seed,
    )
