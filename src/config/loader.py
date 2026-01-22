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
        box_size: (x, y, z) dimensions of the 3D volume
        radius: Maximum distance for structural connectivity
        initial_weight: Initial synaptic weight for connected neurons
        weight_min: Minimum allowed weight value
        weight_max: Maximum allowed weight value
        initial_firing_fraction: Fraction of neurons firing at t=0
    """

    n_neurons: int
    box_size: tuple[float, float, float]
    radius: float
    initial_weight: float
    weight_min: float
    weight_max: float
    initial_firing_fraction: float


@dataclass(frozen=True)
class LearningConfig:
    """Configuration for learning parameters.

    Attributes:
        threshold: Firing threshold γ
        learning_rate: l parameter for LTP
        forgetting_rate: f parameter for LTD
        decay_alpha: α parameter for Oja/decay
    """

    threshold: float
    learning_rate: float
    forgetting_rate: float
    decay_alpha: float


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

    if data.get("radius", 0) <= 0:
        raise ConfigValidationError("radius must be positive")

    firing_fraction = data.get("initial_firing_fraction", -1)
    if not 0 <= firing_fraction <= 1:
        raise ConfigValidationError("initial_firing_fraction must be in [0, 1]")

    weight_min = data.get("weight_min", 0)
    weight_max = data.get("weight_max", 0)
    if weight_min > weight_max:
        raise ConfigValidationError("weight_min must be <= weight_max")

    box_size = data.get("box_size", [])
    if len(box_size) != 3 or any(d <= 0 for d in box_size):
        raise ConfigValidationError("box_size must be 3 positive values")


def _validate_learning_config(data: dict[str, Any]) -> None:
    """Validate learning configuration values."""
    if data.get("threshold", -1) < 0:
        raise ConfigValidationError("threshold must be >= 0")


def _parse_network_config(data: dict[str, Any]) -> NetworkConfig:
    """Parse and validate network configuration section."""
    required_fields = [
        "n_neurons",
        "box_size",
        "radius",
        "initial_weight",
        "weight_min",
        "weight_max",
        "initial_firing_fraction",
    ]

    for field in required_fields:
        if field not in data:
            raise ConfigValidationError(f"Missing required field: network.{field}")

    _validate_network_config(data)

    return NetworkConfig(
        n_neurons=int(data["n_neurons"]),
        box_size=tuple(float(x) for x in data["box_size"]),  # type: ignore
        radius=float(data["radius"]),
        initial_weight=float(data["initial_weight"]),
        weight_min=float(data["weight_min"]),
        weight_max=float(data["weight_max"]),
        initial_firing_fraction=float(data["initial_firing_fraction"]),
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
