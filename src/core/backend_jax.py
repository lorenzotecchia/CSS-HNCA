"""JAX backend for GPU-accelerated array operations.

This module provides a JAX-based backend for GPU acceleration.
It requires JAX to be installed: pip install jax jaxlib

If JAX is not available, importing this module will raise ImportError,
and the system will fall back to NumPyBackend.
"""

try:
    import jax
    import jax.numpy as jnp
    from jax import random as jax_random
    JAX_AVAILABLE = True
except ImportError as e:
    raise ImportError(
        "JAX is required for JAXBackend. Install with: pip install jax jaxlib"
    ) from e

import numpy as np
from numpy import ndarray


class JAXBackend:
    """JAX-based backend for GPU computation.

    This backend uses JAX for GPU-accelerated array operations.
    Arrays are stored as JAX DeviceArrays but can be converted
    to NumPy arrays using to_numpy().

    Note: JAX arrays are immutable, so operations return new arrays
    rather than modifying in-place.
    """

    def __init__(self) -> None:
        """Initialize JAX backend."""
        # Check if GPU is available
        self._devices = jax.devices()
        self._has_gpu = any(d.platform == 'gpu' for d in self._devices)

    @property
    def has_gpu(self) -> bool:
        """Return True if GPU is available."""
        return self._has_gpu

    def zeros(self, shape: tuple[int, ...], dtype: type = np.float64) -> jnp.ndarray:
        """Create array filled with zeros."""
        return jnp.zeros(shape, dtype=dtype)

    def ones(self, shape: tuple[int, ...], dtype: type = np.float64) -> jnp.ndarray:
        """Create array filled with ones."""
        return jnp.ones(shape, dtype=dtype)

    def random_uniform(
        self,
        low: float,
        high: float,
        shape: tuple[int, ...],
        seed: int | None = None,
    ) -> jnp.ndarray:
        """Create array with uniform random values in [low, high)."""
        if seed is None:
            seed = 0
        key = jax_random.PRNGKey(seed)
        return jax_random.uniform(key, shape, minval=low, maxval=high)

    def random_bool(
        self,
        probability: float,
        shape: tuple[int, ...],
        seed: int | None = None,
    ) -> jnp.ndarray:
        """Create boolean array with given probability of True."""
        if seed is None:
            seed = 0
        key = jax_random.PRNGKey(seed)
        return jax_random.uniform(key, shape) < probability

    def matmul(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """Matrix multiplication."""
        return jnp.matmul(a, b)

    def transpose(self, a: jnp.ndarray) -> jnp.ndarray:
        """Matrix transpose."""
        return a.T

    def where(
        self, condition: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray
    ) -> jnp.ndarray:
        """Select elements based on condition."""
        return jnp.where(condition, x, y)

    def sqrt(self, a: jnp.ndarray) -> jnp.ndarray:
        """Element-wise square root."""
        return jnp.sqrt(a)

    def maximum(self, a: jnp.ndarray, b: jnp.ndarray | float) -> jnp.ndarray:
        """Element-wise maximum."""
        return jnp.maximum(a, b)

    def sum(self, a: jnp.ndarray, axis: int | None = None) -> jnp.ndarray | float:
        """Sum of array elements."""
        result = jnp.sum(a, axis=axis)
        if result.ndim == 0:
            return float(result)
        return result

    def mean(self, a: jnp.ndarray, axis: int | None = None) -> jnp.ndarray | float:
        """Mean of array elements."""
        result = jnp.mean(a, axis=axis)
        if result.ndim == 0:
            return float(result)
        return result

    def any(self, a: jnp.ndarray) -> bool:
        """Test if any element is True."""
        return bool(jnp.any(a))

    def copy(self, a: jnp.ndarray) -> jnp.ndarray:
        """Create a copy of the array.

        Note: JAX arrays are immutable, so this returns the same array.
        For true independence, use to_numpy() and back.
        """
        return jnp.array(a)

    def to_numpy(self, a: jnp.ndarray) -> ndarray:
        """Convert JAX array to numpy array."""
        return np.asarray(a)
