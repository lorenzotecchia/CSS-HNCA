"""Backend abstraction for array operations.

Provides a unified interface for NumPy and JAX backends, enabling
CPU and GPU acceleration without changing core simulation code.
"""

from typing import Protocol, runtime_checkable

import numpy as np
from numpy import ndarray


@runtime_checkable
class ArrayBackend(Protocol):
    """Protocol defining array operations for simulation backends.

    This protocol allows the simulation to use either NumPy (CPU) or
    JAX (GPU) for array operations without code changes.
    """

    def zeros(self, shape: tuple[int, ...], dtype: type = np.float64) -> ndarray:
        """Create array filled with zeros."""
        ...

    def ones(self, shape: tuple[int, ...], dtype: type = np.float64) -> ndarray:
        """Create array filled with ones."""
        ...

    def random_uniform(
        self,
        low: float,
        high: float,
        shape: tuple[int, ...],
        seed: int | None = None,
    ) -> ndarray:
        """Create array with uniform random values in [low, high)."""
        ...

    def random_bool(
        self,
        probability: float,
        shape: tuple[int, ...],
        seed: int | None = None,
    ) -> ndarray:
        """Create boolean array with given probability of True."""
        ...

    def matmul(self, a: ndarray, b: ndarray) -> ndarray:
        """Matrix multiplication."""
        ...

    def transpose(self, a: ndarray) -> ndarray:
        """Matrix transpose."""
        ...

    def where(self, condition: ndarray, x: ndarray, y: ndarray) -> ndarray:
        """Select elements based on condition."""
        ...

    def sqrt(self, a: ndarray) -> ndarray:
        """Element-wise square root."""
        ...

    def maximum(self, a: ndarray, b: ndarray | float) -> ndarray:
        """Element-wise maximum."""
        ...

    def sum(self, a: ndarray, axis: int | None = None) -> ndarray | float:
        """Sum of array elements."""
        ...

    def mean(self, a: ndarray, axis: int | None = None) -> ndarray | float:
        """Mean of array elements."""
        ...

    def any(self, a: ndarray) -> bool:
        """Test if any element is True."""
        ...

    def copy(self, a: ndarray) -> ndarray:
        """Create a copy of the array."""
        ...

    def to_numpy(self, a: ndarray) -> ndarray:
        """Convert to numpy array (identity for NumPy backend)."""
        ...


class NumPyBackend:
    """NumPy-based backend for CPU computation.

    This is the default backend, providing all array operations
    using NumPy for portable CPU execution.
    """

    def zeros(self, shape: tuple[int, ...], dtype: type = np.float64) -> ndarray:
        """Create array filled with zeros."""
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape: tuple[int, ...], dtype: type = np.float64) -> ndarray:
        """Create array filled with ones."""
        return np.ones(shape, dtype=dtype)

    def random_uniform(
        self,
        low: float,
        high: float,
        shape: tuple[int, ...],
        seed: int | None = None,
    ) -> ndarray:
        """Create array with uniform random values in [low, high)."""
        rng = np.random.default_rng(seed)
        return rng.uniform(low, high, shape)

    def random_bool(
        self,
        probability: float,
        shape: tuple[int, ...],
        seed: int | None = None,
    ) -> ndarray:
        """Create boolean array with given probability of True."""
        rng = np.random.default_rng(seed)
        return rng.random(shape) < probability

    def matmul(self, a: ndarray, b: ndarray) -> ndarray:
        """Matrix multiplication."""
        return np.matmul(a, b)

    def transpose(self, a: ndarray) -> ndarray:
        """Matrix transpose."""
        return a.T

    def where(self, condition: ndarray, x: ndarray, y: ndarray) -> ndarray:
        """Select elements based on condition."""
        return np.where(condition, x, y)

    def sqrt(self, a: ndarray) -> ndarray:
        """Element-wise square root."""
        return np.sqrt(a)

    def maximum(self, a: ndarray, b: ndarray | float) -> ndarray:
        """Element-wise maximum."""
        return np.maximum(a, b)

    def sum(self, a: ndarray, axis: int | None = None) -> ndarray | float:
        """Sum of array elements."""
        result = np.sum(a, axis=axis)
        # Return Python scalar if result is 0-d array
        if isinstance(result, np.ndarray) and result.ndim == 0:
            return float(result)
        return result

    def mean(self, a: ndarray, axis: int | None = None) -> ndarray | float:
        """Mean of array elements."""
        result = np.mean(a, axis=axis)
        # Return Python scalar if result is 0-d array
        if isinstance(result, np.ndarray) and result.ndim == 0:
            return float(result)
        return result

    def any(self, a: ndarray) -> bool:
        """Test if any element is True."""
        return bool(np.any(a))

    def copy(self, a: ndarray) -> ndarray:
        """Create a copy of the array."""
        return a.copy()

    def to_numpy(self, a: ndarray) -> ndarray:
        """Convert to numpy array (identity for NumPy backend)."""
        return np.asarray(a)


# Global default backend instance
_default_backend: ArrayBackend | None = None


def get_backend(prefer_gpu: bool = False) -> ArrayBackend:
    """Get the array computation backend.

    Args:
        prefer_gpu: If True, attempt to use JAX backend for GPU acceleration.
                   Falls back to NumPy if JAX is not available.

    Returns:
        ArrayBackend instance (NumPyBackend or JAXBackend)
    """
    global _default_backend

    if prefer_gpu:
        # Try to import JAX backend
        try:
            from src.core.backend_jax import JAXBackend
            return JAXBackend()
        except ImportError:
            # JAX not available, fall back to NumPy
            pass

    # Return NumPy backend
    if _default_backend is None:
        _default_backend = NumPyBackend()
    return _default_backend
