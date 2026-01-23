"""Unit tests for backend abstraction.

RED phase: These tests should fail until backend is implemented.
"""

import numpy as np
import pytest

from src.core.backend import ArrayBackend, NumPyBackend, get_backend


class TestNumPyBackendBasicOperations:
    """Tests for basic NumPy backend operations."""

    def test_backend_is_array_backend(self):
        """NumPyBackend should implement ArrayBackend protocol."""
        backend = NumPyBackend()
        assert isinstance(backend, ArrayBackend)

    def test_zeros_shape(self):
        """zeros should create array with correct shape."""
        backend = NumPyBackend()
        arr = backend.zeros((3, 4))
        assert arr.shape == (3, 4)

    def test_zeros_values(self):
        """zeros should create array filled with zeros."""
        backend = NumPyBackend()
        arr = backend.zeros((5, 5))
        assert np.all(arr == 0)

    def test_zeros_dtype_float(self):
        """zeros with float dtype should create float array."""
        backend = NumPyBackend()
        arr = backend.zeros((3, 3), dtype=np.float64)
        assert arr.dtype == np.float64

    def test_zeros_dtype_bool(self):
        """zeros with bool dtype should create bool array."""
        backend = NumPyBackend()
        arr = backend.zeros((3, 3), dtype=np.bool_)
        assert arr.dtype == np.bool_

    def test_ones_shape(self):
        """ones should create array with correct shape."""
        backend = NumPyBackend()
        arr = backend.ones((2, 3))
        assert arr.shape == (2, 3)

    def test_ones_values(self):
        """ones should create array filled with ones."""
        backend = NumPyBackend()
        arr = backend.ones((4, 4))
        assert np.all(arr == 1)


class TestNumPyBackendRandomOperations:
    """Tests for random number generation."""

    def test_random_uniform_shape(self):
        """random_uniform should create array with correct shape."""
        backend = NumPyBackend()
        arr = backend.random_uniform(0.0, 1.0, (10, 5), seed=42)
        assert arr.shape == (10, 5)

    def test_random_uniform_bounds(self):
        """random_uniform values should be within [low, high)."""
        backend = NumPyBackend()
        arr = backend.random_uniform(2.0, 5.0, (100,), seed=42)
        assert np.all(arr >= 2.0)
        assert np.all(arr < 5.0)

    def test_random_uniform_reproducible(self):
        """Same seed should produce same random values."""
        backend = NumPyBackend()
        arr1 = backend.random_uniform(0.0, 1.0, (10,), seed=123)
        arr2 = backend.random_uniform(0.0, 1.0, (10,), seed=123)
        assert np.array_equal(arr1, arr2)

    def test_random_uniform_different_seeds(self):
        """Different seeds should produce different random values."""
        backend = NumPyBackend()
        arr1 = backend.random_uniform(0.0, 1.0, (10,), seed=111)
        arr2 = backend.random_uniform(0.0, 1.0, (10,), seed=222)
        assert not np.array_equal(arr1, arr2)

    def test_random_bool_shape(self):
        """random_bool should create array with correct shape."""
        backend = NumPyBackend()
        arr = backend.random_bool(0.5, (10,), seed=42)
        assert arr.shape == (10,)
        assert arr.dtype == np.bool_

    def test_random_bool_probability(self):
        """random_bool should respect probability parameter."""
        backend = NumPyBackend()
        # With p=0.0, all should be False
        arr_zero = backend.random_bool(0.0, (100,), seed=42)
        assert np.all(~arr_zero)
        # With p=1.0, all should be True
        arr_one = backend.random_bool(1.0, (100,), seed=42)
        assert np.all(arr_one)


class TestNumPyBackendMatrixOperations:
    """Tests for matrix operations."""

    def test_matmul_shape(self):
        """matmul should produce correct output shape."""
        backend = NumPyBackend()
        a = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
        b = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        result = backend.matmul(a, b)
        assert result.shape == (3, 3)

    def test_matmul_values(self):
        """matmul should compute correct matrix multiplication."""
        backend = NumPyBackend()
        a = np.array([[1, 2], [3, 4]], dtype=np.float64)
        b = np.array([[5, 6], [7, 8]], dtype=np.float64)
        result = backend.matmul(a, b)
        expected = np.array([[19, 22], [43, 50]], dtype=np.float64)
        assert np.array_equal(result, expected)

    def test_matmul_vector(self):
        """matmul should work with matrix-vector multiplication."""
        backend = NumPyBackend()
        a = np.array([[1, 2], [3, 4]], dtype=np.float64)
        v = np.array([1, 1], dtype=np.float64)
        result = backend.matmul(a, v)
        expected = np.array([3, 7], dtype=np.float64)
        assert np.array_equal(result, expected)

    def test_transpose(self):
        """transpose should swap axes."""
        backend = NumPyBackend()
        a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        result = backend.transpose(a)
        assert result.shape == (3, 2)
        assert result[0, 1] == 4
        assert result[2, 0] == 3


class TestNumPyBackendElementwiseOperations:
    """Tests for elementwise operations."""

    def test_where_condition_true(self):
        """where should select x where condition is True."""
        backend = NumPyBackend()
        condition = np.array([True, False, True])
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([10.0, 20.0, 30.0])
        result = backend.where(condition, x, y)
        expected = np.array([1.0, 20.0, 3.0])
        assert np.array_equal(result, expected)

    def test_where_all_false(self):
        """where should select y when all conditions are False."""
        backend = NumPyBackend()
        condition = np.array([False, False, False])
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([10.0, 20.0, 30.0])
        result = backend.where(condition, x, y)
        assert np.array_equal(result, y)

    def test_sqrt_values(self):
        """sqrt should compute element-wise square root."""
        backend = NumPyBackend()
        arr = np.array([4.0, 9.0, 16.0, 25.0])
        result = backend.sqrt(arr)
        expected = np.array([2.0, 3.0, 4.0, 5.0])
        assert np.allclose(result, expected)

    def test_maximum_elementwise(self):
        """maximum should compute element-wise maximum."""
        backend = NumPyBackend()
        a = np.array([1.0, 5.0, 3.0])
        b = np.array([2.0, 2.0, 4.0])
        result = backend.maximum(a, b)
        expected = np.array([2.0, 5.0, 4.0])
        assert np.array_equal(result, expected)

    def test_maximum_with_scalar(self):
        """maximum should work with scalar."""
        backend = NumPyBackend()
        a = np.array([-1.0, 2.0, -3.0])
        result = backend.maximum(a, 0.0)
        expected = np.array([0.0, 2.0, 0.0])
        assert np.array_equal(result, expected)


class TestNumPyBackendReductionOperations:
    """Tests for reduction operations."""

    def test_sum_all(self):
        """sum without axis should sum all elements."""
        backend = NumPyBackend()
        arr = np.array([[1, 2], [3, 4]])
        result = backend.sum(arr)
        assert result == 10

    def test_sum_axis_0(self):
        """sum with axis=0 should sum along rows."""
        backend = NumPyBackend()
        arr = np.array([[1, 2], [3, 4]])
        result = backend.sum(arr, axis=0)
        expected = np.array([4, 6])
        assert np.array_equal(result, expected)

    def test_sum_axis_1(self):
        """sum with axis=1 should sum along columns."""
        backend = NumPyBackend()
        arr = np.array([[1, 2], [3, 4]])
        result = backend.sum(arr, axis=1)
        expected = np.array([3, 7])
        assert np.array_equal(result, expected)

    def test_mean_all(self):
        """mean without axis should average all elements."""
        backend = NumPyBackend()
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = backend.mean(arr)
        assert result == 2.5

    def test_mean_axis(self):
        """mean with axis should average along that axis."""
        backend = NumPyBackend()
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = backend.mean(arr, axis=0)
        expected = np.array([2.0, 3.0])
        assert np.array_equal(result, expected)

    def test_any_true(self):
        """any should return True if any element is True."""
        backend = NumPyBackend()
        arr = np.array([False, True, False])
        assert backend.any(arr) is True

    def test_any_false(self):
        """any should return False if all elements are False."""
        backend = NumPyBackend()
        arr = np.array([False, False, False])
        assert backend.any(arr) is False


class TestNumPyBackendCopyOperations:
    """Tests for copy operations."""

    def test_copy_values(self):
        """copy should create array with same values."""
        backend = NumPyBackend()
        original = np.array([1.0, 2.0, 3.0])
        copied = backend.copy(original)
        assert np.array_equal(copied, original)

    def test_copy_independent(self):
        """copy should create independent array."""
        backend = NumPyBackend()
        original = np.array([1.0, 2.0, 3.0])
        copied = backend.copy(original)
        copied[0] = 999.0
        assert original[0] == 1.0  # Original unchanged

    def test_to_numpy_returns_numpy(self):
        """to_numpy should return a numpy array."""
        backend = NumPyBackend()
        arr = np.array([1.0, 2.0, 3.0])
        result = backend.to_numpy(arr)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, arr)


class TestGetBackend:
    """Tests for get_backend function."""

    def test_get_backend_default(self):
        """get_backend with no args should return NumPyBackend."""
        backend = get_backend()
        assert isinstance(backend, NumPyBackend)

    def test_get_backend_prefer_gpu_false(self):
        """get_backend with prefer_gpu=False should return NumPyBackend."""
        backend = get_backend(prefer_gpu=False)
        assert isinstance(backend, NumPyBackend)

    def test_get_backend_prefer_gpu_true_fallback(self):
        """get_backend with prefer_gpu=True should fallback to NumPy if JAX unavailable."""
        # This test passes if JAX is not installed - we get NumPyBackend
        # If JAX is installed, we'd get JAXBackend
        backend = get_backend(prefer_gpu=True)
        # Should return a valid backend either way
        assert isinstance(backend, ArrayBackend)


class TestBackendProtocol:
    """Tests verifying ArrayBackend protocol compliance."""

    def test_numpy_backend_has_required_methods(self):
        """NumPyBackend should have all required protocol methods."""
        backend = NumPyBackend()
        required_methods = [
            'zeros', 'ones', 'random_uniform', 'random_bool',
            'matmul', 'transpose', 'where', 'sqrt', 'maximum',
            'sum', 'mean', 'any', 'copy', 'to_numpy'
        ]
        for method in required_methods:
            assert hasattr(backend, method), f"Missing method: {method}"
            assert callable(getattr(backend, method)), f"Not callable: {method}"
