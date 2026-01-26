"""Property-based tests for Backend abstraction using Hypothesis.

RED phase: These tests should fail until backend is implemented.
"""

import numpy as np
from hypothesis import given, strategies as st, settings, assume

from src.core.backend import NumPyBackend, get_backend


# Custom strategies for backend operations
shape_strategy = st.tuples(
    st.integers(min_value=1, max_value=50),
    st.integers(min_value=1, max_value=50),
)
shape_1d_strategy = st.tuples(st.integers(min_value=1, max_value=100))
seed_strategy = st.integers(min_value=0, max_value=2**31 - 1)
float_strategy = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
positive_float_strategy = st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False)


class TestBackendZerosProperties:
    """Property tests for zeros operation."""

    @given(shape=shape_strategy)
    @settings(max_examples=50)
    def test_zeros_shape_invariant(self, shape):
        """zeros must produce array with exact requested shape."""
        backend = NumPyBackend()
        arr = backend.zeros(shape)
        assert arr.shape == shape

    @given(shape=shape_strategy)
    @settings(max_examples=50)
    def test_zeros_all_zero(self, shape):
        """zeros must produce array with all elements equal to zero."""
        backend = NumPyBackend()
        arr = backend.zeros(shape)
        assert np.all(arr == 0)

    @given(shape=shape_strategy)
    @settings(max_examples=50)
    def test_zeros_sum_is_zero(self, shape):
        """Sum of zeros array must be zero."""
        backend = NumPyBackend()
        arr = backend.zeros(shape)
        assert backend.sum(arr) == 0


class TestBackendOnesProperties:
    """Property tests for ones operation."""

    @given(shape=shape_strategy)
    @settings(max_examples=50)
    def test_ones_shape_invariant(self, shape):
        """ones must produce array with exact requested shape."""
        backend = NumPyBackend()
        arr = backend.ones(shape)
        assert arr.shape == shape

    @given(shape=shape_strategy)
    @settings(max_examples=50)
    def test_ones_sum_equals_size(self, shape):
        """Sum of ones array must equal total number of elements."""
        backend = NumPyBackend()
        arr = backend.ones(shape)
        expected_sum = shape[0] * shape[1]
        assert backend.sum(arr) == expected_sum


class TestBackendRandomProperties:
    """Property tests for random operations."""

    @given(
        low=float_strategy,
        high=float_strategy,
        shape=shape_1d_strategy,
        seed=seed_strategy,
    )
    @settings(max_examples=50)
    def test_random_uniform_bounds(self, low, high, shape, seed):
        """random_uniform values must be within [low, high)."""
        assume(low < high)  # Skip invalid cases
        assume(high - low > 1e-10)  # Skip degenerate ranges
        backend = NumPyBackend()
        arr = backend.random_uniform(low, high, shape, seed=seed)
        assert np.all(arr >= low)
        assert np.all(arr < high)

    @given(shape=shape_1d_strategy, seed=seed_strategy)
    @settings(max_examples=50)
    def test_random_uniform_reproducible(self, shape, seed):
        """Same seed must produce same random values."""
        backend = NumPyBackend()
        arr1 = backend.random_uniform(0.0, 1.0, shape, seed=seed)
        arr2 = backend.random_uniform(0.0, 1.0, shape, seed=seed)
        assert np.array_equal(arr1, arr2)

    @given(
        prob=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        seed=seed_strategy,
    )
    @settings(max_examples=50)
    def test_random_bool_dtype(self, prob, seed):
        """random_bool must produce boolean array."""
        backend = NumPyBackend()
        arr = backend.random_bool(prob, (100,), seed=seed)
        assert arr.dtype == np.bool_


class TestBackendMatrixProperties:
    """Property tests for matrix operations."""

    @given(
        n=st.integers(min_value=2, max_value=20),
        m=st.integers(min_value=2, max_value=20),
        k=st.integers(min_value=2, max_value=20),
        seed=seed_strategy,
    )
    @settings(max_examples=30)
    def test_matmul_shape_property(self, n, m, k, seed):
        """matmul of (n,m) @ (m,k) must produce (n,k)."""
        backend = NumPyBackend()
        a = backend.random_uniform(-1.0, 1.0, (n, m), seed=seed)
        b = backend.random_uniform(-1.0, 1.0, (m, k), seed=seed + 1)
        result = backend.matmul(a, b)
        assert result.shape == (n, k)

    @given(
        n=st.integers(min_value=2, max_value=20),
        m=st.integers(min_value=2, max_value=20),
        seed=seed_strategy,
    )
    @settings(max_examples=30)
    def test_transpose_involutory(self, n, m, seed):
        """Transpose of transpose must equal original."""
        backend = NumPyBackend()
        a = backend.random_uniform(-1.0, 1.0, (n, m), seed=seed)
        result = backend.transpose(backend.transpose(a))
        assert np.array_equal(result, a)

    @given(
        n=st.integers(min_value=2, max_value=20),
        m=st.integers(min_value=2, max_value=20),
        seed=seed_strategy,
    )
    @settings(max_examples=30)
    def test_transpose_shape_swap(self, n, m, seed):
        """Transpose must swap dimensions."""
        backend = NumPyBackend()
        a = backend.random_uniform(-1.0, 1.0, (n, m), seed=seed)
        result = backend.transpose(a)
        assert result.shape == (m, n)


class TestBackendElementwiseProperties:
    """Property tests for elementwise operations."""

    @given(shape=shape_1d_strategy, seed=seed_strategy)
    @settings(max_examples=50)
    def test_where_selects_correctly(self, shape, seed):
        """where must select x where True, y where False."""
        backend = NumPyBackend()
        condition = backend.random_bool(0.5, shape, seed=seed)
        x = backend.ones(shape)
        y = backend.zeros(shape)
        result = backend.where(condition, x, y)
        # Where True, should be 1; where False, should be 0
        assert np.array_equal(result, condition.astype(float))

    @given(shape=shape_1d_strategy, seed=seed_strategy)
    @settings(max_examples=50)
    def test_sqrt_of_squares(self, shape, seed):
        """sqrt(x^2) must equal |x| for non-negative x."""
        backend = NumPyBackend()
        arr = backend.random_uniform(0.0, 10.0, shape, seed=seed)
        squared = arr * arr
        result = backend.sqrt(squared)
        assert np.allclose(result, arr)

    @given(shape=shape_1d_strategy, seed=seed_strategy)
    @settings(max_examples=50)
    def test_maximum_commutative(self, shape, seed):
        """maximum(a, b) must equal maximum(b, a)."""
        backend = NumPyBackend()
        a = backend.random_uniform(-10.0, 10.0, shape, seed=seed)
        b = backend.random_uniform(-10.0, 10.0, shape, seed=seed + 1)
        result1 = backend.maximum(a, b)
        result2 = backend.maximum(b, a)
        assert np.array_equal(result1, result2)

    @given(shape=shape_1d_strategy, seed=seed_strategy)
    @settings(max_examples=50)
    def test_maximum_greater_or_equal(self, shape, seed):
        """maximum(a, b) must be >= a and >= b."""
        backend = NumPyBackend()
        a = backend.random_uniform(-10.0, 10.0, shape, seed=seed)
        b = backend.random_uniform(-10.0, 10.0, shape, seed=seed + 1)
        result = backend.maximum(a, b)
        assert np.all(result >= a)
        assert np.all(result >= b)


class TestBackendReductionProperties:
    """Property tests for reduction operations."""

    @given(shape=shape_strategy, seed=seed_strategy)
    @settings(max_examples=50)
    def test_sum_axis0_shape(self, shape, seed):
        """sum with axis=0 must reduce first dimension."""
        backend = NumPyBackend()
        arr = backend.random_uniform(0.0, 1.0, shape, seed=seed)
        result = backend.sum(arr, axis=0)
        assert result.shape == (shape[1],)

    @given(shape=shape_strategy, seed=seed_strategy)
    @settings(max_examples=50)
    def test_sum_axis1_shape(self, shape, seed):
        """sum with axis=1 must reduce second dimension."""
        backend = NumPyBackend()
        arr = backend.random_uniform(0.0, 1.0, shape, seed=seed)
        result = backend.sum(arr, axis=1)
        assert result.shape == (shape[0],)

    @given(shape=shape_strategy, seed=seed_strategy)
    @settings(max_examples=50)
    def test_mean_within_bounds(self, shape, seed):
        """mean of uniform[0,1] must be in (0, 1)."""
        backend = NumPyBackend()
        arr = backend.random_uniform(0.0, 1.0, shape, seed=seed)
        result = backend.mean(arr)
        assert 0.0 <= result <= 1.0


class TestBackendCopyProperties:
    """Property tests for copy operations."""

    @given(shape=shape_1d_strategy, seed=seed_strategy)
    @settings(max_examples=50)
    def test_copy_equals_original(self, shape, seed):
        """copy must produce array equal to original."""
        backend = NumPyBackend()
        original = backend.random_uniform(0.0, 1.0, shape, seed=seed)
        copied = backend.copy(original)
        assert np.array_equal(copied, original)

    @given(shape=shape_1d_strategy, seed=seed_strategy)
    @settings(max_examples=50)
    def test_copy_independent(self, shape, seed):
        """Modifying copy must not affect original."""
        backend = NumPyBackend()
        original = backend.random_uniform(0.0, 1.0, shape, seed=seed)
        original_value = original[0]
        copied = backend.copy(original)
        copied[0] = 999.0
        assert original[0] == original_value

    @given(shape=shape_1d_strategy, seed=seed_strategy)
    @settings(max_examples=50)
    def test_to_numpy_returns_numpy_array(self, shape, seed):
        """to_numpy must return numpy.ndarray."""
        backend = NumPyBackend()
        arr = backend.random_uniform(0.0, 1.0, shape, seed=seed)
        result = backend.to_numpy(arr)
        assert isinstance(result, np.ndarray)


class TestBackendEquivalenceProperties:
    """Property tests for backend equivalence (NumPy vs JAX)."""

    @given(
        n=st.integers(min_value=5, max_value=30),
        seed=seed_strategy,
    )
    @settings(max_examples=20)
    def test_simulation_step_deterministic(self, n, seed):
        """Same inputs must produce same outputs regardless of backend."""
        # This test verifies that the NumPy backend produces deterministic results
        # When JAX backend is implemented, we can compare both
        backend1 = get_backend()
        backend2 = get_backend()

        # Create identical random arrays
        arr1 = backend1.random_uniform(0.0, 1.0, (n, n), seed=seed)
        arr2 = backend2.random_uniform(0.0, 1.0, (n, n), seed=seed)

        assert np.array_equal(arr1, arr2)

        # Perform same operations
        result1 = backend1.matmul(arr1, backend1.transpose(arr1))
        result2 = backend2.matmul(arr2, backend2.transpose(arr2))

        assert np.allclose(result1, result2)
