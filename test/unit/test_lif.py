"""Unit tests for Leaky Integrate-and-Fire (LIF) neuron dynamics.

RED phase: These tests should fail until LIF dynamics are implemented.

LIF Model:
    V_i(t+1) = (1 - λ) * V_i(t) + input - V_reset * fired(t)
    
    - V_i: membrane potential of neuron i
    - λ: leak rate (potential decays toward 0)
    - V_reset: amount subtracted after firing
    - Fires when V_i >= threshold
"""

import numpy as np
import pytest

from src.core.neuron_state import NeuronState


class TestMembranePotentialInitialization:
    """Tests for membrane potential array initialization."""

    def test_membrane_potential_exists(self):
        """NeuronState should have membrane_potential attribute."""
        state = NeuronState.create(
            n_neurons=10,
            threshold=0.5,
            initial_firing_fraction=0.0,
            seed=42,
        )
        assert hasattr(state, "membrane_potential")

    def test_membrane_potential_shape(self):
        """Membrane potential should have shape (N,) for N neurons."""
        state = NeuronState.create(
            n_neurons=25,
            threshold=0.5,
            initial_firing_fraction=0.0,
            seed=42,
        )
        assert state.membrane_potential.shape == (25,)

    def test_membrane_potential_dtype_float(self):
        """Membrane potential should be float."""
        state = NeuronState.create(
            n_neurons=10,
            threshold=0.5,
            initial_firing_fraction=0.0,
            seed=42,
        )
        assert np.issubdtype(state.membrane_potential.dtype, np.floating)

    def test_membrane_potential_initially_zero(self):
        """Membrane potential should start at zero."""
        state = NeuronState.create(
            n_neurons=10,
            threshold=0.5,
            initial_firing_fraction=0.0,
            seed=42,
        )
        assert np.all(state.membrane_potential == 0.0)


class TestLIFLeakDynamics:
    """Tests for leak behavior in LIF model."""

    def test_potential_leaks_without_input(self):
        """Membrane potential should decay toward zero without input.
        
        V(t+1) = (1 - λ) * V(t) when input=0 and no firing
        """
        state = NeuronState.create(
            n_neurons=5,
            threshold=1.0,  # High threshold so no firing
            initial_firing_fraction=0.0,
            seed=42,
            leak_rate=0.1,
        )
        # Manually set initial potential
        state.membrane_potential[:] = 0.5
        
        # Update with zero input
        zero_input = np.zeros(5)
        state.update_firing(zero_input)
        
        # Potential should decay: 0.5 * (1 - 0.1) = 0.45
        expected = 0.5 * (1 - 0.1)
        np.testing.assert_allclose(state.membrane_potential, expected, rtol=1e-5)

    def test_leak_rate_zero_no_decay(self):
        """With leak_rate=0, potential should not decay."""
        state = NeuronState.create(
            n_neurons=5,
            threshold=1.0,
            initial_firing_fraction=0.0,
            seed=42,
            leak_rate=0.0,
        )
        state.membrane_potential[:] = 0.5
        
        zero_input = np.zeros(5)
        state.update_firing(zero_input)
        
        # No decay
        np.testing.assert_allclose(state.membrane_potential, 0.5, rtol=1e-5)


class TestLIFIntegration:
    """Tests for input integration in LIF model."""

    def test_potential_accumulates_with_input(self):
        """Membrane potential should accumulate input signal.
        
        V(t+1) = (1 - λ) * V(t) + input
        """
        state = NeuronState.create(
            n_neurons=3,
            threshold=1.0,
            initial_firing_fraction=0.0,
            seed=42,
            leak_rate=0.0,  # No leak for simple test
        )
        state.membrane_potential[:] = 0.0
        
        # Apply input signal
        input_signal = np.array([0.2, 0.3, 0.1])
        state.update_firing(input_signal)
        
        # Potential should equal input (started at 0, no leak)
        np.testing.assert_allclose(state.membrane_potential, input_signal, rtol=1e-5)

    def test_potential_accumulates_over_multiple_steps(self):
        """Potential should accumulate over multiple time steps."""
        state = NeuronState.create(
            n_neurons=1,
            threshold=1.0,
            initial_firing_fraction=0.0,
            seed=42,
            leak_rate=0.0,
        )
        
        input_signal = np.array([0.2])
        
        # Step 1
        state.update_firing(input_signal)
        assert state.membrane_potential[0] == pytest.approx(0.2)
        
        # Step 2
        state.update_firing(input_signal)
        assert state.membrane_potential[0] == pytest.approx(0.4)
        
        # Step 3
        state.update_firing(input_signal)
        assert state.membrane_potential[0] == pytest.approx(0.6)


class TestLIFFiring:
    """Tests for firing threshold in LIF model."""

    def test_fires_when_potential_exceeds_threshold(self):
        """Neuron should fire when membrane potential >= threshold."""
        state = NeuronState.create(
            n_neurons=3,
            threshold=0.5,
            initial_firing_fraction=0.0,
            seed=42,
            leak_rate=0.0,
        )
        state.membrane_potential[:] = [0.6, 0.4, 0.5]  # Above, below, exactly at
        
        # Need input to trigger update
        zero_input = np.zeros(3)
        state.update_firing(zero_input)
        
        # Neurons 0 and 2 should fire (>= threshold)
        assert state.firing[0] == True  # 0.6 >= 0.5
        assert state.firing[1] == False  # 0.4 < 0.5
        assert state.firing[2] == True  # 0.5 >= 0.5

    def test_potential_accumulates_to_threshold(self):
        """Neuron should fire once potential accumulates to threshold."""
        state = NeuronState.create(
            n_neurons=1,
            threshold=0.5,
            initial_firing_fraction=0.0,
            seed=42,
            leak_rate=0.0,
        )
        
        input_signal = np.array([0.2])
        
        # Step 1: potential = 0.2, below threshold
        state.update_firing(input_signal)
        assert state.firing[0] == False
        assert state.membrane_potential[0] == pytest.approx(0.2)
        
        # Step 2: potential = 0.4, still below
        state.update_firing(input_signal)
        assert state.firing[0] == False
        assert state.membrane_potential[0] == pytest.approx(0.4)
        
        # Step 3: potential = 0.6, fires!
        state.update_firing(input_signal)
        assert state.firing[0] == True


class TestLIFReset:
    """Tests for potential reset after firing."""

    def test_potential_resets_after_firing(self):
        """Membrane potential should decrease by reset_potential after firing."""
        state = NeuronState.create(
            n_neurons=1,
            threshold=0.5,
            initial_firing_fraction=0.0,
            seed=42,
            leak_rate=0.0,
            reset_potential=0.5,
        )
        state.membrane_potential[:] = [0.7]
        
        zero_input = np.zeros(1)
        state.update_firing(zero_input)
        
        # Fires (0.7 >= 0.5), then resets: 0.7 - 0.5 = 0.2
        assert state.firing[0] == True
        assert state.membrane_potential[0] == pytest.approx(0.2)

    def test_reset_prevents_continuous_firing(self):
        """After reset, neuron should not fire until potential rebuilds."""
        state = NeuronState.create(
            n_neurons=1,
            threshold=0.5,
            initial_firing_fraction=0.0,
            seed=42,
            leak_rate=0.0,
            reset_potential=1.0,  # Full reset
        )
        
        # Build up to firing
        state.membrane_potential[:] = [0.6]
        zero_input = np.zeros(1)
        state.update_firing(zero_input)
        
        # Should fire
        assert state.firing[0] == True
        
        # After reset, potential should be low
        # 0.6 - 1.0 = -0.4, but clamped to 0
        assert state.membrane_potential[0] >= 0  # No negative potential
        assert state.membrane_potential[0] < 0.5  # Below threshold
        
        # Next step: should NOT fire
        state.update_firing(zero_input)
        assert state.firing[0] == False

    def test_non_firing_neurons_no_reset(self):
        """Neurons that don't fire should not have potential reset."""
        state = NeuronState.create(
            n_neurons=2,
            threshold=0.5,
            initial_firing_fraction=0.0,
            seed=42,
            leak_rate=0.0,
            reset_potential=1.0,
        )
        state.membrane_potential[:] = [0.3, 0.7]  # Below, above threshold
        
        zero_input = np.zeros(2)
        state.update_firing(zero_input)
        
        # Neuron 0: didn't fire, potential unchanged
        assert state.firing[0] == False
        assert state.membrane_potential[0] == pytest.approx(0.3)
        
        # Neuron 1: fired, potential reset
        assert state.firing[1] == True


class TestLIFConfigParameters:
    """Tests for LIF configuration parameters."""

    def test_leak_rate_parameter_validation(self):
        """leak_rate should be in [0, 1]."""
        with pytest.raises(ValueError, match="leak_rate"):
            NeuronState.create(
                n_neurons=10,
                threshold=0.5,
                initial_firing_fraction=0.0,
                leak_rate=-0.1,  # Invalid
            )
        
        with pytest.raises(ValueError, match="leak_rate"):
            NeuronState.create(
                n_neurons=10,
                threshold=0.5,
                initial_firing_fraction=0.0,
                leak_rate=1.5,  # Invalid
            )

    def test_reset_potential_parameter_validation(self):
        """reset_potential should be >= 0."""
        with pytest.raises(ValueError, match="reset_potential"):
            NeuronState.create(
                n_neurons=10,
                threshold=0.5,
                initial_firing_fraction=0.0,
                reset_potential=-0.5,  # Invalid
            )


class TestLIFFullDynamics:
    """Integration tests for complete LIF dynamics."""

    def test_full_lif_dynamics(self):
        """Test complete LIF: leak + integrate + fire + reset."""
        state = NeuronState.create(
            n_neurons=1,
            threshold=1.0,
            initial_firing_fraction=0.0,
            seed=42,
            leak_rate=0.1,
            reset_potential=1.0,
        )
        
        input_signal = np.array([0.3])
        
        # Step 1: V = 0 * 0.9 + 0.3 = 0.3
        state.update_firing(input_signal)
        assert state.membrane_potential[0] == pytest.approx(0.3)
        assert state.firing[0] == False
        
        # Step 2: V = 0.3 * 0.9 + 0.3 = 0.57
        state.update_firing(input_signal)
        assert state.membrane_potential[0] == pytest.approx(0.57)
        assert state.firing[0] == False
        
        # Step 3: V = 0.57 * 0.9 + 0.3 = 0.813
        state.update_firing(input_signal)
        assert state.membrane_potential[0] == pytest.approx(0.813)
        assert state.firing[0] == False
        
        # Step 4: V = 0.813 * 0.9 + 0.3 = 1.0317 -> fires!
        state.update_firing(input_signal)
        assert state.firing[0] == True
        # After reset: 1.0317 - 1.0 = 0.0317
        assert state.membrane_potential[0] == pytest.approx(0.0317, rel=0.01)
        
        # Step 5: Not firing, rebuilding
        state.update_firing(input_signal)
        assert state.firing[0] == False
