"""
Comprehensive test suite for algorithm.py module.

This test suite covers all high-level optimization functions in the algorithm.py module:
- fidelity_maximization()
- basis_transformation_with_polar_decomposition()
- basis_transformation_with_gradient_descent()
- energy_minimization_with_polar_decomposition()
- energy_minimization_with_gradient_descent()
- get_fidelity_gradient()
- polar_opt()
- var_gate_exact()
- var_gate_exact_list_of_states()

The tests ensure proper functionality and include finite difference testing
for gradient-related functions.
"""

import pytest
import numpy as np
import scipy.sparse
from stateprep.algorithm import (
    fidelity_maximization,
    basis_transformation_with_polar_decomposition,
    basis_transformation_with_gradient_descent,
    energy_minimization_with_polar_decomposition,
    energy_minimization_with_gradient_descent,
    get_fidelity_gradient,
    polar_opt,
    var_gate_exact,
    var_gate_exact_list_of_states
)
from stateprep.circuit import FermionicCircuit, QubitCircuit
from stateprep.gate import UnitaryGate, U1UnitaryGate
from stateprep.exact_sim import StateVector
from stateprep.utils.common_setup import pauli_to_sparse_op


# --- Define test setups ---
test_setups_1 = [
    (
        QubitCircuit([
            ((0, 1), U1UnitaryGate())
        ]),
        np.array([1, 0, 0, 0], dtype=complex),
        np.array([0, 1, 0, 0], dtype=complex)
    ),
    (
        QubitCircuit([
            ((0, 1), U1UnitaryGate()),
            ((2, 3), U1UnitaryGate()),
            ((0, 2), U1UnitaryGate()),
            ((1, 3), U1UnitaryGate()),
        ]),
        np.array([
            0.0, 0.0, 0.0, -np.sqrt(1/2.)/2,
            0.0, -0.5, -np.sqrt(1/2.)/2, 0.0,
            -0.0, -np.sqrt(1/2.)/2, -0.5, 0.0,
            -np.sqrt(1/2.)/2, 0.0, 0.0, 0.0
        ], dtype=complex),
        np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=complex)
    ),
    (
        FermionicCircuit([
            ((0, 1), U1UnitaryGate()),
            ((2, 3), U1UnitaryGate()),
            ((0, 2), U1UnitaryGate()),
            ((1, 3), U1UnitaryGate()),
        ]).export_QubitCircuit(),
        np.array([
            0.0, 0.0, 0.0, -np.sqrt(1/2.)/2,
            0.0, -0.5, -np.sqrt(1/2.)/2, 0.0,
            -0.0, -np.sqrt(1/2.)/2, -0.5, 0.0,
            -np.sqrt(1/2.)/2, 0.0, 0.0, 0.0
        ], dtype=complex),
        np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=complex)
    ),
]

test_setups_2 = [
    (
        QubitCircuit([
            ((0, 1), U1UnitaryGate())
        ]),
        [np.array([1, 0, 0, 0], dtype=complex),  # |00⟩
         np.array([0, 0, 1, 0], dtype=complex)   # |10)
         ],
        [np.array([0, 1, 0, 0], dtype=complex),  # |01⟩
         np.array([0, 0, 0, 1], dtype=complex)   # |11⟩
        ],
    ),
    (
        QubitCircuit([
            ((0, 1), U1UnitaryGate()),
            ((2, 3), U1UnitaryGate()),
            ((0, 2), U1UnitaryGate()),
            ((1, 3), U1UnitaryGate()),
        ]),
        [np.array([
            0.0, 0.0, 0.0, -np.sqrt(1/2.)/2,
            0.0, -0.5, -np.sqrt(1/2.)/2, 0.0,
            -0.0, -np.sqrt(1/2.)/2, -0.5, 0.0,
            -np.sqrt(1/2.)/2, 0.0, 0.0, 0.0
        ], dtype=complex),],
        [np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=complex),],
    ),
    (
        FermionicCircuit([
            ((0, 1), U1UnitaryGate()),
            ((2, 3), U1UnitaryGate()),
            ((0, 2), U1UnitaryGate()),
            ((1, 3), U1UnitaryGate()),
        ]).export_QubitCircuit(),
        [np.array([
            0.0, 0.0, 0.0, -np.sqrt(1/2.)/2,
            0.0, -0.5, -np.sqrt(1/2.)/2, 0.0,
            -0.0, -np.sqrt(1/2.)/2, -0.5, 0.0,
            -np.sqrt(1/2.)/2, 0.0, 0.0, 0.0
        ], dtype=complex),],
        [np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=complex),
         ],
    ),
]

test_setups_energy = [
        (
            QubitCircuit([
                ((0, 1), UnitaryGate()),
                ((2, 3), UnitaryGate()),
                ((0, 2), UnitaryGate()),
                ((1, 3), UnitaryGate())
            ]),
            pauli_to_sparse_op('IIXX') + 3 * pauli_to_sparse_op('IYYI') - \
            2 * pauli_to_sparse_op('ZZII'),
        ),
        (
            FermionicCircuit([
                ((0, 1), UnitaryGate()),
                ((2, 3), UnitaryGate()),
                ((0, 2), UnitaryGate()),
                ((1, 3), UnitaryGate())
            ]).export_QubitCircuit(),
            pauli_to_sparse_op('IIXX') + 3 * pauli_to_sparse_op('IYYI') - \
            2 * pauli_to_sparse_op('ZZII'),
        ),
        (
            QubitCircuit([
                ((0, 1), U1UnitaryGate()),
                ]),
            scipy.sparse.csc_matrix(np.diag([1, -1, 1, -1])),
        ),
]

@pytest.mark.parametrize("circuit,target_state,initial_state", test_setups_1)
class TestFidelityMaximization:
    """Test cases for fidelity maximization functions."""

    def test_fidelity_maximization_polar_decomposition(self, circuit, target_state, initial_state):
        """Test fidelity maximization with polar decomposition method."""
        optimized_circuit, info = fidelity_maximization(
            circuit,
            target_state,
            initial_state,
            method='polar_decomposition',
            num_steps=10,
            verbose=False
        )

        assert isinstance(optimized_circuit, QubitCircuit)
        assert isinstance(info, dict)
        assert 'converged' in info
        assert 'cost' in info
        assert 'num_steps' in info
        assert info['cost'] >= -1e-10  # Allow for small numerical errors

    def test_fidelity_maximization_gradient_descent(self, circuit, target_state, initial_state):
        """Test fidelity maximization with gradient descent method."""
        optimized_circuit, info = fidelity_maximization(
            circuit,
            target_state,
            initial_state,
            method='gradient_descent',
            num_steps=10,
            verbose=False
        )

        assert isinstance(optimized_circuit, QubitCircuit)
        assert hasattr(info, 'x')  # scipy optimize result object
        assert hasattr(info, 'fun')  # final cost

    def test_fidelity_maximization_invalid_method(self, circuit, target_state, initial_state):
        """Test error handling for invalid optimization method."""
        with pytest.raises(ValueError, match="Unknown method"):
            fidelity_maximization(
                circuit,
                target_state,
                initial_state,
                method='invalid_method'
            )


@pytest.mark.parametrize("circuit,target_states,initial_states", test_setups_2)
class TestBasisTransformation:
    """Test cases for basis transformation functions."""

    def test_basis_transformation_polar_decomposition(self, circuit, target_states, initial_states):
        """Test basis transformation with polar decomposition."""
        optimized_circuit, info = basis_transformation_with_polar_decomposition(
            circuit,
            target_states,
            initial_states,
            num_steps=5,
            verbose=False
        )

        assert isinstance(optimized_circuit, QubitCircuit)
        assert isinstance(info, dict)
        assert 'converged' in info
        assert 'cost' in info
        assert 'num_steps' in info
        assert info['cost'] >= -1e-10  # Allow for small numerical errors
        assert optimized_circuit.num_gates == circuit.num_gates

    def test_basis_transformation_gradient_descent(self, circuit, target_states, initial_states):
        """Test basis transformation with gradient descent."""
        optimized_circuit, result = basis_transformation_with_gradient_descent(
            circuit,
            target_states,
            initial_states,
            num_steps=5,
            verbose=False
        )

        assert isinstance(optimized_circuit, QubitCircuit)
        assert hasattr(result, 'x')  # optimization result
        assert hasattr(result, 'fun')  # final cost
        assert optimized_circuit.num_gates == circuit.num_gates


@pytest.mark.parametrize("circuit,H", test_setups_energy)
class TestEnergyMinimization:
    """Test cases for energy minimization functions."""

    def test_energy_minimization_polar_decomposition_not_implemented(self, circuit, H):
        """Test that polar decomposition energy minimization raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            energy_minimization_with_polar_decomposition(
                circuit,
                H,
                verbose=False
            )

    def test_energy_minimization_gradient_descent(self, circuit, H):
        """Test energy minimization with gradient descent."""
        optimized_circuit = energy_minimization_with_gradient_descent(
            circuit,
            H,
            init_vec=None,
            method='L-BFGS-B',
            verbose=False
        )

        assert isinstance(optimized_circuit, QubitCircuit)
        assert optimized_circuit.num_gates == circuit.num_gates

        # Check that energy was actually computed
        initial_energy = circuit.get_energy(H, None)
        final_energy = optimized_circuit.get_energy(H, None)
        assert isinstance(initial_energy, (complex, float))
        assert isinstance(final_energy, (complex, float))


@pytest.mark.parametrize("circuit,target_states,initial_states", test_setups_2)
class TestGradientFunctions:
    """Test cases for gradient computation functions."""

    def test_get_fidelity_gradient_basic(self, circuit, target_states, initial_states):
        """Test basic functionality of fidelity gradient computation."""
        cost, grads = get_fidelity_gradient(
            circuit,
            target_states,
            initial_states,
            verbose=False
        )

        assert isinstance(cost, float)
        assert cost >= -1e-10  # Allow for small numerical errors
        assert isinstance(grads, list)
        assert len(grads) == circuit.num_trainable_gates

        # Check gradient shapes match parameter shapes
        trainable_idx = 0
        for idx, pair in enumerate(circuit.pairs_of_indices_and_Us):
            if  circuit.trainable[idx]:
                gate_params = pair[1].params
                assert grads[trainable_idx].shape == gate_params.shape
                trainable_idx += 1

        assert trainable_idx == len(grads)  # All trainable gates should have gradients

    def test_get_fidelity_gradient_finite_difference(self, circuit, target_states, initial_states):
        """Test fidelity gradient computation using finite difference validation."""
        eps = 1e-6

        # Get analytical gradients
        cost, grads = get_fidelity_gradient(
            circuit,
            target_states,
            initial_states,
            verbose=False
        )

        # Compute finite difference gradients
        original_params = circuit.get_concatenated_params()

        finite_diff_grads = []
        for param_idx in range(len(original_params)):
            # Forward difference
            params_plus = original_params.copy()
            params_plus[param_idx] += eps
            circuit.set_params(params_plus)
            cost_plus, _ = get_fidelity_gradient(
                circuit,
                target_states,
                initial_states,
                verbose=False
            )

            # Backward difference
            params_minus = original_params.copy()
            params_minus[param_idx] -= eps
            circuit.set_params(params_minus)
            cost_minus, _ = get_fidelity_gradient(
                circuit,
                target_states,
                initial_states,
                verbose=False
            )

            finite_diff_grad = (cost_plus - cost_minus) / (2 * eps)
            finite_diff_grads.append(finite_diff_grad)

        # Restore original parameters
        circuit.set_params(original_params)

        # Compare analytical and finite difference gradients
        analytical_grads_flat = np.concatenate([grad.flatten() for grad in grads])
        finite_diff_grads = np.array(finite_diff_grads)

        # Allow for some numerical tolerance
        np.testing.assert_allclose(
            analytical_grads_flat,
            finite_diff_grads,
            rtol=1e-4,
            atol=1e-6,
            err_msg="Analytical and finite difference gradients don't match"
        )

    def test_get_fidelity_gradient_multiple_states(self, circuit, target_states, initial_states):
        """Test fidelity gradient with multiple target/initial state pairs."""
        cost, grads = get_fidelity_gradient(
            circuit,
            target_states,
            initial_states,
            verbose=False
        )

        assert isinstance(cost, float)
        assert cost >= -1e-10  # Allow for small numerical errors
        assert isinstance(grads, list)
        assert len(grads) == circuit.num_trainable_gates


class TestPolarOptimization:
    """Test cases for polar optimization functions."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a simple 2-qubit circuit
        self.num_qubits = 2
        pairs = [((0, 1), U1UnitaryGate())]
        self.circuit = QubitCircuit(pairs)

        # Create target and initial states
        self.target_states = [np.array([1, 0, 0, 0], dtype=complex)]  # |00⟩
        self.initial_states = [np.array([0, 1, 0, 0], dtype=complex)]  # |01⟩

    def test_polar_opt_basic(self):
        """Test basic polar optimization functionality."""
        cost, bottom_states = polar_opt(
            self.circuit,
            self.target_states,
            self.initial_states,
            verbose=False
        )

        assert isinstance(cost, float)
        assert cost >= -1e-10  # Allow for small numerical errors
        assert isinstance(bottom_states, list)
        assert len(bottom_states) == len(self.initial_states)
        assert all(isinstance(state, StateVector) for state in bottom_states)

    def test_polar_opt_with_provided_bottom_states(self):
        """Test polar optimization with pre-computed bottom states."""
        # First run to get bottom states
        _, bottom_states = polar_opt(
            self.circuit,
            self.target_states,
            self.initial_states,
            verbose=False
        )

        # Second run with provided bottom states
        cost, new_bottom_states = polar_opt(
            self.circuit,
            self.target_states,
            self.initial_states,
            list_of_bottom_states=bottom_states,
            verbose=False
        )

        assert isinstance(cost, float)
        assert cost >= -1e-10  # Allow for small numerical errors
        assert len(new_bottom_states) == len(bottom_states)


class TestVariationalGates:
    """Test cases for variational gate optimization functions."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create random state vectors
        np.random.seed(42)  # For reproducibility
        self.top_state = StateVector(np.random.randn(4) + 1j * np.random.randn(4))
        self.bottom_state = StateVector(np.random.randn(4) + 1j * np.random.randn(4))

        # Normalize
        self.top_state.state_vector /= np.linalg.norm(self.top_state.state_vector)
        self.bottom_state.state_vector /= np.linalg.norm(self.bottom_state.state_vector)

        self.indices = (0, 1)

    def test_var_gate_exact(self):
        """Test exact variational gate optimization for single state pair."""
        gate = var_gate_exact(self.top_state, self.indices, self.bottom_state)

        # Check gate properties
        assert gate.shape == (2, 2, 2, 2)
        assert np.allclose(np.abs(np.linalg.det(gate.reshape(4, 4))), 1.0, atol=1e-10)

        # Gate should be unitary (U @ U^\dagger = I)
        gate_matrix = gate.reshape(4, 4)
        should_be_identity = gate_matrix @ gate_matrix.conj().T
        np.testing.assert_allclose(should_be_identity, np.eye(4), atol=1e-10)

    def test_var_gate_exact_list_of_states(self):
        """Test exact variational gate optimization for multiple state pairs."""
        # Create another state pair
        top_state2 = StateVector(np.random.randn(4) + 1j * np.random.randn(4))
        bottom_state2 = StateVector(np.random.randn(4) + 1j * np.random.randn(4))
        top_state2.state_vector /= np.linalg.norm(top_state2.state_vector)
        bottom_state2.state_vector /= np.linalg.norm(bottom_state2.state_vector)

        top_states = [self.top_state, top_state2]
        bottom_states = [self.bottom_state, bottom_state2]

        gate = var_gate_exact_list_of_states(top_states, self.indices, bottom_states)

        # Check gate properties
        assert gate.shape == (2, 2, 2, 2)

        # Gate should be unitary
        gate_matrix = gate.reshape(4, 4)
        should_be_identity = gate_matrix @ gate_matrix.conj().T
        np.testing.assert_allclose(should_be_identity, np.eye(4), atol=1e-10)

    def test_var_gate_exact_swapped_indices(self):
        """Test variational gate optimization with swapped indices."""
        # Test with indices in different order
        swapped_indices = (1, 0)
        gate = var_gate_exact(self.top_state, swapped_indices, self.bottom_state)

        assert gate.shape == (2, 2, 2, 2)

        # Gate should still be unitary
        gate_matrix = gate.reshape(4, 4)
        should_be_identity = gate_matrix @ gate_matrix.conj().T
        np.testing.assert_allclose(should_be_identity, np.eye(4), atol=1e-10)


@pytest.mark.parametrize("circuit,H", test_setups_energy)
class TestEnergyGradientFiniteDifference:
    """Test cases for energy gradient finite difference validation."""

    def test_energy_gradient_finite_difference(self, circuit, H):
        """Test energy gradient computation using finite difference validation."""
        eps = 1e-6

        # Get analytical gradients
        analytical_grads = circuit.get_energy_gradient(H, None)

        # Compute finite difference gradients
        original_params = circuit.get_concatenated_params()

        finite_diff_grads = []
        for param_idx in range(len(original_params)):
            # Forward difference
            params_plus = original_params.copy()
            params_plus[param_idx] += eps
            circuit.set_params(params_plus)
            energy_plus = circuit.get_energy(H, None)

            # Backward difference
            params_minus = original_params.copy()
            params_minus[param_idx] -= eps
            circuit.set_params(params_minus)
            energy_minus = circuit.get_energy(H, None)

            finite_diff_grad = (energy_plus - energy_minus) / (2 * eps)
            finite_diff_grads.append(finite_diff_grad)

        # Restore original parameters
        circuit.set_params(original_params)

        # Compare analytical and finite difference gradients
        analytical_grads_flat = np.concatenate([grad.flatten() for grad in analytical_grads])
        finite_diff_grads = np.array(finite_diff_grads)

        # Allow for some numerical tolerance
        np.testing.assert_allclose(
            np.real(analytical_grads_flat),
            np.real(finite_diff_grads),
            rtol=1e-4,
            atol=1e-6,
            err_msg="Real parts of analytical and finite difference gradients don't match"
        )

        np.testing.assert_allclose(
            np.imag(analytical_grads_flat),
            np.imag(finite_diff_grads),
            rtol=1e-4,
            atol=1e-6,
            err_msg="Imaginary parts of analytical and finite difference gradients don't match"
        )


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_states_lists(self):
        """Test behavior with empty state lists."""
        circuit = QubitCircuit([((0, 1), U1UnitaryGate())])

        # Empty lists should return zero cost and zero gradients
        cost, grads = get_fidelity_gradient(circuit, [], [], verbose=False)
        assert cost == 0
        assert len(grads) == circuit.num_trainable_gates
        # All gradients should be zero for empty input
        for grad in grads:
            np.testing.assert_allclose(grad, np.zeros_like(grad))

    def test_mismatched_state_list_lengths(self):
        """Test behavior with mismatched target/initial state list lengths."""
        circuit = QubitCircuit([((0, 1), U1UnitaryGate())])
        target_states = [np.array([1, 0, 0, 0], dtype=complex)]
        initial_states = [
            np.array([0, 1, 0, 0], dtype=complex),
            np.array([0, 0, 1, 0], dtype=complex)
        ]

        # With mismatched lengths, it will process only up to the shorter list
        cost, grads = get_fidelity_gradient(circuit, target_states, initial_states, verbose=False)
        assert isinstance(cost, float)
        assert len(grads) == circuit.num_trainable_gates

    def test_var_gate_exact_identical_indices(self):
        """Test error handling for identical indices in variational gate optimization."""
        top_state = StateVector(np.array([1, 0, 0, 0], dtype=complex))
        bottom_state = StateVector(np.array([0, 1, 0, 0], dtype=complex))

        # Should raise assertion error for identical indices
        with pytest.raises(AssertionError):
            var_gate_exact(top_state, (0, 0), bottom_state)


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_optimization_workflow(self):
        """Test a complete optimization workflow."""
        # Create a more complex circuit
        num_qubits = 2
        pairs = [
            ((0, 1), U1UnitaryGate()),
            ((0, 1), U1UnitaryGate())
        ]
        circuit = QubitCircuit(pairs)

        # Define optimization target
        target_state = np.array([0, 0, 0, 1], dtype=complex)  # |11⟩
        initial_state = np.array([1, 0, 0, 0], dtype=complex)  # |00⟩

        # Test polar decomposition method
        optimized_circuit_polar, info_polar = fidelity_maximization(
            circuit.copy(),
            target_state,
            initial_state,
            method='polar_decomposition',
            num_steps=5
        )

        # Test gradient descent method
        optimized_circuit_grad, info_grad = fidelity_maximization(
            circuit.copy(),
            target_state,
            initial_state,
            method='gradient_descent',
            num_steps=5
        )

        # Both methods should produce valid results
        assert isinstance(optimized_circuit_polar, QubitCircuit)
        assert isinstance(optimized_circuit_grad, QubitCircuit)
        assert info_polar['cost'] >= -1e-10  # Allow for small numerical errors
        assert hasattr(info_grad, 'fun')

        # The optimized circuits should be different from the original
        original_output = circuit.to_state_vector(initial_state)
        polar_output = optimized_circuit_polar.to_state_vector(initial_state)
        grad_output = optimized_circuit_grad.to_state_vector(initial_state)

        # At least one should be different (unless by coincidence they're already optimal)
        assert (not np.allclose(original_output, polar_output, atol=1e-10) or
                not np.allclose(original_output, grad_output, atol=1e-10) or
                info_polar['cost'] < 1e-6)  # Unless already converged
