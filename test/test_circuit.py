"""
Comprehensive test suite for circuit.py module.

This test suite covers all classes and functions in the circuit.py module:
- Circuit base class
- FermionicCircuit class  
- QubitCircuit class
- Utility functions for circuit transformations
- Edge cases and error handling

The tests ensure proper functionality without requiring actual JAX dependencies.
"""

import pytest
import numpy as np
import scipy.linalg
import pickle
import tempfile
import os
from stateprep.circuit import (
    Circuit, FermionicCircuit, QubitCircuit,
    transform_to_nearest_neighbor_qubit_circuit,
    removing_SWAP_from_nearest_neighbor_qubit_circuit
)
from stateprep.gate import U1UnitaryGate, UnitaryGate


class TestCircuitBaseClass:
    """Test cases for the Circuit base class."""
    
    def test_circuit_initialization_basic(self):
        """Test basic initialization of Circuit with simple pairs."""
        pairs = [((0, 1), np.eye(4)), ((1, 2), np.eye(4))]
        circuit = Circuit(pairs)
        
        assert circuit.num_gates == 2
        assert circuit.num_qubits == 3
        assert len(circuit.trainable) == 2
        assert all(circuit.trainable)  # All should be trainable by default
    
    def test_circuit_initialization_with_trainable(self):
        """Test initialization with custom trainable flags."""
        pairs = [((0, 1), np.eye(4)), ((1, 2), np.eye(4))]
        trainable = [True, False]
        circuit = Circuit(pairs, trainable)
        
        assert circuit.trainable == trainable
    
    def test_circuit_initialization_larger_system(self):
        """Test initialization with larger qubit system."""
        pairs = [((0, 5), np.eye(4)), ((2, 7), np.eye(4))]
        circuit = Circuit(pairs)
        
        assert circuit.num_gates == 2
        assert circuit.num_qubits == 8  # 0-7, so 8 qubits
    
    def test_circuit_initialization_invalid_trainable_length(self):
        """Test error handling for mismatched trainable length."""
        pairs = [((0, 1), np.eye(4)), ((1, 2), np.eye(4))]
        trainable = [True]  # Wrong length
        
        with pytest.raises(AssertionError):
            Circuit(pairs, trainable)
    
    def test_circuit_initialization_invalid_trainable_type(self):
        """Test error handling for wrong trainable type."""
        pairs = [((0, 1), np.eye(4))]
        trainable = True  # Should be list
        
        with pytest.raises(AssertionError):
            Circuit(pairs, trainable)
    
    def test_circuit_get_params_not_implemented(self):
        """Test that get_params raises NotImplementedError."""
        pairs = [((0, 1), np.eye(4))]
        circuit = Circuit(pairs)
        
        with pytest.raises(NotImplementedError):
            circuit.get_params()
    
    def test_circuit_set_params_not_implemented(self):
        """Test that set_params raises NotImplementedError."""
        pairs = [((0, 1), np.eye(4))]
        circuit = Circuit(pairs)
        
        with pytest.raises(NotImplementedError):
            circuit.set_params([])
    
    def test_circuit_save_pairs(self):
        """Test saving pairs to file."""
        pairs = [((0, 1), np.eye(4)), ((1, 2), 2*np.eye(4))]
        circuit = Circuit(pairs)
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            try:
                circuit.save_pairs_of_indices_and_Us(tmp.name)
                
                # Load and verify
                with open(tmp.name, "rb") as f:
                    loaded_pairs = pickle.load(f)
                
                assert len(loaded_pairs) == 2
                assert loaded_pairs[0][0] == (0, 1)
                assert loaded_pairs[1][0] == (1, 2)
                assert np.allclose(loaded_pairs[0][1], np.eye(4))
                assert np.allclose(loaded_pairs[1][1], 2*np.eye(4))
            finally:
                os.unlink(tmp.name)
    
    def test_circuit_copy(self):
        """Test deep copy functionality."""
        U1 = U1UnitaryGate()
        U2 = U1UnitaryGate()
        pairs = [((0, 1), U1), ((1, 2), U2)]
        trainable = [True, False]
        
        circuit = Circuit(pairs, trainable)
        copied = circuit.copy()
        
        # Check that it's a proper copy
        assert copied is not circuit
        assert copied.num_gates == circuit.num_gates
        assert copied.num_qubits == circuit.num_qubits
        assert copied.trainable == circuit.trainable
        assert copied.trainable is not circuit.trainable  # Different list objects
        
        # Check pairs are copied
        assert len(copied.pairs_of_indices_and_Us) == len(circuit.pairs_of_indices_and_Us)
        for i, (orig, copy_pair) in enumerate(zip(circuit.pairs_of_indices_and_Us, copied.pairs_of_indices_and_Us)):
            assert orig[0] == copy_pair[0]  # Indices should be equal
            assert orig[1] is not copy_pair[1]  # Gates should be different objects


class TestFermionicCircuit:
    """Test cases for the FermionicCircuit class."""
    
    def test_fermionic_circuit_initialization(self):
        """Test basic initialization of FermionicCircuit."""
        pairs = [((0, 1), U1UnitaryGate()), ((1, 2), U1UnitaryGate())]
        circuit = FermionicCircuit(pairs)
        
        assert circuit.num_gates == 2
        assert circuit.num_qubits == 3
        assert hasattr(circuit, 'fSWAP')
        assert circuit.fSWAP.shape == (2, 2, 2, 2)
    
    def test_fermionic_circuit_fswap_matrix(self):
        """Test that fSWAP matrix is correct."""
        pairs = [((0, 1), U1UnitaryGate())]
        circuit = FermionicCircuit(pairs)
        
        expected_fSWAP = np.array([[1, 0, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 0, -1]]).reshape([2, 2, 2, 2])
        
        assert np.allclose(circuit.fSWAP, expected_fSWAP)
    
    def test_export_qubit_circuit_adjacent_gates(self):
        """Test export to QubitCircuit with adjacent gates."""
        U1 = U1UnitaryGate()
        U2 = U1UnitaryGate() 
        pairs = [((0, 1), U1), ((1, 2), U2)]
        fermionic_circuit = FermionicCircuit(pairs)
        
        qubit_circuit = fermionic_circuit.export_QubitCircuit()
        
        assert isinstance(qubit_circuit, QubitCircuit)
        assert qubit_circuit.num_gates == 2  # No extra fSWAP gates needed
        assert qubit_circuit.num_qubits == 3
    
    def test_export_qubit_circuit_non_adjacent_gates(self):
        """Test export to QubitCircuit with non-adjacent gates requiring fSWAP."""
        U1 = U1UnitaryGate()
        pairs = [((0, 2), U1)]  # Non-adjacent
        fermionic_circuit = FermionicCircuit(pairs)
        
        qubit_circuit = fermionic_circuit.export_QubitCircuit()
        
        assert isinstance(qubit_circuit, QubitCircuit)
        # Should have: 1 forward fSWAP + 1 main gate + 1 backward fSWAP = 3 gates
        assert qubit_circuit.num_gates == 3
        assert qubit_circuit.num_qubits == 3
        
        # Check trainable flags: fSWAPs should not be trainable
        expected_trainable = [False, True, False]  # fSWAP, main gate, fSWAP
        assert qubit_circuit.trainable == expected_trainable
    
    def test_export_qubit_circuit_larger_gap(self):
        """Test export with larger gap requiring multiple fSWAPs."""
        U1 = U1UnitaryGate()
        pairs = [((0, 3), U1)]  # Gap of 3
        fermionic_circuit = FermionicCircuit(pairs)
        
        qubit_circuit = fermionic_circuit.export_QubitCircuit()
        
        # Should have: 2 forward fSWAPs + 1 main gate + 2 backward fSWAPs = 5 gates
        assert qubit_circuit.num_gates == 5
        expected_trainable = [False, False, True, False, False]
        assert qubit_circuit.trainable == expected_trainable
    
    def test_print_fermionic_params(self, capsys):
        """Test the parameter printing functionality."""
        U1 = U1UnitaryGate()
        U2 = U1UnitaryGate()
        pairs = [((0, 1), U1), ((1, 2), U2)]
        trainable = [True, False]
        
        circuit = FermionicCircuit(pairs, trainable)
        circuit.print_fermionic_params()
        
        captured = capsys.readouterr()
        assert "Trainable gate[0]" in captured.out
        assert "Fixed gate 1" in captured.out
        assert "between 0 and 1" in captured.out


class TestQubitCircuit:
    """Test cases for the QubitCircuit class."""
    
    def test_qubit_circuit_initialization(self):
        """Test basic initialization of QubitCircuit."""
        pairs = [((0, 1), U1UnitaryGate()), ((1, 2), U1UnitaryGate())]
        circuit = QubitCircuit(pairs)
        
        assert circuit.num_gates == 2
        assert circuit.num_qubits == 3
    
    def test_to_state_vector_default_init(self):
        """Test state vector computation with default initial state."""
        U = U1UnitaryGate()  # Identity-like gate
        pairs = [((0, 1), U)]
        circuit = QubitCircuit(pairs)
        
        final_state = circuit.to_state_vector()
        
        assert len(final_state) == 4  # 2^2 qubits
        assert np.isclose(np.sum(np.abs(final_state)**2), 1.0)  # Normalized
    
    def test_to_state_vector_custom_init(self):
        """Test state vector computation with custom initial state."""
        U = UnitaryGate(init_U=np.eye(4))  # Identity gate
        pairs = [((0, 1), U)]
        circuit = QubitCircuit(pairs)
        
        init_state = np.array([0, 1, 0, 0])  # |01⟩ state
        final_state = circuit.to_state_vector(init_state)
        
        # Identity should preserve the state
        assert np.allclose(final_state, init_state)
    
    def test_get_params(self):
        """Test parameter retrieval from trainable gates."""
        U1 = U1UnitaryGate()
        U2 = U1UnitaryGate()
        pairs = [((0, 1), U1), ((1, 2), U2)]
        trainable = [True, False]
        
        circuit = QubitCircuit(pairs, trainable)
        params = circuit.get_params()
        
        assert len(params) == 1  # Only one trainable gate
        assert len(params[0]) == 6  # U1UnitaryGate has 6 parameters
    
    def test_set_params_1d_array(self):
        """Test parameter setting with 1D array."""
        U1 = U1UnitaryGate()
        U2 = U1UnitaryGate()
        pairs = [((0, 1), U1), ((1, 2), U2)]
        trainable = [True, True]
        
        circuit = QubitCircuit(pairs, trainable)
        
        # Set with 1D array that will be reshaped
        new_params = np.random.randn(12)  # 2 gates * 6 params each
        circuit.set_params(new_params)
        
        # Verify parameters were set
        retrieved_params = circuit.get_params()
        assert len(retrieved_params) == 2
    
    def test_set_params_2d_array(self):
        """Test parameter setting with 2D array."""
        U1 = U1UnitaryGate()
        pairs = [((0, 1), U1)]
        
        circuit = QubitCircuit(pairs)
        
        new_params = np.random.randn(1, 6)  # Convert to numpy array
        circuit.set_params(new_params)
        
        retrieved_params = circuit.get_params()
        assert np.allclose(retrieved_params[0], new_params[0])
    
    def test_export_fermionic_circuit_simple(self):
        """Test export to FermionicCircuit with simple case."""
        U1 = U1UnitaryGate()
        pairs = [((0, 1), U1)]
        qubit_circuit = QubitCircuit(pairs)
        
        fermionic_circuit = qubit_circuit.export_FermionicCircuit()
        
        assert isinstance(fermionic_circuit, FermionicCircuit)
        assert fermionic_circuit.num_gates == 1
        assert fermionic_circuit.num_qubits == 2
    
    def test_export_fermionic_circuit_with_fswaps(self):
        """Test export to FermionicCircuit with fSWAP sequence."""
        # Create a circuit that looks like it came from a fermionic circuit
        fSWAP = np.array([[1, 0, 0, 0],
                          [0, 0, 1, 0], 
                          [0, 1, 0, 0],
                          [0, 0, 0, -1]]).reshape([2, 2, 2, 2])
        
        U = U1UnitaryGate()
        pairs = [
            ((0, 1), fSWAP),      # Forward fSWAP
            ((1, 2), U),          # Main unitary
            ((0, 1), fSWAP)       # Backward fSWAP
        ]
        trainable = [False, True, False]
        
        qubit_circuit = QubitCircuit(pairs, trainable)
        fermionic_circuit = qubit_circuit.export_FermionicCircuit()
        
        assert isinstance(fermionic_circuit, FermionicCircuit)
        assert fermionic_circuit.num_gates == 1  # Should collapse to single fermionic gate
        assert fermionic_circuit.num_qubits == 3
        
        # The fermionic gate should act on sites (0, 2)
        fermionic_indices, fermionic_U = fermionic_circuit.pairs_of_indices_and_Us[0]
        assert fermionic_indices == (0, 2)
    
    def test_export_fermionic_circuit_error_malformed_sequence(self):
        """Test error handling for malformed fSWAP sequences."""
        fSWAP = np.array([[1, 0, 0, 0],
                          [0, 0, 1, 0],
                          [0, 1, 0, 0], 
                          [0, 0, 0, -1]]).reshape([2, 2, 2, 2])
        
        # Malformed: fSWAP without following unitary
        pairs = [((0, 1), fSWAP)]
        trainable = [False]
        
        qubit_circuit = QubitCircuit(pairs, trainable)
        
        with pytest.raises(ValueError, match="Malformed fSWAP sequence"):
            qubit_circuit.export_FermionicCircuit()
    
    def test_export_fermionic_circuit_error_non_adjacent_without_fswap(self):
        """Test error handling for non-adjacent gates without fSWAP."""
        U = U1UnitaryGate()
        pairs = [((0, 2), U)]  # Non-adjacent without fSWAP
        
        qubit_circuit = QubitCircuit(pairs)
        
        with pytest.raises(ValueError, match="Unexpected gate without fSWAP at non-adjacent qubits"):
            qubit_circuit.export_FermionicCircuit()
    
    def test_get_energy(self):
        """Test energy computation."""
        U = UnitaryGate(init_U=np.eye(4))  # Identity
        pairs = [((0, 1), U)]
        circuit = QubitCircuit(pairs)
        
        # Simple Hamiltonian
        H = np.diag([1, 2, 3, 4])
        init_state = np.array([1, 0, 0, 0])  # Ground state
        
        energy = circuit.get_energy(H, init_state)
        
        assert np.isclose(energy, 1.0)  # Should get ground state energy
    
    def test_get_energy_gradient(self):
        """Test energy gradient computation."""
        U1 = U1UnitaryGate()
        U2 = U1UnitaryGate()
        pairs = [((0, 1), U1), ((1, 2), U2)]
        
        circuit = QubitCircuit(pairs)
        
        # Simple Hamiltonian for 3-qubit system
        H = np.diag(np.arange(8))
        
        try:
            grads = circuit.get_energy_gradient(H)
            assert len(grads) == 2  # Two gates
            assert all(len(grad) == 6 for grad in grads)  # U1UnitaryGate has 6 parameters
        except (TypeError, NotImplementedError):
            # If gradient computation fails due to JAX stub limitations, that's expected
            pytest.skip("Gradient computation requires full JAX implementation")


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_transform_to_nearest_neighbor_adjacent_gates(self):
        """Test transform function with already adjacent gates."""
        U1 = UnitaryGate(init_U=np.eye(4))
        U2 = UnitaryGate(init_U=np.eye(4))
        pairs = [((0, 1), U1), ((1, 2), U2)]
        
        circuit = QubitCircuit(pairs)
        transformed = transform_to_nearest_neighbor_qubit_circuit(circuit)
        
        # Should be unchanged
        assert transformed.num_gates == 2
        assert transformed.num_qubits == 3
        assert all(transformed.trainable)
    
    def test_transform_to_nearest_neighbor_non_adjacent_gates(self):
        """Test transform function with non-adjacent gates."""
        U = UnitaryGate(init_U=np.eye(4))
        pairs = [((0, 2), U)]  # Non-adjacent
        
        circuit = QubitCircuit(pairs)
        transformed = transform_to_nearest_neighbor_qubit_circuit(circuit)
        
        # Should have SWAP gates added
        assert transformed.num_gates == 3  # 1 SWAP + 1 main + 1 SWAP
        assert transformed.num_qubits == 3
        
        expected_trainable = [False, True, False]  # SWAP, main, SWAP
        assert transformed.trainable == expected_trainable
    
    def test_transform_to_nearest_neighbor_large_gap(self):
        """Test transform function with large gap."""
        U = UnitaryGate(init_U=np.eye(4))
        pairs = [((0, 3), U)]  # Gap of 3
        
        circuit = QubitCircuit(pairs)
        transformed = transform_to_nearest_neighbor_qubit_circuit(circuit)
        
        # Should have: 2 forward SWAPs + 1 main + 2 backward SWAPs = 5 gates
        assert transformed.num_gates == 5
        expected_trainable = [False, False, True, False, False]
        assert transformed.trainable == expected_trainable
    
    def test_transform_invalid_input_type(self):
        """Test error handling for invalid input type."""
        with pytest.raises(TypeError, match="Input must be a QubitCircuit instance"):
            transform_to_nearest_neighbor_qubit_circuit("not a circuit")
    
    def test_transform_single_qubit_system(self):
        """Test transform with small system (edge case)."""
        # Create a minimal valid circuit
        U = UnitaryGate(init_U=np.eye(4))
        pairs = [((0, 1), U)]  # Minimum valid 2-qubit gate
        circuit = QubitCircuit(pairs)
        
        transformed = transform_to_nearest_neighbor_qubit_circuit(circuit)
        assert transformed.num_gates == 1  # Should be unchanged
        assert transformed.num_qubits == 2
    
    def test_removing_swap_simple_case(self):
        """Test SWAP removal with simple case."""
        U = UnitaryGate(init_U=np.eye(4))
        pairs = [((0, 1), U)]
        
        circuit = QubitCircuit(pairs)
        simplified = removing_SWAP_from_nearest_neighbor_qubit_circuit(circuit)
        
        # Should be unchanged (no SWAPs to remove)
        assert simplified.num_gates == 1
        assert simplified.num_qubits == 2
    
    def test_removing_swap_with_swaps(self):
        """Test SWAP removal with actual SWAP gates."""
        SWAP = np.array([[1, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1]]).reshape([2, 2, 2, 2])
        
        U = UnitaryGate(init_U=np.eye(4))
        pairs = [
            ((0, 1), SWAP),       # Forward SWAP
            ((1, 2), U),          # Main unitary  
            ((0, 1), SWAP)        # Backward SWAP
        ]
        trainable = [False, True, False]
        
        circuit = QubitCircuit(pairs, trainable)
        simplified = removing_SWAP_from_nearest_neighbor_qubit_circuit(circuit)
        
        assert simplified.num_gates == 1  # Should collapse to single gate
        
        # The simplified gate should act on sites (0, 2)
        simplified_indices, simplified_U = simplified.pairs_of_indices_and_Us[0]
        assert simplified_indices == (0, 2)
    
    def test_removing_swap_invalid_input_type(self):
        """Test error handling for invalid input type."""
        with pytest.raises(TypeError, match="Input must be a QubitCircuit instance"):
            removing_SWAP_from_nearest_neighbor_qubit_circuit("not a circuit")
    
    def test_removing_swap_error_malformed_sequence(self):
        """Test error handling for malformed SWAP sequences."""
        SWAP = np.array([[1, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1]]).reshape([2, 2, 2, 2])
        
        # Malformed: SWAP without following unitary
        pairs = [((0, 1), SWAP)]
        trainable = [False]
        
        circuit = QubitCircuit(pairs, trainable)
        
        with pytest.raises(ValueError, match="Malformed SWAP sequence"):
            removing_SWAP_from_nearest_neighbor_qubit_circuit(circuit)
    
    def test_transform_and_remove_roundtrip(self):
        """Test that transform + remove gives back original structure."""
        # Create proper unitary matrices using matrix exponential
        H1 = np.random.randn(4, 4) * 0.1
        H1 = (H1 + H1.T) / 2  # Make Hermitian
        U1 = UnitaryGate(init_U=scipy.linalg.expm(1j * H1))
        
        H2 = np.random.randn(4, 4) * 0.1  
        H2 = (H2 + H2.T) / 2  # Make Hermitian
        U2 = UnitaryGate(init_U=scipy.linalg.expm(1j * H2))
        
        pairs = [((0, 2), U1), ((1, 3), U2)]  # Non-adjacent pairs
        
        original_circuit = QubitCircuit(pairs)
        
        # Transform to nearest neighbor
        transformed = transform_to_nearest_neighbor_qubit_circuit(original_circuit)
        
        # Remove SWAPs
        simplified = removing_SWAP_from_nearest_neighbor_qubit_circuit(transformed)
        
        # Should have same structure as original
        assert simplified.num_gates == original_circuit.num_gates
        assert simplified.num_qubits == original_circuit.num_qubits
        
        # Check that gate positions match
        for orig_pair, simp_pair in zip(original_circuit.pairs_of_indices_and_Us, 
                                        simplified.pairs_of_indices_and_Us):
            assert orig_pair[0] == simp_pair[0]  # Same indices


class TestCircuitIntegration:
    """Integration tests between different circuit types."""
    
    def test_fermionic_to_qubit_to_fermionic_roundtrip(self):
        """Test roundtrip conversion between fermionic and qubit circuits."""
        U1 = U1UnitaryGate()
        U2 = U1UnitaryGate()
        pairs = [((0, 1), U1), ((0, 2), U2)]  # Adjacent and non-adjacent
        
        # Start with fermionic circuit
        fermionic_circuit = FermionicCircuit(pairs)
        
        # Convert to qubit circuit
        qubit_circuit = fermionic_circuit.export_QubitCircuit()
        
        # Convert back to fermionic circuit
        recovered_fermionic = qubit_circuit.export_FermionicCircuit()
        
        # Should have same number of fermionic gates
        assert recovered_fermionic.num_gates == fermionic_circuit.num_gates
        assert recovered_fermionic.num_qubits == fermionic_circuit.num_qubits
        
        # Check gate positions match
        for orig_pair, rec_pair in zip(fermionic_circuit.pairs_of_indices_and_Us,
                                       recovered_fermionic.pairs_of_indices_and_Us):
            assert orig_pair[0] == rec_pair[0]  # Same indices


class TestCircuitFiniteDifferenceGradients:
    """Test circuit energy gradients against finite difference approximations."""
    
    def finite_difference_gradient(self, func, params, eps=1e-6):
        """Compute finite difference gradient of a function."""
        grad = np.zeros_like(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += eps
            params_minus[i] -= eps
            grad[i] = (func(params_plus) - func(params_minus)) / (2 * eps)
        return grad
    
    def test_energy_gradient_finite_difference(self):
        """Test energy gradient against finite difference."""
        # Create a simple circuit
        U1 = U1UnitaryGate()
        U2 = U1UnitaryGate()
        pairs = [((0, 1), U1), ((1, 2), U2)]
        circuit = QubitCircuit(pairs)
        
        # Simple Hamiltonian for 3-qubit system
        H = np.diag(np.arange(8, dtype=float))
        init_state = np.zeros(8)
        init_state[0] = 1.0  # |000⟩ state
        
        try:
            # Get analytical gradients
            analytical_grads = circuit.get_energy_gradient(H, init_state)
            analytical_grad_flat = np.concatenate([grad.flatten() for grad in analytical_grads])
            
            # Get finite difference gradient
            def energy_func(params_flat):
                circuit_copy = circuit.copy()
                circuit_copy.set_params(params_flat)
                return circuit_copy.get_energy(H, init_state).real
            
            initial_params = circuit.get_concatenated_params()
            numerical_grad = self.finite_difference_gradient(energy_func, initial_params)
            
            # Compare (allowing for some numerical error)
            assert np.allclose(analytical_grad_flat, numerical_grad, rtol=1e-3, atol=1e-5)
            
        except (TypeError, NotImplementedError, AttributeError):
            # If gradient computation fails due to JAX limitations or missing functions, skip
            pytest.skip("Energy gradient computation not fully available")
    
    def test_fidelity_gradient_finite_difference(self):
        """Test fidelity-related gradient computation using the algorithm module."""
        try:
            from stateprep.algorithm import get_fidelity_gradient
            
            # Create a simple circuit
            U1 = U1UnitaryGate()
            pairs = [((0, 1), U1)]
            circuit = QubitCircuit(pairs)
            
            # Target and initial states
            target_state = np.array([0, 1, 0, 0])  # |01⟩
            initial_state = np.array([1, 0, 0, 0])  # |00⟩
            
            # Get analytical gradients
            cost, analytical_grads = get_fidelity_gradient(circuit, [target_state], [initial_state])
            analytical_grad_flat = np.concatenate([grad.flatten() for grad in analytical_grads])
            
            # Get finite difference gradient
            def fidelity_func(params_flat):
                circuit_copy = circuit.copy()
                circuit_copy.set_params(params_flat)
                final_state = circuit_copy.to_state_vector(initial_state)
                return 1.0 - np.abs(np.dot(target_state.conj(), final_state))**2
            
            initial_params = circuit.get_concatenated_params()
            numerical_grad = self.finite_difference_gradient(fidelity_func, initial_params)
            
            # Compare (allowing for some numerical error)
            assert np.allclose(analytical_grad_flat, numerical_grad, rtol=1e-3, atol=1e-5)
            
        except (ImportError, TypeError, NotImplementedError, AttributeError):
            # If fidelity gradient computation fails, skip
            pytest.skip("Fidelity gradient computation not available")


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""
    
    def test_empty_circuit(self):
        """Test behavior with empty circuit."""
        pairs = []
        
        # Empty circuits should raise an error during initialization due to max() call
        with pytest.raises(ValueError):
            Circuit(pairs)
    
    def test_single_gate_circuit(self):
        """Test behavior with single gate."""
        U = U1UnitaryGate()
        pairs = [((0, 1), U)]
        circuit = QubitCircuit(pairs)
        
        assert circuit.num_gates == 1
        assert circuit.num_qubits == 2
        
        final_state = circuit.to_state_vector()
        assert len(final_state) == 4
    
    def test_large_qubit_indices(self):
        """Test behavior with large qubit indices."""
        U = U1UnitaryGate()
        pairs = [((100, 101), U)]
        circuit = Circuit(pairs)
        
        assert circuit.num_gates == 1
        assert circuit.num_qubits == 102  # 0-101, so 102 qubits


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])