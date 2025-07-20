"""
Comprehensive test suite for gate.py module.

This test suite covers all classes and functions in the gate.py module:
- UnitaryGate class
- U1UnitaryGate class  
- FreeFermionGate class
- FermionicGate class
- All utility functions (including new free fermion and fermionic functions)
- JAX implementations
- Parameter consistency checks
- Edge cases

Total tests: 61
"""

import pytest
import numpy as np
import scipy.linalg
import jax.numpy as jnp
from stateprep.gate import (
    UnitaryGate, U1UnitaryGate, FreeFermionGate, FermionicGate,
    get_unitary_gate, get_unitary_params,
    get_U1_unitary_gate, get_U1_unitary_params,
    get_free_fermion_gate, get_free_fermion_unitary_params,
    get_fermionic_gate, get_fermionic_unitary_params,
    jax_get_unitary_gate, jax_get_U1_unitary_gate,
    func_val_from_h_params, gradient_from_h_params,
    func_val_from_U1_h_params, gradient_from_U1_h_params,
    gradient_from_free_fermion_h_params, gradient_from_fermionic_h_params
)


class TestUnitaryGate:
    """Test cases for the UnitaryGate class."""
    
    def test_init_default(self):
        """Test default initialization of UnitaryGate."""
        gate = UnitaryGate()
        assert gate.shape == (4, 4)
        assert hasattr(gate, 'params')
        assert len(gate.params) == 16
        # Check unitarity
        assert np.allclose(gate @ gate.conj().T, np.eye(4))
    
    def test_init_with_params(self):
        """Test initialization with specific parameters."""
        params = np.random.randn(16) * 0.1
        gate = UnitaryGate(init_params=params)
        assert gate.shape == (4, 4)
        assert np.allclose(gate.params, params)
        assert np.allclose(gate @ gate.conj().T, np.eye(4))
    
    def test_init_with_unitary(self):
        """Test initialization with unitary matrix."""
        # Create a proper unitary matrix using expm
        H = np.random.randn(4, 4) * 0.1
        H = (H + H.T) / 2  # Make Hermitian
        U = scipy.linalg.expm(1j * H)  # Proper unitary
        gate = UnitaryGate(init_U=U)
        assert gate.shape == (4, 4)
        assert hasattr(gate, 'params')
        assert len(gate.params) == 16
    
    def test_get_parameters(self):
        """Test parameter retrieval."""
        params = np.random.randn(16) * 0.1
        gate = UnitaryGate(init_params=params)
        retrieved_params = gate.get_parameters()
        assert np.allclose(retrieved_params, params)
    
    def test_set_parameters(self):
        """Test parameter setting."""
        gate = UnitaryGate()
        new_params = np.random.randn(16) * 0.1
        gate.set_parameters(new_params)
        assert np.allclose(gate.params, new_params)
        assert np.allclose(gate @ gate.conj().T, np.eye(4))
    
    def test_get_gradient(self):
        """Test gradient computation."""
        gate = UnitaryGate()
        dU_mat = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
        grad = gate.get_gradient(dU_mat)
        assert len(grad) == 16
        assert np.all(np.isfinite(grad))
    
    def test_invalid_params_length(self):
        """Test error handling for wrong parameter length."""
        with pytest.raises(AssertionError):
            UnitaryGate(init_params=np.random.randn(10))
    
    def test_invalid_unitary_shape(self):
        """Test error handling for wrong unitary shape."""
        with pytest.raises(AssertionError):
            UnitaryGate(init_U=np.random.randn(3, 3))
    
    def test_array_finalize(self):
        """Test __array_finalize__ method."""
        gate1 = UnitaryGate()
        gate2 = gate1.copy()
        assert hasattr(gate2, 'params')
        assert np.allclose(gate2.params, gate1.params)


class TestU1UnitaryGate:
    """Test cases for the U1UnitaryGate class."""
    
    def test_init_default(self):
        """Test default initialization of U1UnitaryGate."""
        gate = U1UnitaryGate()
        assert gate.shape == (4, 4)
        assert hasattr(gate, 'params')
        assert len(gate.params) == 6
        # Check unitarity
        assert np.allclose(gate @ gate.conj().T, np.eye(4))
    
    def test_init_with_params(self):
        """Test initialization with specific parameters."""
        params = np.random.randn(6) * 0.1
        gate = U1UnitaryGate(init_params=params)
        assert gate.shape == (4, 4)
        assert np.allclose(gate.params, params)
        assert np.allclose(gate @ gate.conj().T, np.eye(4))
    
    def test_get_parameters(self):
        """Test parameter retrieval."""
        params = np.random.randn(6) * 0.1
        gate = U1UnitaryGate(init_params=params)
        retrieved_params = gate.get_parameters()
        assert np.allclose(retrieved_params, params)
    
    def test_set_parameters(self):
        """Test parameter setting."""
        gate = U1UnitaryGate()
        new_params = np.random.randn(6) * 0.1
        gate.set_parameters(new_params)
        assert np.allclose(gate.params, new_params)
        assert np.allclose(gate @ gate.conj().T, np.eye(4))
    
    def test_get_gradient(self):
        """Test gradient computation."""
        gate = U1UnitaryGate()
        dU_mat = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
        grad = gate.get_gradient(dU_mat)
        assert len(grad) == 6
        assert np.all(np.isfinite(grad))
    
    def test_invalid_params_length(self):
        """Test error handling for wrong parameter length."""
        with pytest.raises(AssertionError):
            U1UnitaryGate(init_params=np.random.randn(10))
    
    def test_invalid_unitary_shape(self):
        """Test error handling for wrong unitary shape."""
        with pytest.raises(AssertionError):
            U1UnitaryGate(init_U=np.random.randn(3, 3))
    
    def test_decompose_fermionic_gate(self):
        """Test decompose_fermionic_gate method."""
        gate = U1UnitaryGate()
        coefficients = gate.decompose_fermionic_gate()
        assert len(coefficients) == 6
        assert np.all(np.isfinite(coefficients))
        assert np.all(np.isreal(coefficients))


class TestFreeFermionGate:
    """Test cases for the FreeFermionGate class."""
    
    def test_init_default(self):
        """Test default initialization of FreeFermionGate."""
        gate = FreeFermionGate()
        assert gate.shape == (4, 4)
        assert hasattr(gate, 'params')
        assert len(gate.params) == 6
        # Check unitarity
        assert np.allclose(gate @ gate.conj().T, np.eye(4))
    
    def test_init_with_params(self):
        """Test initialization with specific parameters."""
        params = np.random.randn(6) * 0.1
        gate = FreeFermionGate(init_params=params)
        assert gate.shape == (4, 4)
        assert np.allclose(gate.params, params)
        assert np.allclose(gate @ gate.conj().T, np.eye(4))
    
    def test_init_with_unitary(self):
        """Test initialization with unitary matrix."""
        # Create a proper free fermion unitary matrix using the utility function
        params = np.random.randn(6) * 0.1
        U = get_free_fermion_gate(params)
        gate = FreeFermionGate(init_U=U)
        assert gate.shape == (4, 4)
        assert hasattr(gate, 'params')
        assert len(gate.params) == 6
    
    def test_get_parameters(self):
        """Test parameter retrieval."""
        params = np.random.randn(6) * 0.1
        gate = FreeFermionGate(init_params=params)
        retrieved_params = gate.get_parameters()
        assert np.allclose(retrieved_params, params)
    
    def test_set_parameters(self):
        """Test parameter setting."""
        gate = FreeFermionGate()
        new_params = np.random.randn(6) * 0.1
        gate.set_parameters(new_params)
        assert np.allclose(gate.params, new_params)
        assert np.allclose(gate @ gate.conj().T, np.eye(4))
    
    def test_get_gradient_raises_not_implemented(self):
        """Test that gradient computation raises NotImplementedError."""
        gate = FreeFermionGate()
        dU_mat = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
        with pytest.raises(NotImplementedError, match="Free fermion gate gradient is not implemented yet"):
            gate.get_gradient(dU_mat)
    
    def test_invalid_params_length(self):
        """Test error handling for wrong parameter length."""
        with pytest.raises(AssertionError):
            FreeFermionGate(init_params=np.random.randn(10))
    
    def test_invalid_unitary_shape(self):
        """Test error handling for wrong unitary shape."""
        with pytest.raises(AssertionError):
            FreeFermionGate(init_U=np.random.randn(3, 3))
    
    def test_array_finalize(self):
        """Test __array_finalize__ method."""
        gate1 = FreeFermionGate()
        gate2 = gate1.copy()
        assert hasattr(gate2, 'params')
        assert np.allclose(gate2.params, gate1.params)

class TestFermionicGate:
    """Test cases for the FermionicGate class."""
    
    def test_init_default(self):
        """Test default initialization of FermionicGate."""
        gate = FermionicGate()
        assert gate.shape == (4, 4)
        assert hasattr(gate, 'params')
        assert len(gate.params) == 8
        # Check unitarity
        assert np.allclose(gate @ gate.conj().T, np.eye(4))
    
    def test_init_with_params(self):
        """Test initialization with specific parameters."""
        params = np.random.randn(8) * 0.1
        gate = FermionicGate(init_params=params)
        assert gate.shape == (4, 4)
        assert np.allclose(gate.params, params)
        assert np.allclose(gate @ gate.conj().T, np.eye(4))
    
    def test_init_with_unitary(self):
        """Test initialization with unitary matrix."""
        # Create a proper fermionic unitary matrix using the utility function
        params = np.random.randn(8) * 0.1
        U = get_fermionic_gate(params)
        gate = FermionicGate(init_U=U)
        assert gate.shape == (4, 4)
        assert hasattr(gate, 'params')
        assert len(gate.params) == 8
    
    def test_get_parameters(self):
        """Test parameter retrieval."""
        params = np.random.randn(8) * 0.1
        gate = FermionicGate(init_params=params)
        retrieved_params = gate.get_parameters()
        assert np.allclose(retrieved_params, params)
    
    def test_set_parameters(self):
        """Test parameter setting."""
        gate = FermionicGate()
        new_params = np.random.randn(8) * 0.1
        gate.set_parameters(new_params)
        assert np.allclose(gate.params, new_params)
        assert np.allclose(gate @ gate.conj().T, np.eye(4))
    
    def test_get_gradient_raises_not_implemented(self):
        """Test that gradient computation raises NotImplementedError."""
        gate = FermionicGate()
        dU_mat = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
        with pytest.raises(NotImplementedError, match="Fermionic gate gradient is not implemented yet"):
            gate.get_gradient(dU_mat)
    
    def test_invalid_params_length(self):
        """Test error handling for wrong parameter length."""
        with pytest.raises(AssertionError):
            FermionicGate(init_params=np.random.randn(10))
    
    def test_invalid_unitary_shape(self):
        """Test error handling for wrong unitary shape."""
        with pytest.raises(AssertionError):
            FermionicGate(init_U=np.random.randn(3, 3))
    
    def test_array_finalize(self):
        """Test __array_finalize__ method."""
        gate1 = FermionicGate()
        gate2 = gate1.copy()
        assert hasattr(gate2, 'params')
        assert np.allclose(gate2.params, gate1.params)


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    def test_get_unitary_gate(self):
        """Test get_unitary_gate function."""
        params = np.random.randn(16) * 0.1
        U = get_unitary_gate(params)
        assert U.shape == (4, 4)
        assert np.allclose(U @ U.conj().T, np.eye(4))
    
    def test_get_unitary_params(self):
        """Test get_unitary_params function."""
        params = np.random.randn(16) * 0.1
        U = get_unitary_gate(params)
        recovered_params = get_unitary_params(U)
        assert len(recovered_params) == 16
        # Check if we can recover similar unitary (up to phase)
        U_recovered = get_unitary_gate(recovered_params)
        assert np.allclose(np.abs(U), np.abs(U_recovered), atol=1e-10)
    
    def test_get_U1_unitary_gate(self):
        """Test get_U1_unitary_gate function."""
        params = np.random.randn(6) * 0.1
        U = get_U1_unitary_gate(params)
        assert U.shape == (4, 4)
        assert np.allclose(U @ U.conj().T, np.eye(4))
    
    def test_get_U1_unitary_params(self):
        """Test get_U1_unitary_params function."""
        params = np.random.randn(6) * 0.1
        U = get_U1_unitary_gate(params)
        recovered_params = get_U1_unitary_params(U)
        assert len(recovered_params) == 6
        # Check if we can recover similar unitary (up to phase)
        U_recovered = get_U1_unitary_gate(recovered_params)
        assert np.allclose(np.abs(U), np.abs(U_recovered), atol=1e-10)
    
    def test_jax_get_unitary_gate(self):
        """Test JAX implementation of get_unitary_gate."""
        params = np.random.randn(16) * 0.1
        U_np = get_unitary_gate(params)
        U_jax = jax_get_unitary_gate(params)
        assert np.allclose(U_np, U_jax)
    
    def test_jax_get_U1_unitary_gate(self):
        """Test JAX implementation of get_U1_unitary_gate."""
        params = np.random.randn(6) * 0.1
        U_np = get_U1_unitary_gate(params)
        U_jax = jax_get_U1_unitary_gate(params)
        assert np.allclose(U_np, U_jax)
    
    def test_func_val_from_h_params(self):
        """Test energy computation from parameters."""
        params = np.random.randn(16) * 0.1
        dU_mat = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
        energy = func_val_from_h_params(params, dU_mat)
        assert energy.ndim == 0  # JAX scalar
        assert np.isreal(energy)
    
    def test_gradient_from_h_params(self):
        """Test gradient computation from parameters."""
        params = np.random.randn(16) * 0.1
        dU_mat = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
        grad = gradient_from_h_params(params, dU_mat)
        assert len(grad) == 16
        assert np.all(np.isfinite(grad))
    
    def test_func_val_from_U1_h_params(self):
        """Test U1 energy computation from parameters."""
        params = np.random.randn(6) * 0.1
        dU_mat = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
        energy = func_val_from_U1_h_params(params, dU_mat)
        assert energy.ndim == 0  # JAX scalar
        assert np.isreal(energy)
    
    def test_gradient_from_U1_h_params(self):
        """Test U1 gradient computation from parameters."""
        params = np.random.randn(6) * 0.1
        dU_mat = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
        grad = gradient_from_U1_h_params(params, dU_mat)
        assert len(grad) == 6
        assert np.all(np.isfinite(grad))
    
    def test_get_free_fermion_gate(self):
        """Test get_free_fermion_gate function."""
        params = np.random.randn(6) * 0.1
        U = get_free_fermion_gate(params)
        assert U.shape == (4, 4)
        assert np.allclose(U @ U.conj().T, np.eye(4))
    
    def test_get_free_fermion_unitary_params(self):
        """Test get_free_fermion_unitary_params function."""
        params = np.random.randn(6) * 0.1
        U = get_free_fermion_gate(params)
        recovered_params = get_free_fermion_unitary_params(U)
        assert len(recovered_params) == 6
        # Check if we can recover similar unitary (up to phase)
        U_recovered = get_free_fermion_gate(recovered_params)
        assert np.allclose(np.abs(U), np.abs(U_recovered), atol=1e-10)
    
    def test_get_fermionic_gate(self):
        """Test get_fermionic_gate function."""
        params = np.random.randn(8) * 0.1
        U = get_fermionic_gate(params)
        assert U.shape == (4, 4)
        assert np.allclose(U @ U.conj().T, np.eye(4))
    
    def test_get_fermionic_unitary_params(self):
        """Test get_fermionic_unitary_params function."""
        params = np.random.randn(8) * 0.1
        U = get_fermionic_gate(params)
        recovered_params = get_fermionic_unitary_params(U)
        assert len(recovered_params) == 8
        # Check if we can recover similar unitary (up to phase)
        U_recovered = get_fermionic_gate(recovered_params)
        assert np.allclose(np.abs(U), np.abs(U_recovered), atol=1e-10)
    
    def test_gradient_from_free_fermion_h_params_raises_not_implemented(self):
        """Test that free fermion gradient raises NotImplementedError."""
        params = np.random.randn(6) * 0.1
        dU_mat = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
        with pytest.raises(NotImplementedError, match="Free fermion gate gradient is not implemented yet"):
            gradient_from_free_fermion_h_params(params, dU_mat)
    
    def test_gradient_from_fermionic_h_params_raises_not_implemented(self):
        """Test that fermionic gradient raises NotImplementedError."""
        params = np.random.randn(8) * 0.1
        dU_mat = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
        with pytest.raises(NotImplementedError, match="Fermionic gate gradient is not implemented yet"):
            gradient_from_fermionic_h_params(params, dU_mat)


class TestParameterConsistency:
    """Test parameter consistency across conversions."""
    
    def test_unitary_parameter_roundtrip(self):
        """Test that parameter -> unitary -> parameter roundtrip works."""
        original_params = np.random.randn(16) * 0.1
        U = get_unitary_gate(original_params)
        recovered_params = get_unitary_params(U)
        U_recovered = get_unitary_gate(recovered_params)
        # Check that unitaries are equivalent (up to global phase)
        assert np.allclose(np.abs(U), np.abs(U_recovered), atol=1e-10)
    
    def test_U1_parameter_roundtrip(self):
        """Test that U1 parameter -> unitary -> parameter roundtrip works."""
        original_params = np.random.randn(6) * 0.1
        U = get_U1_unitary_gate(original_params)
        recovered_params = get_U1_unitary_params(U)
        U_recovered = get_U1_unitary_gate(recovered_params)
        # Check that unitaries are equivalent (up to global phase)
        assert np.allclose(np.abs(U), np.abs(U_recovered), atol=1e-10)
    
    def test_free_fermion_parameter_roundtrip(self):
        """Test that free fermion parameter -> unitary -> parameter roundtrip works."""
        original_params = np.random.randn(6) * 0.1
        U = get_free_fermion_gate(original_params)
        recovered_params = get_free_fermion_unitary_params(U)
        U_recovered = get_free_fermion_gate(recovered_params)
        # Check that unitaries are equivalent (up to global phase)
        assert np.allclose(np.abs(U), np.abs(U_recovered), atol=1e-10)
    
    def test_fermionic_parameter_roundtrip(self):
        """Test that fermionic parameter -> unitary -> parameter roundtrip works."""
        original_params = np.random.randn(8) * 0.1
        U = get_fermionic_gate(original_params)
        recovered_params = get_fermionic_unitary_params(U)
        U_recovered = get_fermionic_gate(recovered_params)
        # Check that unitaries are equivalent (up to global phase)
        assert np.allclose(np.abs(U), np.abs(U_recovered), atol=1e-10)


class TestJAXNumpyEquivalence:
    """Test that JAX and NumPy implementations give equivalent results."""
    
    def test_jax_numpy_unitary_equivalence(self):
        """Test JAX and NumPy implementations give same results."""
        params = np.random.randn(16) * 0.1
        U_np = get_unitary_gate(params)
        U_jax = np.array(jax_get_unitary_gate(params))
        assert np.allclose(U_np, U_jax)
    
    def test_jax_numpy_U1_equivalence(self):
        """Test JAX and NumPy U1 implementations give same results."""
        params = np.random.randn(6) * 0.1
        U_np = get_U1_unitary_gate(params)
        U_jax = np.array(jax_get_U1_unitary_gate(params))
        assert np.allclose(U_np, U_jax)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_parameters(self):
        """Test behavior with zero parameters."""
        zero_params = np.zeros(16)
        U = get_unitary_gate(zero_params)
        assert np.allclose(U, np.eye(4))
        
        zero_params_U1 = np.zeros(6)
        U_U1 = get_U1_unitary_gate(zero_params_U1)
        assert np.allclose(U_U1, np.eye(4))
    
    def test_large_parameters(self):
        """Test behavior with large parameters."""
        large_params = np.random.randn(16) * 10
        U = get_unitary_gate(large_params)
        assert np.allclose(U @ U.conj().T, np.eye(4))
        
        large_params_U1 = np.random.randn(6) * 10
        U_U1 = get_U1_unitary_gate(large_params_U1)
        assert np.allclose(U_U1 @ U_U1.conj().T, np.eye(4))
    
    def test_complex_dU_matrix(self):
        """Test gradient computation with complex dU matrix."""
        params = np.random.randn(16) * 0.1
        dU_complex = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
        grad = gradient_from_h_params(params, dU_complex)
        assert len(grad) == 16
        assert np.all(np.isfinite(grad))
        
        params_U1 = np.random.randn(6) * 0.1
        grad_U1 = gradient_from_U1_h_params(params_U1, dU_complex)
        assert len(grad_U1) == 6
        assert np.all(np.isfinite(grad_U1))


class TestArrayFinalize:
    """Test __array_finalize__ method behavior."""
    
    def test_array_finalize_preserves_params(self):
        """Test that array operations preserve params attribute."""
        gate = UnitaryGate()
        original_params = gate.params.copy()
        
        # Test copy
        gate_copy = gate.copy()
        assert hasattr(gate_copy, 'params')
        assert np.allclose(gate_copy.params, original_params)
        
        # Test view
        gate_view = gate.view(UnitaryGate)
        assert hasattr(gate_view, 'params')
        assert np.allclose(gate_view.params, original_params)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
