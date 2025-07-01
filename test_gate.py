"""
Test module for gate.py classes.
"""
import pytest
from gate import UnitaryGate, FreeFermionGate


class TestUnitaryGate:
    """Test cases for the UnitaryGate class."""
    
    def test_unitarygate_initialization_with_name(self):
        """Test UnitaryGate initialization with a custom name."""
        gate = UnitaryGate(name="TestGate")
        assert gate.name == "TestGate"
    
    def test_unitarygate_initialization_without_name(self):
        """Test UnitaryGate initialization without a custom name."""
        gate = UnitaryGate()
        assert gate.name == "UnitaryGate"
    
    def test_unitarygate_str_representation(self):
        """Test string representation of UnitaryGate."""
        gate = UnitaryGate(name="CustomGate")
        assert str(gate) == "CustomGate"
        
        gate_default = UnitaryGate()
        assert str(gate_default) == "UnitaryGate"
    
    def test_unitarygate_apply_not_implemented(self):
        """Test that UnitaryGate.apply raises NotImplementedError."""
        gate = UnitaryGate()
        with pytest.raises(NotImplementedError, match="Subclasses must implement the apply method"):
            gate.apply("dummy_state")
    
    def test_unitarygate_matrix_not_implemented(self):
        """Test that UnitaryGate.matrix raises NotImplementedError."""
        gate = UnitaryGate()
        with pytest.raises(NotImplementedError, match="Subclasses must implement the matrix method"):
            gate.matrix()


class TestFreeFermionGate:
    """Test cases for the FreeFermionGate class."""
    
    def test_freefermiongate_initialization_with_name(self):
        """Test FreeFermionGate initialization with a custom name."""
        gate = FreeFermionGate(name="CustomFreeFermion")
        assert gate.name == "CustomFreeFermion"
    
    def test_freefermiongate_initialization_without_name(self):
        """Test FreeFermionGate initialization without a custom name."""
        gate = FreeFermionGate()
        assert gate.name == "FreeFermionGate"
    
    def test_freefermiongate_inherits_from_unitarygate(self):
        """Test that FreeFermionGate inherits from UnitaryGate."""
        gate = FreeFermionGate()
        assert isinstance(gate, UnitaryGate)
        assert isinstance(gate, FreeFermionGate)
    
    def test_freefermiongate_str_representation(self):
        """Test string representation of FreeFermionGate."""
        gate = FreeFermionGate(name="MyFreeFermionGate")
        assert str(gate) == "MyFreeFermionGate"
        
        gate_default = FreeFermionGate()
        assert str(gate_default) == "FreeFermionGate"
    
    def test_freefermiongate_apply_not_implemented(self):
        """Test that FreeFermionGate.apply raises NotImplementedError."""
        gate = FreeFermionGate()
        with pytest.raises(NotImplementedError, match="FreeFermionGate apply method is not implemented"):
            gate.apply("dummy_state")
    
    def test_freefermiongate_matrix_not_implemented(self):
        """Test that FreeFermionGate.matrix raises NotImplementedError."""
        gate = FreeFermionGate()
        with pytest.raises(NotImplementedError, match="FreeFermionGate matrix method is not implemented"):
            gate.matrix()
    
    def test_freefermiongate_fermionic_transform_not_implemented(self):
        """Test that FreeFermionGate.fermionic_transform raises NotImplementedError."""
        gate = FreeFermionGate()
        with pytest.raises(NotImplementedError, match="FreeFermionGate fermionic_transform method is not implemented"):
            gate.fermionic_transform("dummy_operators")


class TestGateInteractions:
    """Test cases for interactions between gate classes."""
    
    def test_inheritance_hierarchy(self):
        """Test the inheritance hierarchy is correct."""
        # Test that FreeFermionGate is a subclass of UnitaryGate
        assert issubclass(FreeFermionGate, UnitaryGate)
        
        # Test method resolution order
        gate = FreeFermionGate()
        assert gate.__class__.__mro__ == (FreeFermionGate, UnitaryGate, object)
    
    def test_different_instances_are_independent(self):
        """Test that different gate instances are independent."""
        gate1 = FreeFermionGate(name="Gate1")
        gate2 = FreeFermionGate(name="Gate2")
        unitary_gate = UnitaryGate(name="UnitaryGate")
        
        assert gate1.name != gate2.name
        assert gate1.name != unitary_gate.name
        assert gate2.name != unitary_gate.name
    
    def test_class_names_match_expected(self):
        """Test that class names are as expected."""
        unitary_gate = UnitaryGate()
        fermion_gate = FreeFermionGate()
        
        assert unitary_gate.__class__.__name__ == "UnitaryGate"
        assert fermion_gate.__class__.__name__ == "FreeFermionGate"