"""
Gate module containing quantum gate classes.
"""


class UnitaryGate:
    """
    Base class for unitary quantum gates.
    """
    
    def __init__(self, name=None):
        """
        Initialize the unitary gate.
        
        Args:
            name (str, optional): Name of the gate. Defaults to None.
        """
        self.name = name or self.__class__.__name__
    
    def apply(self, state):
        """
        Apply the gate to a quantum state.
        
        Args:
            state: The quantum state to apply the gate to.
            
        Returns:
            The transformed quantum state.
            
        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the apply method")
    
    def matrix(self):
        """
        Get the matrix representation of the gate.
        
        Returns:
            numpy.ndarray: The matrix representation of the gate.
            
        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the matrix method")
    
    def __str__(self):
        """
        String representation of the gate.
        
        Returns:
            str: The name of the gate.
        """
        return self.name


class FreeFermionGate(UnitaryGate):
    """
    A gate class for free fermion operations.
    Inherits from UnitaryGate class.
    """
    
    def __init__(self, name=None):
        """
        Initialize the free fermion gate.
        
        Args:
            name (str, optional): Name of the gate. Defaults to None.
        """
        super().__init__(name)
    
    def apply(self, state):
        """
        Apply the free fermion gate to a quantum state.
        
        Args:
            state: The quantum state to apply the gate to.
            
        Raises:
            NotImplementedError: This functionality is not yet implemented.
        """
        raise NotImplementedError("FreeFermionGate apply method is not implemented")
    
    def matrix(self):
        """
        Get the matrix representation of the free fermion gate.
        
        Returns:
            numpy.ndarray: The matrix representation of the gate.
            
        Raises:
            NotImplementedError: This functionality is not yet implemented.
        """
        raise NotImplementedError("FreeFermionGate matrix method is not implemented")
    
    def fermionic_transform(self, operators):
        """
        Apply fermionic transformation to operators.
        
        Args:
            operators: The fermionic operators to transform.
            
        Raises:
            NotImplementedError: This functionality is not yet implemented.
        """
        raise NotImplementedError("FreeFermionGate fermionic_transform method is not implemented")