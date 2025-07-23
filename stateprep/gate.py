import numpy as np
import scipy
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.linalg import expm as jax_expm
from stateprep.utils.common_setup import hopping, current, ZZ, Z1, Z2

"""
gate.py:
	Classes:
		1. Generic SU(4) gate -> 16 parameters including phase
		2. U1 preserving gate -> 6 parameters including phase
		3. Free fermion gate
        4. Generic fermion gate
        5. Diagonal gate -> 4 parameters including phase
	Class_methods:
		get_parameters()
        set_parameters(params)
		get_decomp_params()
		get_dU_dp()

The class is consturcted by subclassing the numpy.ndarray class.
Additional class methods are added to the class to
    1. compute the parameters of the unitary gate.
    2. compute the derivative of the unitary gate with respect to the parameters.
    3. get the decomposition of unitary gates to elementary gates.
"""

class UnitaryGate(np.ndarray):
    def __new__(cls, init_U=None, init_params=None, scale=1e-3):
        """
        Create a new UnitaryGate instance.

        Parameters
        ----------
        init_U : np.ndarray, optional
            Initial unitary matrix (default is None).
        init_params : np.ndarray, optional
            Initial parameters (default is None).

        Returns
        -------
        UnitaryGate
            A new instance of the UnitaryGate class.
        """
        if init_U is not None:
            obj = np.asarray(init_U).view(cls)
            assert obj.shape == (4, 4), "Unitary matrix must be 4x4."
            assert np.allclose(np.dot(obj, obj.conj().T), np.eye(4)), "Matrix must be unitary."
            obj.params = get_unitary_params(init_U)
        elif init_params is not None:
            assert len(init_params) == 16, "Parameters must be a vector of length 16."
            obj = get_unitary_gate(init_params).view(cls)
            obj.params = init_params
        else:
            init_params = (np.random.rand(16) - 0.5) * scale
            obj = get_unitary_gate(init_params).view(cls)
            obj.params = init_params

        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.params = getattr(obj, 'params', None)
        # ---- class specific attributes ----
        self.num_params = 16  # Number of parameters for a U(4) gate
        self.get_unitary_from = get_unitary_gate
        self.get_gradient_from = gradient_from_h_params

    def get_parameters(self):
        """
        Get the parameters of the unitary gate.

        Returns
        -------
        np.ndarray
            Parameters of the unitary gate.
        """
        return self.params

    def set_parameters(self, params):
        """
        Set the parameters of the unitary gate.

        Parameters
        ----------
        params : np.ndarray
            New parameters for the unitary gate.
        """
        assert len(params) == self.num_params, f"Parameters must be a vector of length {self.num_params}."
        self.params = params
        self[:] = self.get_unitary_from(params)

    def get_gradient(self, dU_mat):
        """
        Compute the gradient of the unitary gate with respect to the parameters.

        Parameters
        ----------
        dU_mat : np.ndarray
            Matrix to compute the gradient with respect to.

        Returns
        -------
        np.ndarray
            Gradient of the unitary gate.
        """
        return np.array(self.get_gradient_from(self.params, dU_mat))

class U1UnitaryGate(UnitaryGate):
    def __new__(cls, init_U=None, init_params=None, scale=1e-3):
        """
        Create a new U1UnitaryGate instance.

        Parameters
        ----------
        init_U : np.ndarray, optional
            Initial unitary matrix (default is None).
        init_params : np.ndarray, optional
            Initial parameters (default is None).

        Returns
        -------
        U1UnitaryGate
            A new instance of the U1UnitaryGate class.
        """
        if init_U is not None:
            obj = np.asarray(init_U).view(cls)
            assert obj.shape == (4, 4), "Unitary matrix must be 4x4."
            assert np.allclose(np.dot(obj, obj.conj().T), np.eye(4)), "Matrix must be unitary."
            obj.params = get_U1_unitary_params(init_U)
        elif init_params is not None:
            assert len(init_params) == 6, "Parameters must be a vector of length 6."
            obj = get_U1_unitary_gate(init_params).view(cls)
            obj.params = init_params
        else:
            init_params = (np.random.rand(6) - 0.5) * scale
            obj = get_U1_unitary_gate(init_params).view(cls)
            obj.params = init_params

        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.params = getattr(obj, 'params', None)
        # ---- class specific attributes ----
        self.num_params = 6  # Number of parameters for the U1 preserving gate
        self.get_unitary_from = get_U1_unitary_gate
        self.get_gradient_from = gradient_from_U1_h_params

    def decompose_fermionic_gate(self):
        """
        Decompose the U1 unitary gate into fermionic gates.

        Returns
        -------
        list
            List of fermionic gates.
        """
        H = scipy.linalg.logm(self) / -1.j
        c0 = np.trace(H @ np.eye(4)) / 4.
        c1 = np.trace(H @ hopping) / 2.
        c2 = np.trace(H @ current) / 2.
        c3 = np.trace(H @ ZZ) / 4.
        c4 = np.trace(H @ Z1) / 4.
        c5 = np.trace(H @ Z2) / 4.
        coefficients = np.array([c0, c1, c2, c3, c4, c5])
        coefficients = np.real_if_close(coefficients, 1e-10)
        return coefficients

class FreeFermionGate(UnitaryGate):
    """
    Free Fermion Gate class.

    This class represents a free fermion gate as described in Eq.(1) of
    https://arxiv.org/pdf/0804.4050

    A free fermion gate is defined as a unitary gate of the form:
    G(A, B) = [[p, 0, 0, q],
               [0, w, x, 0],
               [0, y, z, 0],
               [r, 0, 0, s]]
    where A = [[p, q],  and B = [[w, x],
               [r, s]]           [y, z]]
    The free fermion condition is that
    A and B are either
    (i) both in SU(2) or
    (ii) both in U(2) with the same determinant.
    Thus the total number of parameters is 6.

    - Note the only difference to the generic fermionic gate is the
      determinant condition.
    - Note that a free fermion gate can break the U(1) symmetry.
    - Note that a U(1) preserving gate can be a non free fermion gate.

    If a circuit is constructed using only free fermion gates,
    it can be efficiently simulated using a free fermion simulator.
    """

    def __new__(cls, init_U=None, init_params=None, scale=1e-3):
        """
        Create a new FreeFermionGate instance.

        Currently raises NotImplementedError as the implementation is not yet available.

        Parameters
        ----------
        init_U : np.ndarray, optional
            Initial unitary matrix (default is None).
        init_params : np.ndarray, optional
            Initial parameters (default is None).
        scale : float, optional
            Scale factor for random initialization (default is 1e-3).

        Returns
        -------
        FreeFermionGate
            A new instance of the FreeFermionGate class.
        """
        if init_U is not None:
            obj = np.asarray(init_U).view(cls)
            assert obj.shape == (4, 4), "Unitary matrix must be 4x4."
            assert np.allclose(np.dot(obj, obj.conj().T), np.eye(4)), "Matrix must be unitary."
            obj.params = get_free_fermion_unitary_params(init_U)
        elif init_params is not None:
            assert len(init_params) == 6, "Parameters must be a vector of length 6."
            obj = get_free_fermion_gate(init_params).view(cls)
            obj.params = init_params
        else:
            init_params = (np.random.rand(6) - 0.5) * scale
            obj = get_free_fermion_gate(init_params).view(cls)
            obj.params = init_params

        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.params = getattr(obj, 'params', None)
        # ---- class specific attributes ----
        self.num_params = 6  # Number of parameters for the free fermion gate
        self.get_unitary_from = get_free_fermion_gate
        self.get_gradient_from = gradient_from_free_fermion_h_params

class FermionicGate(UnitaryGate):
    """
    Generic fermionic gate class.

    A fermionic gate is defined as a unitary gate of the form:
    G(A, B) = [[p, 0, 0, q],
               [0, w, x, 0],
               [0, y, z, 0],
               [r, 0, 0, s]]
    where A = [[p, q],  and B = [[w, x],
               [r, s]]           [y, z]]
    A and B are both U(2) matrices.

    - Note that a fermionic gate can break the U(1) symmetry.
    """

    def __new__(cls, init_U=None, init_params=None, scale=1e-3):
        """
        Create a new FermionicGate instance.

        Parameters
        ----------
        init_U : np.ndarray, optional
            Initial unitary matrix (default is None).
        init_params : np.ndarray, optional
            Initial parameters (default is None).
        scale : float, optional
            Scale factor for random initialization (default is 1e-3).

        Returns
        -------
        FermionicGate
            A new instance of the FermionicGate class.
        """
        if init_U is not None:
            obj = np.asarray(init_U).view(cls)
            assert obj.shape == (4, 4), "Unitary matrix must be 4x4."
            assert np.allclose(np.dot(obj, obj.conj().T), np.eye(4)), "Matrix must be unitary."
            obj.params = get_fermionic_unitary_params(init_U)
        elif init_params is not None:
            assert len(init_params) == 8, "Parameters must be a vector of length 8."
            obj = get_fermionic_gate(init_params).view(cls)
            obj.params = init_params
        else:
            init_params = (np.random.rand(8) - 0.5) * scale
            obj = get_fermionic_gate(init_params).view(cls)
            obj.params = init_params

        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.params = getattr(obj, 'params', None)
        # ---- class specific attributes ----
        self.num_params = 8  # Number of parameters for the fermionic gate
        self.get_unitary_from = get_fermionic_gate
        self.get_gradient_from = gradient_from_fermionic_h_params

class DiagonalUnitaryGate(UnitaryGate):
    """
    Diagonal Unitary Gate class.

    A diagonal gate is defined as a unitary gate of the form:
    U = [[a, 0, 0, 0],
         [0, b, 0, 0],
         [0, 0, c, 0],
         [0, 0, 0, d]]
    where a, b, c, d are phases.
    """

    def __new__(cls, init_U=None, init_params=None, scale=1e-3):
        """
        Create a new DiagonalUnitaryGate instance.

        Parameters
        ----------
        init_U : np.ndarray, optional
            Initial unitary matrix (default is None).
        init_params : np.ndarray, optional
            Initial parameters (default is None).
        scale : float, optional
            Scale factor for random initialization (default is 1e-3).

        Returns
        -------
        DiagonalUnitaryGate
            A new instance of the DiagonalUnitaryGate class.
        """
        if init_U is not None:
            obj = np.asarray(init_U).view(cls)
            assert obj.shape == (4, 4), "Unitary matrix must be 4x4."
            assert np.allclose(np.dot(obj, obj.conj().T), np.eye(4)), "Matrix must be unitary."
            obj.params = get_diagonal_unitary_params(init_U)
        elif init_params is not None:
            assert len(init_params) == 4, "Parameters must be a vector of length 4."
            obj = get_diagonal_unitary_gate(init_params).view(cls)
            obj.params = init_params
        else:
            init_params = (np.random.rand(4) - 0.5) * scale
            obj = get_diagonal_unitary_gate(init_params).view(cls)
            obj.params = init_params

        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.params = getattr(obj, 'params', None)
        # ---- class specific attributes ----
        self.num_params = 4
        self.get_unitary_from = get_diagonal_unitary_gate
        self.get_gradient_from = gradient_from_diagonal_h_params


def get_unitary_gate(h_params):
    """
    Compute the unitary gate U = exp(-i H) from the Hamiltonian matrix

    Parameters
    ----------
    H_mat : np.array
        Hamiltonian matrix

    Returns
    -------
    U : np.array
        Unitary gate
    """
    H_mat_real = np.array([[h_params[0], h_params[1], h_params[2], h_params[3]],
                           [h_params[1], h_params[4], h_params[5], h_params[6]],
                           [h_params[2], h_params[5], h_params[7], h_params[8]],
                           [h_params[3], h_params[6], h_params[8], h_params[9]]])
    H_mat_imag = np.array([[0, 1j*h_params[10], 1j*h_params[11], 1j*h_params[12]],
                           [-1j*h_params[10], 0, 1j*h_params[13], 1j*h_params[14]],
                           [-1j*h_params[11], -1j*h_params[13], 0, 1j*h_params[15]],
                           [-1j*h_params[12], -1j*h_params[14], -1j*h_params[15], 0]])
    H_mat = H_mat_real + H_mat_imag
    return scipy.linalg.expm(-1j * H_mat)

def get_unitary_params(unitary):
    """
    Compute the vector of parameters from the unitary U.
    We first find out the Hermitian matrix H_mat
    defining the unitary U = exp(-i H).
    We want to map the Hermitian matrix to a vector of real-valued parameters.

    Parameters
    ----------
    unitary : np.array
        an unitary matrix

    Returns
    -------
    params : np.array
        Vector of parameters
    """
    H_mat = scipy.linalg.logm(unitary) / (-1.j)
    # concatenate the real and imaginary parts of the upper triangular part of the matrix
    return np.concatenate([H_mat[np.triu_indices(4, k=0)].real,
                           H_mat[np.triu_indices(4, k=1)].imag])

def get_U1_unitary_gate(h_params):
    """
    Compute the unitary gate U = exp(-i H) from the Hamiltonian matrix

    Parameters
    ----------
    h_params : np.array

    Returns
    -------
    U : np.array
        Unitary gate
    """
    H_mat_real = np.array([[h_params[0], 0, 0, 0],
                           [0, h_params[1], h_params[2], 0],
                           [0, h_params[2], h_params[3], 0],
                           [0, 0, 0, h_params[4]]])
    H_mat_imag = np.array([[0, 0, 0, 0],
                           [0, 0, 1j*h_params[5], 0],
                           [0, -1j*h_params[5], 0, 0],
                           [0, 0, 0, 0]])
    H_mat = H_mat_real + H_mat_imag
    return scipy.linalg.expm(-1j * H_mat)

def get_free_fermion_gate(h_params):
    """
    Compute the free fermion gate U = exp(-i H) from the Hamiltonian matrix
    Both the sub-blocks H_A and H_B are hermitian and traceless.

    Parameters
    ----------
    h_params : np.array

    Returns
    -------
    U : np.array
        Unitary gate
    """
    H_mat_real = np.array([[h_params[0], 0, 0, h_params[1]],
                           [0, h_params[2], h_params[3], 0],
                           [0, h_params[3], -h_params[2], 0],
                           [h_params[1], 0, 0, -h_params[0]]])
    H_mat_imag = np.array([[0, 0, 0, 1j*h_params[4]],
                           [0, 0, 1j*h_params[5], 0],
                           [0, -1j*h_params[5], 0, 0],
                           [-1j*h_params[4], 0, 0, 0]])
    H_mat = H_mat_real + H_mat_imag
    return scipy.linalg.expm(-1j * H_mat)

def get_fermionic_gate(h_params):
    """
    Compute the fermionic gate U = exp(-i H) from the Hamiltonian matrix

    Parameters
    ----------
    h_params : np.array

    Returns
    -------
    U : np.array
        Unitary gate
    """
    H_mat_real = np.array([[h_params[0], 0, 0, h_params[1]],
                           [0, h_params[2], h_params[3], 0],
                           [0, h_params[3], h_params[4], 0],
                           [h_params[1], 0, 0, h_params[5]]])
    H_mat_imag = np.array([[0, 0, 0, 1j*h_params[6]],
                           [0, 0, 1j*h_params[7], 0],
                           [0, -1j*h_params[7], 0, 0],
                           [-1j*h_params[6], 0, 0, 0]])
    H_mat = H_mat_real + H_mat_imag
    return scipy.linalg.expm(-1j * H_mat)

def get_U1_unitary_params(unitary):
    """
    Compute the vector of parameters from the unitary U.
    We first find out the Hermitian matrix H_mat
    defining the unitary U = exp(-i H).
    We want to map the Hermitian matrix to a vector of real-valued parameters.

    Parameters
    ----------
    unitary : np.array
        an unitary matrix

    Returns
    -------
    params : np.array
        Vector of parameters
    """
    H_mat = scipy.linalg.logm(unitary) / (-1.j)
    return np.array([H_mat[0, 0].real, H_mat[1, 1].real, H_mat[1, 2].real,
                     H_mat[2, 2].real, H_mat[3, 3].real, H_mat[1, 2].imag])

def get_free_fermion_unitary_params(unitary):
    """
    Compute the vector of parameters from the free fermion unitary U.
    A free fermion gate is defined as a unitary gate of the form:
    G(A, B) = [[p, 0, 0, q],
               [0, w, x, 0],
               [0, y, z, 0],
               [r, 0, 0, s]]
    where A = [[p, q],  and B = [[w, x],
               [r, s]]           [y, z]]
    The free fermion condition is that
    A and B are either
    (i) both in SU(2) or
    (ii) both in U(2) with the same determinant.

    Parameters
    ----------
    unitary : np.array
        an unitary matrix

    Returns
    -------
    params : np.array
        Vector of parameters
    """
    H_mat = scipy.linalg.logm(unitary) / (-1.j)
    return np.array([H_mat[0, 0].real, H_mat[0, 3].real, H_mat[1, 1].real,
                     H_mat[1, 2].real, H_mat[0, 3].imag, H_mat[1, 2].imag])

def get_fermionic_unitary_params(unitary):
    """
    Compute the vector of parameters from the fermionic unitary U.
    A fermionic gate is defined as a unitary gate of the form:
    G(A, B) = [[p, 0, 0, q],
               [0, w, x, 0],
               [0, y, z, 0],
               [r, 0, 0, s]]

    Parameters
    ----------
    unitary : np.array
        an unitary matrix

    Returns
    -------
    params : np.array
        Vector of parameters
    """
    H_mat = scipy.linalg.logm(unitary) / (-1.j)
    return np.array([H_mat[0, 0].real, H_mat[0, 3].real, H_mat[1, 1].real,
                     H_mat[1, 2].real, H_mat[2, 2].real, H_mat[3, 3].real,
                     H_mat[0, 3].imag, H_mat[1, 2].imag])

def get_diagonal_unitary_gate(h_params):
    """
    Compute the diagonal unitary gate U = exp(-i H) from the Hamiltonian matrix

    Parameters
    ----------
    h_params : np.array

    Returns
    -------
    U : np.array
        Unitary gate
    """
    H_mat_real = np.array([[h_params[0], 0, 0, 0],
                           [0, h_params[1], 0, 0],
                           [0, 0, h_params[2], 0],
                           [0, 0, 0, h_params[3]]])
    return scipy.linalg.expm(-1j * H_mat_real)

def get_diagonal_unitary_params(unitary):
    """
    Compute the vector of parameters from the diagonal unitary U.
    A diagonal gate is defined as a unitary gate of the form:
    U = [[a, 0, 0, 0],
         [0, b, 0, 0],
         [0, 0, c, 0],
         [0, 0, 0, d]]
    where a, b, c, d are phases.

    Parameters
    ----------
    unitary : np.array
        an unitary matrix

    Returns
    -------
    params : np.array
        Vector of parameters
    """
    H_mat = scipy.linalg.logm(unitary) / (-1.j)
    return np.array([H_mat[0, 0].real, H_mat[1, 1].real,
                     H_mat[2, 2].real, H_mat[3, 3].real])

# -----------------------------------------------------------------------------
@jax.jit
def jax_get_unitary_gate(h_params):
    """
    Same as get_unitary_gate but using jax.numpy
    """
    H_mat_real = jnp.array([[h_params[0], h_params[1], h_params[2], h_params[3]],
                            [h_params[1], h_params[4], h_params[5], h_params[6]],
                            [h_params[2], h_params[5], h_params[7], h_params[8]],
                            [h_params[3], h_params[6], h_params[8], h_params[9]]])
    H_mat_imag = jnp.array([[0, 1j*h_params[10], 1j*h_params[11], 1j*h_params[12]],
                            [-1j*h_params[10], 0, 1j*h_params[13], 1j*h_params[14]],
                            [-1j*h_params[11], -1j*h_params[13], 0, 1j*h_params[15]],
                            [-1j*h_params[12], -1j*h_params[14], -1j*h_params[15], 0]])
    H_mat = H_mat_real + H_mat_imag
    return jax_expm(-1j * H_mat)

@jax.jit
def func_val_from_h_params(params, dU_mat):
    U = jax_get_unitary_gate(params)
    Ud = U.T.conj()
    return jnp.tensordot(dU_mat, Ud, axes=2).real

@jax.jit
def gradient_from_h_params(params, dU_mat):
    return jax.grad(func_val_from_h_params, 0)(params, dU_mat)

# -----------------------------------------------------------------------------
@jax.jit
def jax_get_U1_unitary_gate(h_params):
    """
    Same as get_U1_unitary_gate but using jax.numpy
    """
    H_mat_real = jnp.array([[h_params[0], 0, 0, 0],
                            [0, h_params[1], h_params[2], 0],
                            [0, h_params[2], h_params[3], 0],
                            [0, 0, 0, h_params[4]]])
    H_mat_imag = jnp.array([[0, 0, 0, 0],
                            [0, 0, 1j*h_params[5], 0],
                            [0, -1j*h_params[5], 0, 0],
                            [0, 0, 0, 0]])
    H_mat = H_mat_real + H_mat_imag
    return jax_expm(-1j * H_mat)

@jax.jit
def func_val_from_U1_h_params(params, dU_mat):
    U = jax_get_U1_unitary_gate(params)
    Ud = U.T.conj()
    return jnp.tensordot(dU_mat, Ud, axes=2).real

@jax.jit
def gradient_from_U1_h_params(params, dU_mat):
    return jax.grad(func_val_from_U1_h_params, 0)(params, dU_mat)

# -----------------------------------------------------------------------------
@jax.jit
def jax_get_diagonal_unitary_gate(h_params):
    """
    Same as get_diagonal_unitary_gate but using jax.numpy
    """
    H_mat_real = jnp.array([[h_params[0], 0, 0, 0],
                            [0, h_params[1], 0, 0],
                            [0, 0, h_params[2], 0],
                            [0, 0, 0, h_params[3]]])
    return jax_expm(-1j * H_mat_real)

@jax.jit
def func_val_from_diagonal_h_params(params, dU_mat):
    U = jax_get_diagonal_unitary_gate(params)
    Ud = U.T.conj()
    return jnp.tensordot(dU_mat, Ud, axes=2).real

@jax.jit
def gradient_from_diagonal_h_params(params, dU_mat):
    return jax.grad(func_val_from_diagonal_h_params, 0)(params, dU_mat)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
@jax.jit
def jax_get_free_fermion_gate(h_params):
    """
    Same as get_free_fermion_gate but using jax.numpy
    """
    H_mat_real = jnp.array([[h_params[0], 0, 0, h_params[1]],
                            [0, h_params[2], h_params[3], 0],
                            [0, h_params[3], -h_params[2], 0],
                            [h_params[1], 0, 0, -h_params[0]]])
    H_mat_imag = jnp.array([[0, 0, 0, 1j*h_params[4]],
                            [0, 0, 1j*h_params[5], 0],
                            [0, -1j*h_params[5], 0, 0],
                            [-1j*h_params[4], 0, 0, 0]])
    H_mat = H_mat_real + H_mat_imag
    return jax_expm(-1j * H_mat)

@jax.jit
def func_val_from_free_fermion_h_params(params, dU_mat):
    U = jax_get_free_fermion_gate(params)
    Ud = U.T.conj()
    return jnp.tensordot(dU_mat, Ud, axes=2).real

@jax.jit
def jax_gradient_from_free_fermion_h_params(params, dU_mat):
    return jax.grad(func_val_from_free_fermion_h_params, 0)(params, dU_mat)

def gradient_from_free_fermion_h_params(params, dU_mat):
    """
    Compute the gradient of the free fermion unitary gate with respect to the parameters.

    Parameters
    ----------
    params : np.ndarray
        Parameters of the free fermion unitary gate.
    dU_mat : np.ndarray
        Matrix to compute the gradient with respect to.

    Returns
    -------
    np.ndarray
        Gradient of the free fermion unitary gate.
    """
    return np.array(jax_gradient_from_free_fermion_h_params(params, dU_mat))

# -----------------------------------------------------------------------------
@jax.jit
def jax_get_fermionic_gate(h_params):
    """
    Same as get_fermionic_gate but using jax.numpy
    """
    H_mat_real = jnp.array([[h_params[0], 0, 0, h_params[1]],
                            [0, h_params[2], h_params[3], 0],
                            [0, h_params[3], h_params[4], 0],
                            [h_params[1], 0, 0, h_params[5]]])
    H_mat_imag = jnp.array([[0, 0, 0, 1j*h_params[6]],
                            [0, 0, 1j*h_params[7], 0],
                            [0, -1j*h_params[7], 0, 0],
                            [-1j*h_params[6], 0, 0, 0]])
    H_mat = H_mat_real + H_mat_imag
    return jax_expm(-1j * H_mat)

@jax.jit
def func_val_from_fermionic_h_params(params, dU_mat):
    U = jax_get_fermionic_gate(params)
    Ud = U.T.conj()
    return jnp.tensordot(dU_mat, Ud, axes=2).real

@jax.jit
def jax_gradient_from_fermionic_h_params(params, dU_mat):
    return jax.grad(func_val_from_fermionic_h_params, 0)(params, dU_mat)

def gradient_from_fermionic_h_params(params, dU_mat):
    """
    Compute the gradient of the fermionic unitary gate with respect to the parameters.

    Parameters
    ----------
    params : np.ndarray
        Parameters of the fermionic unitary gate.
    dU_mat : np.ndarray
        Matrix to compute the gradient with respect to.

    Returns
    -------
    np.ndarray
        Gradient of the fermionic unitary gate.
    """
    return np.array(jax_gradient_from_fermionic_h_params(params, dU_mat))
