import numpy as np
import scipy
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.linalg import expm as jax_expm

"""
gate.py:
	Classes:
		1. Generic SU(4) gate -> 16 parameters including phase
		2. U1 preserving gate -> 6 parameters including phase
		3. Free fermion gate
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
        assert len(params) == 16, "Parameters must be a vector of length 16."
        self.params = params
        self[:] = get_unitary_gate(params)

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
        return np.array(gradient_from_h_params(self.params, dU_mat))

class U1UnitaryGate(np.ndarray):
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

    def get_parameters(self):
        """
        Get the parameters of the U1 unitary gate.

        Returns
        -------
        np.ndarray
            Parameters of the U1 unitary gate.
        """
        return self.params

    def set_parameters(self, params):
        """
        Set the parameters of the U1 unitary gate.

        Parameters
        ----------
        params : np.ndarray
            New parameters for the U1 unitary gate.
        """
        assert len(params) == 6, "Parameters must be a vector of length 6."
        self.params = params
        self[:] = get_U1_unitary_gate(params)

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
        return np.array(gradient_from_U1_h_params(self.params, dU_mat))


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
    H_mat : np.array
        Hamiltonian matrix

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

def energy_from_h_params(params, dU_mat):
    U = jax_get_unitary_gate(params)
    Ud = U.T.conj()
    return jnp.tensordot(dU_mat, Ud, axes=2).real

def gradient_from_h_params(params, dU_mat):
    return jax.grad(energy_from_h_params, 0)(params, dU_mat)

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
def energy_from_U1_h_params(params, dU_mat):
    U = jax_get_U1_unitary_gate(params)
    Ud = U.T.conj()
    return jnp.tensordot(dU_mat, Ud, axes=2).real

@jax.jit
def gradient_from_U1_h_params(params, dU_mat):
    return jax.grad(energy_from_U1_h_params, 0)(params, dU_mat)
