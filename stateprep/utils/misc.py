import numpy as np
import scipy
import scipy.linalg

def svd(theta, compute_uv=True, full_matrices=True):
    """
    Performs a Singular Value Decomposition avoiding possible errors.

    Parameters:
    ----------
    matrix: array_like
        The matrix to which we perform the SVD. Shape (M, N).

    Returns:
    --------
    U: array_like
        Unitary matrix having left singular vectors as columns with shape (M, K).
    S: array_like
        The singular values, sorted in non-increasing order. Of shape (K,), with K = min(M, N).
    V: array_like
        Unitary matrix having right singular vectors as rows. Of shape (K, N).
    """
    try:
        U, S, Vd = scipy.linalg.svd(theta, compute_uv=compute_uv,
                                    full_matrices=full_matrices,
                                    lapack_driver='gesdd',
                                    check_finite=True
                                    )
        check1 = np.sum(U)
        check2 = np.sum(Vd)

        if np.isnan(check1) or np.isnan(check2):
            print("*gesdd*")
            raise np.linalg.LinAlgError

    except np.linalg.LinAlgError:
        print("*gesvd*", "Using generic SVD")
        U, S, Vd = scipy.linalg.svd(theta,
                                    compute_uv=compute_uv,
                                    full_matrices=full_matrices,
                                    lapack_driver='gesvd',
                                    check_finite=True,
                                    )

    return U, S, Vd

def get_random_u1_2q_gate(using_complex=True, scale=1e-4):
    """
    Generate a random U1 2-qubit unitary gate.
   """
    if using_complex:
        M = np.random.rand(2, 2) + 1j * np.random.rand(2, 2) - 0.5 - 0.5j
    else:
        M = np.random.rand(2, 2) - 0.5

    M = M * scale + np.eye(2)
    Q, R = np.linalg.qr(M)

    if using_complex:
        two_qubit_unitary = np.zeros((4, 4), dtype=np.complex128)
    else:
        two_qubit_unitary = np.zeros((4, 4), dtype=np.float64)

    two_qubit_unitary[0, 0] = 1.
    two_qubit_unitary[1:3, 1:3] = Q

    if using_complex:
        theta = np.random.rand(1) * np.pi * 2 * scale
        phase = np.exp(1j * theta)
        two_qubit_unitary[3, 3] = phase.item() 
    else:
        two_qubit_unitary[3, 3] = 1.

    return two_qubit_unitary

def get_random_2q_gate(using_complex=True, scale=1e-4):
    """
    Generate a random 2-qubit unitary gate.
   """
    if using_complex:
        M = np.random.rand(4, 4) + 1j * np.random.rand(4, 4) - 0.5 - 0.5j
    else:
        M = np.random.rand(4, 4) - 0.5

    M = M * scale + np.eye(4)
    Q, R = np.linalg.qr(M)
    return Q

def gen_vec_from_dict(string_coeff_dict, using_complex=True):
    """
    Generate a vector from a dictionary of string coefficients.
    The keys of the dictionary are strings representing the states,
    and the values are the coefficients for those states.
    """
    one_key = next(iter(string_coeff_dict))
    L = len(one_key)
    if using_complex:
        vec = np.zeros(2**L, dtype=np.complex128)
    else:
        vec = np.zeros(2**L, dtype=np.float64)

    for key, coeff in string_coeff_dict.items():
        index = int(key, 2)
        vec[index] = coeff

    return vec

def get_env(top_state, bottom_state, indices, num_qubits):
    """
    Generate the environment tensor for the given top and bottom state vectors
    based on the specified indices.
    Parameters:
    ----------
    top_state: np.ndarray
        The state vector of the top tensor. (without complex conjugation)
    bottom_state: np.ndarray
        The state vector of the bottom tensor. (without complex conjugation)
    indices: tuple
        A tuple containing the indices that left open.
    num_qubits: int

    Returns:
    -------
    env: np.ndarray
        <top | bottom> with (i'j',ij) indices not contracted.
    """
    idx0, idx1 = indices
    if idx0 > idx1:
        swap_indices = True
        idx0, idx1 = idx1, idx0
    else:
        swap_indices = False

    top_theta = np.reshape(top_state,
                           [(2**idx0), 2, 2**(idx1-idx0-1), 2, 2**(num_qubits-(idx1+1))])
    bottom_theta = np.reshape(bottom_state,
                              [(2**idx0), 2, 2**(idx1-idx0-1), 2, 2**(num_qubits-(idx1+1))])
    # [left, i, mid, j, right]
    env = np.tensordot(top_theta.conj(), bottom_theta, axes=([0, 2, 4], [0, 2, 4]))
    if swap_indices:
        env = np.transpose(env, [1, 0, 3, 2])

    return env
