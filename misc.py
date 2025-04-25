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


