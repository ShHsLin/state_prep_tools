"""
Module Name or Brief Description

Copyright (c) <Year> <Author or Organization>
Licensed under the <License Name> (see LICENSE file for details)

This code is adapted from previous project with
[Adam](https://sites.google.com/view/adamsmith-cmtheory)
"""

#from qiskit import QuantumCircuit  # uncomment if we are using add_2q_unitary

import numpy as np
import scipy.linalg as linalg
import scipy as sc

# magic matrix
#M = (1/np.sqrt(2))*np.array([[1,0,0,1j],[0,1j,1,0],[0,1j,-1,0],[1,0,0,-1j]])
M = (1/np.sqrt(2))*np.array([[1,1j,0,0],[0,0,1j,1],[0,0,1j,-1],[1,-1j,0,0]])


def haar_measure(n):
    # A Random unitary matrix distributed with Haar measure
    z = np.random.randn(n,n) + 1j*np.random.randn(n,n)
    q,r = linalg.qr(z)
    d = np.diagonal(r)
    ph = d/np.abs(d)
    # q = sc.multiply(q,ph,q)
    q = q @ np.diag(ph) @ q.T.conj()
    return q


def add_2q_unitary(U, q1, q2, qc, qr):
    """Decompose the unitary U = (A1xA2)N(B1xB2) and add to the quantum circuit qc, actng on qubits qr[q1] and qr[q2]
    This is for use with Qiskit!
    """

    A1, A2, B1, B2, N, N_diag = two_qubit_decomp(U)

    # remember reverse order of gates to matrix multiplication
    theta, phi, lmbda, phase = extract_1q_params(B1)
    qc.u(theta, phi, lmbda, qr[q1])
    theta, phi, lmbda, phase = extract_1q_params(B2)
    qc.u(theta, phi, lmbda, qr[q2])

    alpha, beta, gamma, k0 = extract_2q_params(N_diag)
    addN(alpha, beta, gamma,q1,q2,qc,qr)

    theta, phi, lmbda, phase = extract_1q_params(A1)
    qc.u(theta, phi, lmbda, qr[q1])
    theta, phi, lmbda, phase = extract_1q_params(A2)
    qc.u(theta, phi, lmbda, qr[q2])

    return


def addN(alpha,beta,gamma,q1,q2,qc,qr):

    qc.rz(-np.pi/2, qr[q1])
    qc.cx(qr[q1], qr[q2])
    qc.ry(2*alpha - np.pi/2, qr[q1])
    qc.rz(np.pi/2 - 2*gamma, qr[q2])
    qc.cx(qr[q2], qr[q1])
    qc.ry(np.pi/2 - 2*beta, qr[q1])
    qc.cx(qr[q1], qr[q2])
    qc.rz(np.pi/2, qr[q2])


def two_qubit_decomp(U):
    """Decomposes a 4x4 unitary matrix into the form (A1xA2)N(B1xB2), where N = expm(-1j(k0 + kx(XxX) + ky(YxY) + kz(ZxZ))), and A1,A2,B1,B2 are U(2).

    n.b. A1,A2,B1,B2 can be made SU(2) by moving the phase from A1 to A2, and similarly for B1, B2. Since we drop phases anyway, this is not necessary.
    """

    U_prime = M.conj().T.dot(U).dot(M)
    Q_L, Q_R, N_diag = KAK_decomp(U_prime)  # Cartan decomposition

    N = M.dot(N_diag).dot(M.conj().T)

    A1, A2 = KPSVD(M.dot(Q_L.dot(M.conj().T)),2,2,2,2)  # splits Q_L into Kronecker product of A1xA2
    B1, B2 = KPSVD(M.dot(Q_R.T.dot(M.conj().T)),2,2,2,2) # splits Q_R.T into Kronecker product of B1xB2

    return A1, A2, B1, B2, N, N_diag


def KAK_decomp(U):
    """Cartan decomposition: Decomposes a 4x4 unitary matrix U into the form QL.D.QR, where QL, QR are in SO(4) and D is a complex diagonal matrix
    """

    # real and imaginary parts of U
    U_r = np.real((U + U.conj())/2)
    U_i = np.real((U - U.conj())/(2j))

    # simultaneous SVD of the commuting real/imag parts
    Q_L, D_r, D_i, Q_R = simultaneous_SVD(U_r,U_i)

    # Check that Q_L and Q_R are in SO(4)
    if np.max(Q_L.T.dot(Q_L) - np.identity(4)) > 10**(-8) or np.max(Q_L.dot(Q_L.T) - np.identity(4)) > 10**(-8):
        raise ValueError("Q_L not orthogonal")
    if np.max(Q_R.T.dot(Q_R) - np.identity(4)) > 10**(-8) or np.max(Q_R.dot(Q_R.T) - np.identity(4)) > 10**(-8):
        raise ValueError("Q_R not orthogonal")
    if np.abs(linalg.det(Q_L) - 1) > 10**(-8):
        print(linalg.det(Q_L))
        raise ValueError("Q_L not special")
    if np.abs(linalg.det(Q_R) - 1) > 10**(-8):
        print(linalg.det(Q_L))
        raise ValueError("Q_R not special")

    N_diag = D_r + 1j*D_i  # U = Q_L N_diag Q_R.T

    # check the decompostition
    if np.max(U - Q_L.dot(N_diag).dot(Q_R.T)) > 10**(-8):
        print("KAK error: {}".format(np.max(U - Q_L.dot(N_diag).dot(Q_R.T))))
        raise ValueError("KAK decomposition failed!")

    return Q_L, Q_R, N_diag


def KPSVD(A, m1=2, n1=2, m2=2, n2=2):
    """Kronecker Product SVD: This code splits a matrix A (assumed to be of Kronecker product form) into a Kronecker product, A = BxC. This code will fail if A is not in Kronecker product form.

    n.b. this code works more generally than the 4x4 -> (2x2) x (2x2) case!
    """

    # check Kronecker product is dimensionally possible
    m, n = A.shape
    if (m != m1*m2 or n != n1*n2):
        raise ValueError("dimensions don't match")

    block_A = [[0]*m1]

    # construct R matrix that reshapes A such that it is an outer product of vectors, rather than a Kronecker product of matrices
    R = np.zeros((m1*n1,m2*n2), dtype=complex)
    for nn in range(n1):
        for mm in range(m1):
            index = mm + m1*nn
            A_block = A[mm*m2:(mm+1)*m2,nn*n2:(nn+1)*n2]
            R[index,:] = np.reshape(A_block,-1,order='F')


    U, S, V = linalg.svd(R)
    V = V.T

    # second check that matrix is a Kronecker product
    if S[1] > 10**(-8):
        print("Matrix is not a Kronecker product!! S: {}".format(S))

    # U[:,0] and V[:,0] are the vectorized matrices B and C
    B = np.sqrt(S[0])*np.reshape(U[:,0],(m1,n1),order='F')
    C = np.sqrt(S[0])*np.reshape(V[:,0],(m2,n2),order='F')

    return B, C


def simultaneous_SVD(A,B):
    """Perform the simultaneous SVD of matrices A and B, i.e., find matrices U, D_A, D_B, V, such that A = U D_A V', B = U D_B V'.

    n.b. this code may fail if the matrix with higher rank has degenerate singular values.
    n.b. both input matrices should always be full rank if they came from the KAK decomposition of a unitary matrix!
    """

    # perform SVD on highest rank matrix first
    swapped = False
    rankA = np.linalg.matrix_rank(A)
    rankB = np.linalg.matrix_rank(B)
    if rankB > rankA :
        temp_mat = A
        A = B
        B = temp_mat
        swapped = True
        print("Simultaneous SVD swap")

    # SVD of higher rank matrix
    U, D_A, V = linalg.svd(A)
    D_A = np.diag(D_A)
    V = V.conj().T

    # U and V should also be a SVD for B
    D_B = U.conj().T.dot(B).dot(V)

    # in case of repeated eigenvalues in D_A!
    D_B = (D_B + D_B.T)/2
    D_B, P = linalg.eigh(D_B)
    D_B = np.diag(D_B)
    V = V.dot(P)
    U = U.dot(P)
    #D_A = U.conj().T.dot(A).dot(V)
    D_A = P.conj().T.dot(D_A).dot(P)

    # check that U and V correctly diagonalize A
    if np.max(np.abs(D_A - np.diag(np.diag(D_A)))) > 10**(-10):
        print(D_A)
        print(D_B)
        raise ValueError("Simultaneous SVD fail! D_A")

    # check that U and V correctly diagonalize B
    if np.max(np.abs(D_B - np.diag(np.diag(D_B)))) > 10**(-10):
        print(D_A)
        print(D_B)
        raise ValueError("Simultaneous SVD fail! D_B")

    D_A = np.diag(np.diag(D_A))
    D_B = np.diag(np.diag(D_B))


    detU = linalg.det(U)
    detV = linalg.det(V)

    # IMPORTANT since we need SO(4)!!!
    if detU < 0:
        U[:,0] = -U[:,0]
        D_A[0] = -D_A[0]
        D_B[0] = -D_B[0]
    if detV < 0:
        V[:,0] = -V[:,0]
        D_A[0] = -D_A[0]
        D_B[0] = -D_B[0]

    # swap back if we swapped at the beginning
    if swapped:
        tempD = D_A
        D_A = D_B
        D_B = tempD

    return U, D_A, D_B, V

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

def extract_1q_params(U):
    """U = phase*( cos(theta/2)             -e^{i lmbda}sin(theta/2)   )
                 ( e^{i phi}sin(theta/2)  e^{i(lmbda+phi)}cos(theta/2) )
    """


    phase = U[0,0]/np.abs(U[0,0])
    U = U/phase
    u00_clamped = clamp(U[0,0], -1.0, 1.0)
    theta = 2*np.arccos(np.abs(u00_clamped))
    lmbda = np.angle(-U[0,1])
    phi = np.angle(U[1,0])

    """
    theta = 2*np.arccos(np.abs(U[0,0]))
    phi = np.angle(-U[1,0]*U[1,1]/(U[0,1]*U[0,0]))/2
    lmbda = np.angle(-U[0,1]*U[1,1]/(U[1,0]*U[0,0]))/2

    print(np.sin(theta/2))

    if -U[0,1]/np.sin(theta/2)/np.exp(1j*lmbda) < 0:
        lmbda = lmbda + np.pi
    if U[1,0]/np.sin(theta/2)/np.exp(1j*phi) < 0 :
        phi = phi + np.pi
    """

    return theta, phi, lmbda, phase

def extract_2q_params(N_diag):
    """N = expm(1j*(k0 + alpha XxX + beta YxY + gamma ZxZ))
    """

    angles = np.angle(np.diag(N_diag))

    Gamma = np.array([[1,1,1,1],[1,-1,1,-1],[-1,1,1,-1],[1,1,-1,-1]])/4
    vec = Gamma.dot(angles)
    k0 = vec[0]
    alpha = vec[1]
    beta = vec[2]
    gamma = vec[3]

    return alpha, beta, gamma, k0


if __name__ == "__main__":

    #test decomposition for 1000 random matrices.
    for ii in range(1000):

        # without charge conservation
        U = haar_measure(4)

        # with charge conservation
        #U = linalg.block_diag(np.exp(1j*2*np.pi*sc.rand(1,1)),haar_measure(2),np.exp(1j*2*np.pi*sc.rand(1,1)))

        # the matrices from the decomposition
        A1, A2, B1, B2, N, N_diag = two_qubit_decomp(U)

        # the reconstructed matrix
        U_prime = np.kron(A1,A2).dot(N).dot(np.kron(B1,B2))

        if np.abs(np.max(U-U_prime)) > 10**(-12):
            raise ValueError("It Failed!!")

    print("It worked!!")
