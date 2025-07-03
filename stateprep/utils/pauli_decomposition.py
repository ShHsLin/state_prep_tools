"""
Module Name or Brief Description

Copyright (c) <Year> <Author or Organization>
Licensed under the <License Name> (see LICENSE file for details)

This code is adapted from previous project with
[Adam](https://sites.google.com/view/adamsmith-cmtheory)
"""

import numpy as np
import scipy.linalg

X = np.array([[0, 1.], [1., 0.]])
Y = np.array([[0., -1j], [1j, 0.]])
Z = np.array([[1., 0.], [0., -1.]])
I = np.eye(2)

SU2_basis = [I, X, Y, Z]

SU4_basis = []
for Op1 in [I, X, Y, Z]:
    for Op2 in [I, X, Y, Z]:
        SU4_basis.append(np.kron(Op1, Op2))


def decomp_1(U):
    log_U = scipy.linalg.logm(U)
    # print("log(U) = \n", log_U)
    c_i = np.trace(I.T.conj().dot(log_U)) / 2j
    c_x = np.trace(X.T.conj().dot(log_U)) / 2j
    c_y = np.trace(Y.T.conj().dot(log_U)) / 2j
    c_z = np.trace(Z.T.conj().dot(log_U)) / 2j
    coeffs = c_i, c_x, c_y, c_z
    assert np.allclose(np.imag(coeffs), 0)
    return np.real(coeffs)

def decomp_2(U):
    log_U = scipy.linalg.logm(U.reshape([4, 4]))
    # print("log(U) = \n", log_U)
    coeffs = []
    for basis in SU4_basis:
        coeffs.append(np.trace(basis.T.conj().dot(log_U)) / 4j)

    assert np.allclose(np.imag(coeffs), 0)
    return np.real(coeffs)

def restore_1(coeffs):
    if len(coeffs) == 3:
        g = 1j * (coeffs[0] * X + coeffs[1] * Y + coeffs[2] * Z)
    elif len(coeffs) == 4:
        g = 1j * (coeffs[0] * I + coeffs[1] * X + coeffs[2] * Y + coeffs[3] * Z)
    else:
        raise ValueError("Coefficients should have 3 or 4 elements.")
                              
    # print("restore log U = \n", g)
    return scipy.linalg.expm(g)

def restore_2(coeffs):
    if len(coeffs) == 15:
        g = 1j * np.sum([coeffs[i] * SU4_basis[i+1] for i in range(len(coeffs))], axis=0)
    elif len(coeffs) == 16:
        g = 1j * np.sum([coeffs[i] * SU4_basis[i] for i in range(len(coeffs))], axis=0)
    else:
        raise ValueError("Coefficients should have 15 or 16 elements.")

    # print("restore log U = \n", g)
    return scipy.linalg.expm(g)

