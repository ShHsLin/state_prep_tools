import pickle
import numpy as np
import scipy
import sys
np.set_printoptions(precision=3, suppress=True, linewidth=200)
np.random.seed(0)

from common_setup import *

if __name__ == "__main__":
    coefficients = np.random.rand(6)
    coefficients[0] = -coefficients[-3] - coefficients[-2] - coefficients[-1]
    list_of_operators = np.array([np.eye(4), hop, current, ZZ, Z1, Z2])
    H_reconstructed = np.tensordot(coefficients, list_of_operators, [[0], [0]])

    U_reconstructed = scipy.linalg.expm(-1j * H_reconstructed)
    print(U_reconstructed)

    decomp_coeff = np.random.rand(6)
    def seq_U(decomp_coeff):
        list_of_operators = np.array([np.eye(4), hop, current, ZZ, Z1, Z2])
        U = scipy.linalg.expm(-1j * decomp_coeff[0] * list_of_operators[0])
        for i in range(1, len(decomp_coeff)):
            U = U @ scipy.linalg.expm(-1j * decomp_coeff[i] * list_of_operators[i])

        return U

    def diff(U, U_sequential):
        return 4 - np.trace(U.T.conj() @ U_sequential).real


    # Write a scipy minimization function to find the coefficients that minimize the difference between U and U_sequential
    from scipy.optimize import minimize
    def objective_function(decomp_coeff):
        U_sequential = seq_U(decomp_coeff)
        return diff(U_reconstructed, U_sequential)

    # Initial guess for the coefficients
    result = minimize(objective_function, decomp_coeff,)
    print(seq_U(result.x))
    print("original coefficients:", coefficients)
    print("sequential coefficients:", result.x)
    print(result)


    final_decomp_coeff = result.x
    # Z = 1-2n
    vec = np.zeros([4])
    vec[1] = 1.
    nice_print(vec)
    vec = scipy.linalg.expm(-1j * final_decomp_coeff[-1] * Z2) @ vec
    # (0.9443707207252205+0.32888286947928347j) |01>
    vec = scipy.linalg.expm(-1j * final_decomp_coeff[-2] * Z1) @ vec
    # (0.9213293452412225-0.38878302123084063j) |01>
    vec = scipy.linalg.expm(-1j * final_decomp_coeff[-3] * ZZ) @ vec
    # (0.9894230213850287+0.14505890097929572j) |01>
    vec = scipy.linalg.expm(-1j * final_decomp_coeff[2] * current) @ vec
    # (0.8773596812016387+0.1286293409066828j) |01>
    # (-0.4573815748895422-0.06705652400202011j) |10>
    vec = scipy.linalg.expm(-1j * final_decomp_coeff[1] * hop) @ vec
    # (0.391263080883762+0.4587963659524129j) |01>
    # (-0.12348808436195396-0.7881432542037148j) |10>
    vec = scipy.linalg.expm(-1j * final_decomp_coeff[0] * np.eye(4)) @ vec
    nice_print(vec)

    vec0 = np.zeros([4])
    vec0[1] = 1.
    nice_print(U_reconstructed @ vec0)

    import pdb;pdb.set_trace()
