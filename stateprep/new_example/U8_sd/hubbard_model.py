import numpy as np
from common_setup import *
np.set_printoptions(precision=6, suppress=True)
import os

"""
Hubbard Model:
    We consider hubbard model on a 2x2 lattice.
    The Hamiltonian is given by:
    H = -t * (c†_i c_j + c†_j c_i) + U * (n_i-1/2) (n_j-1/2)
    where t is the hopping parameter, U is the on-site interaction, and n_i is the number operator.

    It is also common to write the Hamiltonian as
    H = -t * (c†_i c_j + c†_j c_i) + U * n_i n_j [- U / 2 * (n_i + n_j) + U / 4]
    when U = 8 and half-filling, - U / 4 is a constant term = -2 per site
    when U = 8 and quarter-filling, 0 is a constant term = 0 per site

    Z = (1-2n)
    (n-1/2) = -Z / 2
    n = (1-Z) / 2

    In the JW transformation, we have:
    c†_i c_j + c†_j c_i = (XX + YY) / 2
    (n_i-1/2) (n_j-1/2) = ZZ / 4

    Again, our convention over the qubits is as follows:
    0   1   2   3   4   5   6   7
    1u  1d  2u  2d  3u  3d  4u  4d


   45    67
    3 -- 4
    |    |
    |    |
    1 -- 2
   01    23
"""

def Z_power(n):
    """
    Generate Z^ \\otimes n operator for n qubits.
    """
    if n == 1:
        return Z
    if n == 2:
        return np.kron(Z, Z)
    if n > 2:
        return np.kron(Z_power(n-1), Z)

def generate_Hubbard_Hamiltonian(t_dict, U_dict, N):
    """
    Generate free fermion Hamiltonian:
    H = \sum_{i=1}^{N-1} (c_i^\dagger c_{i+1} + c_{i+1}^\dagger c_i)
    """
    H = np.zeros((2**N, 2**N))
    for key in t_dict.keys():
        print(key)
        i, j = key
        assert i < j
        H += t_dict[key] * np.kron(np.eye(2**i),
                                   np.kron(X,
                                           np.kron(Z_power(j-i-1),
                                                   np.kron(X, np.eye(2**(N-j-1)))))
                                   ) / 2
        H += t_dict[key] * np.kron(np.eye(2**i),
                                   np.kron(Y,
                                           np.kron(Z_power(j-i-1),
                                                   np.kron(Y, np.eye(2**(N-j-1)))))
                                   ).real / 2
    for key in U_dict.keys():
        i, j = key
        assert i < j
        H += U_dict[key] * np.kron(np.eye(2**i),
                                   np.kron(interaction_term, np.eye(2**(N-j-1))))

    return H

if __name__ == "__main__":
    # Terms:
    # 1. Hopping term
    #     -t [(0, 2), (0, 4), (2, 6), (4, 6)]
    #     -t [(1, 3), (1, 5), (3, 7), (5, 7)]
    # 2. On-site interaction term
    #      U [(0, 1), (2, 3), (4, 5), (6, 7)]

    t = 1.0
    U = 8.0

    interaction_term = ZZ / 4.
    # interaction_term = np.kron((np.eye(2) - Z) / 2, (np.eye(2) - Z) / 2)

    t_dict = {
        (0, 2): -t,
        (0, 4): -t,
        (2, 6): -t,
        (4, 6): -t,
        (1, 3): -t,
        (1, 5): -t,
        (3, 7): -t,
        (5, 7): -t
    }

    U_dict = {
        (0, 1): U,
        (2, 3): U,
        (4, 5): U,
        (6, 7): U
    }

    H = generate_Hubbard_Hamiltonian(t_dict, U_dict, N=8)
    E, V = np.linalg.eigh(H) 
    print("Eigenvalues: ", E[:10])
    niceprint(V[:, 0])

    s_state = np.load('plaquette_s.npy')
    d_state = np.load('plaquette_d.npy')


    # We want to compute the schrieffer wolff transformation,
    # where the low-energy subspace is spanned by |ss>, |sd>, |ds>, |dd>.
    # H = H0 + V
    # H0 in this case is the Hubbard model on a 2x2 lattice
    # V is the hopping term across the plaquettes.

    # We want to compute the effective Hamiltonian in the low-energy subspace
    # H_eff = Proj ( H0 + V + 1/2 [S, V] ) Proj schrieffer wolff transformation
    # where S is the generator of the transformation, and Proj is the projector
    # onto the low-energy subspace.

    # S_{ij} = <i | V | j> / (E_i - E_j),  S_{ij} = 0 for i=j

    # The projector is given by:
    # P = |ss><ss| + |sd><sd| + |ds><ds| + |dd><dd|

    # The non-trivial term is given in the form:
    # < low-energy-i | V |k><k| V | low-energy-j> / (E_i - E_k)
    # V = \sum_n Pn, where Pn is the Pauli string 
    # for the hopping term across the plaquettes.

    # Due to the tensor product structure, for each pauli string,
    # we can compute the < k | Pn | low-energy-j > as
    # < k_1 | Pn_1 | j_1 > * < k_2 | Pn_2 | j_2 >


    # for low-energy states i,j in [|ss>, |sd>, |ds>, |dd>]
    # for k in [all eigenstates]  # double for loop k,k'
    # 


    # There are 4 hopping terms across the plaquettes:
    # -t (2, 8) 
    # -t (3, 9)
    # -t (6, 12)
    # -t (7, 13)

    P_str_pairs = [('IICZZZZZ', 'AIIIIIII'),
                   ('IIAZZZZZ', 'CIIIIIII'),
                   ('IIICZZZZ', 'ZAIIIIII'),
                   ('IIIAZZZZ', 'ZCIIIIII'),
                   ('IIIIIICZ', 'ZZZZAIII'),
                   ('IIIIIIAZ', 'ZZZZCIII'),
                   ('IIIIIIIC', 'ZZZZZAII'),
                   ('IIIIIIIA', 'ZZZZZCII'),
                   ]

    sd_to_states = {
            's': s_state,
            'd': d_state,
            }

    sd_to_energies = {
            's': E[0],
            'd': E[64],
            }

    print(E[0], E[64])

    low_energy_states = ['ss', 'sd', 'ds', 'dd']
    for V_1_idx in range(8):
        for V_2_idx in range(8):
            if os.path.exists(f'plaquette_hubbard_H_eff_{V_1_idx}_{V_2_idx}.npy'):
                print(f'plaquette_hubbard_H_eff_{V_1_idx}_{V_2_idx}.npy already exists')
                continue
            else:
                print(f'plaquette_hubbard_H_eff_{V_1_idx}_{V_2_idx}.npy does not exist')
                print(f'working on {V_1_idx}, {V_2_idx}')
                print("===" * 40)

            P_str_1_1 = P_str_pairs[V_1_idx][0]
            P_str_1_2 = P_str_pairs[V_1_idx][1]
            P_str_2_1 = P_str_pairs[V_2_idx][0]
            P_str_2_2 = P_str_pairs[V_2_idx][1]

            Op_1_1 = pauli_to_sparse_op(P_str_1_1)
            Op_1_2 = pauli_to_sparse_op(P_str_1_2)
            Op_2_1 = pauli_to_sparse_op(P_str_2_1)
            Op_2_2 = pauli_to_sparse_op(P_str_2_2)

            H_eff = np.zeros((4, 4))
            for bra_idx in range(4):
                bra_string = low_energy_states[bra_idx]

                bra_state_1 = sd_to_states[bra_string[0]] 
                bra_state_2 = sd_to_states[bra_string[1]]
                bra_E_1 = sd_to_energies[bra_string[0]]
                bra_E_2 = sd_to_energies[bra_string[1]]

                for ket_idx in range(4):
                    ket_string = low_energy_states[ket_idx]
                    print("working on", bra_string, ket_string)

                    ket_state_1 = sd_to_states[ket_string[0]]
                    ket_state_2 = sd_to_states[ket_string[1]]

                    for k1 in range(2**8):
                        
                        k1_state = V[:, k1]
                        for k2 in range(2**8):

                            k2_state = V[:, k2]

                            term1_1 = bra_state_1.conj() @ Op_1_1 @ k1_state
                            term1_2 = bra_state_2.conj() @ Op_1_2 @ k2_state
                            term2_1 = k1_state.conj() @ Op_2_1 @ ket_state_1
                            term2_2 = k2_state.conj() @ Op_2_2 @ ket_state_2

                            tmp = term1_1 * term1_2 * term2_1 * term2_2
                            E_diff = E[k1] + E[k2] - bra_E_1 - bra_E_2
                            if np.isclose(E_diff, 0):
                                continue

                            tmp = tmp / (E[k1] + E[k2] - bra_E_1 - bra_E_2)

                            if np.abs(tmp) > 1e-4:
                                print(k1, k2, tmp)

                            H_eff[bra_idx, ket_idx] += tmp

            print("H_eff: ")
            print(H_eff)
            np.save(f'plaquette_hubbard_H_eff_{V_1_idx}_{V_2_idx}.npy', H_eff)
