import numpy as np
import scipy
import stateprep.circuit as circuit
import stateprep.exact_sim as exact_sim

import stateprep.utils.misc as misc
from stateprep.utils.common_setup import *

import sys
sys.path.append('../../../state_vec_sim/')
from many_body import gen_H_2d_XXZ
from scipy.sparse.linalg import eigsh


"""
Find the GS of a XX model on a 2D lattice.
-0.27268046 (3x3) with 6 |1>s and 3 |0>s

(6) -- (7) -- (8)
 |      |      |
 |      |      |
(3) -- (4) -- (5)
 |      |      |
 |      |      |
(0) -- (1) -- (2)

[(0, 1), (3, 4), (6, 7),
 (1, 2), (4, 5), (7, 8),
 (2, 0), (5, 3), (8, 6),
 (0, 3), (1, 4), (2, 5),
 (3, 6), (4, 7), (5, 8),
 (6, 0), (7, 1), (8, 2)]

"""

# TODO
"""
1. Write a function map parameters to U1 gate
[c0, c1, c2, c3, c4, c5] -> U1 gate

2. Write the generic circuit according to the order
   described above.
   Per depth, there are 18 gates.

3. Write a function to calculate the energy

4. Write a function to calculate the gradient

5. Write the optimization

"""


if __name__ == '__main__':
    # init product state 
    Lx = Ly = 3
    N = Lx * Ly

    init_vec = np.zeros((2**N,), dtype=np.complex128)
    init_vec[int('011101110', 2)] = 1.

    H = gen_H_2d_XXZ(Lx, Ly, -0.737073337887132, True)
    H = - H
    evals_small, evecs_small = eigsh(H, 16, which='SA', v0=init_vec)
    print(evals_small)

    vec = evecs_small[:, 0]
    gs = exact_sim.StateVector(vec)
    gs.apply_gate(hopping.reshape([2, 2, 2, 2]), (0,1))
    print("delta (0,1) : ", vec.conj() @ gs.state_vector)

    gs = exact_sim.StateVector(vec)
    gs.apply_gate(hopping.reshape([2, 2, 2, 2]), (0,2))
    print("delta (0,2) : ", vec.conj() @ gs.state_vector)

    gs = exact_sim.StateVector(vec)
    gs.apply_gate(hopping.reshape([2, 2, 2, 2]), (0,4))
    print("delta (0,4) : ", vec.conj() @ gs.state_vector)
