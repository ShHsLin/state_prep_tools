import numpy as np
import scipy
import sys
sys.path.append('../')
import circuit
import utils.misc as misc
from utils.common_setup import *

sys.path.append('../../state_vec_sim/')
from many_body import gen_H_2d_XXZ


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
    depth = int(sys.argv[1])

    init_vec = np.zeros((2**N,), dtype=np.complex128)
    init_vec[int('011101110', 2)] = 1.

    # init circuit
    gate_indices = [(0, 1), (3, 4), (6, 7),
                   (1, 2), (4, 5), (7, 8),
                   # (2, 0), (5, 3), (8, 6),
                   (0, 3), (1, 4), (2, 5),
                   (3, 6), (4, 7), (5, 8),
                   # (6, 0), (7, 1), (8, 2)
                    ]

    pair_of_indices_and_Us = []
    for depth_idx in range(depth):
        for indices in gate_indices:
            pair_of_indices_and_Us.append((indices, misc.get_random_u1_2q_gate()))

    my_circ = circuit.QubitCircuit(pair_of_indices_and_Us)

    H = gen_H_2d_XXZ(Lx, Ly, 0., True)
    # We are targeting H = - XX - YY
    H = - H
    E = my_circ.get_energy(H, init_vec)
    print(E)

    my_circ_params = np.array(my_circ.get_params()).flatten()
    print(my_circ_params)
    my_circ.set_params(my_circ_params)
    E = my_circ.get_energy(H, init_vec)
    print(E)

    def f(params):
        my_circ.set_params(params)
        E = my_circ.get_energy(H, init_vec)
        return E


    result = scipy.optimize.minimize(f, my_circ_params, method='L-BFGS-B', options={'disp': True})
    my_circ_params = result.x

    import pdb;pdb.set_trace()

    result = scipy.optimize.minimize(f, my_circ_params, method='L-BFGS-B', options={'disp': True})
    my_circ_params = result.x

    import pdb;pdb.set_trace()

    result = scipy.optimize.minimize(f, my_circ_params, method='L-BFGS-B', options={'disp': True})
    my_circ_params = result.x

    import pdb;pdb.set_trace()





