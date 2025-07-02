import numpy as np
import scipy
import sys
sys.path.append('../')
import gate
import circuit
import utils.misc as misc
from utils.common_setup import *

sys.path.append('../../state_vec_sim/')
from many_body import gen_H_2d_XXZ


# TODO

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
            # pair_of_indices_and_Us.append((indices, misc.get_random_u1_2q_gate()))
            init_U = misc.get_random_u1_2q_gate(scale=1e-1)
            # init_U = None
            U = gate.U1UnitaryGate(init_U=init_U)
            pair_of_indices_and_Us.append((indices, U))

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
        return E.real

    def f_and_g(params):
        my_circ.set_params(params)
        E = my_circ.get_energy(H, init_vec)
        g = my_circ.get_energy_gradient(H, init_vec)
        g = np.array(g).flatten()
        return E.real, g


    x0 = my_circ_params
    finite_diff_grad = []
    fx0 = f(x0)
    for i in range(len(x0)):
        x1 = x0.copy()
        x1[i] += 1e-4

        grad_0 = (f(x1) - fx0) / 1e-4
        finite_diff_grad.append(grad_0)

    print("finite_diff_grad =", finite_diff_grad)

    _, grad_2 = f_and_g(x0)
    print("grad_2 =", grad_2)


    import matplotlib.pyplot as plt
    plt.plot(finite_diff_grad, 'o-', label='finite_diff_grad')
    plt.plot(np.array(grad_2)*2, 's-', label='jax_grad')
    plt.legend()
    plt.show()

    # result = scipy.optimize.minimize(f_and_g, my_circ_params, method='L-BFGS-B',
    #                                  options={'disp': True}, jac=True)
    # my_circ_params = result.x

