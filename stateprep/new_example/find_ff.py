import numpy as np
import sys
sys.path.append("..")
import circuit
# np.set_printoptions(precision=3)
import pickle
from utils.misc import *


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






if __name__ == "__main__":


    vec_1a = np.zeros([16])
    vec_1a[5] = 1.
    a = np.sqrt(1/2.)/2
    vec_1b = np.array([0.0, 0.0, 0.0, -a, 0.0, -0.5000, -a, 0.0000, -0.0000, -a, -0.5000, 0.0000, -a, 0.0000, 0.0000, 0.0000,])

    assert np.isclose(np.linalg.norm(vec_1a), 1.), f"the vectors are not normalized {np.linalg.norm(vec_1a)}"
    assert np.isclose(np.linalg.norm(vec_1b), 1.), f"the vectors are not normalized {np.linalg.norm(vec_1b)}"

    list_of_initial_states = [vec_1a,]
    list_of_target_states = [vec_1b,]

    depth = int(sys.argv[1])


    Id = np.eye(4)

    list_of_indices = [
                       (0, 1), (2, 3),
                       (0, 2), (1, 3),
                       ]



    pair_of_indices_and_Us = []
    for depth_idx in range(depth):
        for indices in list_of_indices:
            pair_of_indices_and_Us.append((indices, get_random_u1_2q_gate()))

    C = circuit.FermionicCircuit(pair_of_indices_and_Us)
    C = C.export_to_QubitCircuit()
    cost, list_of_bottom_states = C.polar_opt(list_of_target_states,
                                              list_of_initial_states,
                                              verbose=True,)

    for i in range(1000):
        cost, list_of_bottom_states = C.polar_opt(list_of_target_states,
                                                  list_of_initial_states,
                                                  list_of_bottom_states=list_of_bottom_states,
                                                  verbose=True,
                                                  )

    # with open(f"sequence_{seq}_complex_depth{depth}.txt", "a") as f:
    #     # wrtie the cost to 5 decimal places
    #     f.write(str(cost) + "\n")

    with open(f"circuit_sequence_depth{depth}.pickle", "wb") as f:
        pickle.dump(C.pairs_of_indices_and_Us, f)

    import matplotlib.pyplot as plt
    plt.plot(np.real(list_of_bottom_states[0].state_vector), 'bo-', label='vec_1b')
    plt.plot(vec_1b, 'bs', label='True vec_1b', mfc='none')
    plt.legend()
    plt.show()

    for pair in C.pairs_of_indices_and_Us:
        print(pair[0], "\n", pair[1].reshape([4, 4]) / pair[1][0,0,0,0])


