import numpy as np
import sys
sys.path.append("..")
import gate
import circuit
# np.set_printoptions(precision=3)
import pickle
import utils.misc as misc

"""
q4,q5   q6,q7
site2   site3
 | >  --- | >

  |        |
  |        |

 | >  --- | >
site0,   site1
q0,q1    q2,q3

We have states on first sites where each site can have one of the states {|hole>, |up>, |down>, |double>}
We denote it as |00>, |10>, |01>, |11> respectively.

1/sqrt(8) x (
|01 01 10 10> + |10 01 10 01> + |10 10 01 01 > + |01 10 01 10>
+ 2 ( |01 10 10 01 > + | 10 01 01 10 > )
)
<— |01 10 10 01>
1/sqrt(8) x (
|01 00 10 00 > - |10 00 01 00> + |00 00 10 01> - |00 00 01 10>
+ |01 10 00 00> - |10 01 00 00> + |00 01 00 10> - |00 10 00 01>
)
<—
|01 00 10 00>

# The objective function is: 
# ArgMax_U Re[<vec_1b| U |vec_1a> + <vec_2b| U |vec_2a>]
# which has maximal value 2.

# Need to figure out how to do polar.
# - - - - - - - - - - - - - - - - - -
# Construct the representation of \\Prod U_i |vec_1a>
# Get Env by backward prop with <vec_1b| \\Prod U_i |vec_1a>
# 
# Do exactly the same with |vec_2a> and <vec_2b|.
#
# Do polar decompisition only on the inner block (U1 block).

# Check how many layer works.
# D * 8 * 4 * 9 <-- probably only 1, 2 layer is affordable.

# The tool kit should include trimming the circuit.

"""

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

    # -----------------------------------------------------#
    state_1a_string_coeff = {"01101001": 1.}
    state_1b_string_coeff = {"01011010": 1./np.sqrt(12),
                             "10011001": 1./np.sqrt(12),
                             "10100101": 1./np.sqrt(12),
                             "01100110": 1./np.sqrt(12),
                             "01101001": 2./np.sqrt(12),
                             "10010110": 2./np.sqrt(12),
                             }

    state_2a_string_coeff = {"01001000": 1.}
    state_2b_string_coeff = {"01001000": 1./np.sqrt(8),
                             "10000100": -1./np.sqrt(8),
                             "00001001": 1./np.sqrt(8),
                             "00000110": -1./np.sqrt(8),
                             "01100000": 1./np.sqrt(8),
                             "10010000": -1./np.sqrt(8),
                             "00010001": 1./np.sqrt(8),
                             "00100001": -1./np.sqrt(8),
                             }
    # -----------------------------------------------------#
    state_1b_string_coeff = {
            "00001111": +0.0205442099135214,
            "00011110": -0.0889576356835152,
            "00101101": +0.0889576356835288,
            "00110011": -0.0205442099134936,
            "00110110": +0.0889576356834869,
            "00111001": -0.0889576356834992,
            "01001011": -0.0889576356835277,
            "01011010": -0.2695206186630775,
            "01100011": +0.0889576356834966,
            "01100110": -0.2695206186628770,
            "01101001": +0.5390412373260107,
            "01101100": +0.0889576356834963,
            "01111000": -0.0889576356835192,
            "10000111": +0.0889576356835141,
            "10010011": -0.0889576356834884,
            "10010110": +0.5390412373259271,
            "10011001": -0.2695206186628957,
            "10011100": -0.0889576356834889,
            "10100101": -0.2695206186630793,
            "10110100": +0.0889576356835094,
            "11000110": +0.0889576356834873,
            "11001001": -0.0889576356834984,
            "11001100": -0.0205442099134957,
            "11010010": -0.0889576356835096,
            "11100001": +0.0889576356835191,
            "11110000": +0.0205442099135349,
            }

    state_2b_string_coeff = {
            "00000011": -0.0929929460083306,
            "00000110": +0.2605604445846077,
            "00001001": -0.2605604445845812,
            "00001100": -0.0929929460083926,
            "00010010": +0.2605604445846001,
            "00011000": +0.3249135599328987,
            "00100001": -0.2605604445845984,
            "00100100": -0.3249135599329071,
            "00110000": -0.0929929460083880,
            "01000010": +0.3249135599329077,
            "01001000": +0.2605604445849125,
            "01100000": +0.2605604445849106,
            "10000001": -0.3249135599328974,
            "10000100": -0.2605604445849120,
            "10010000": -0.2605604445849061,
            "11000000": -0.0929929460084456,
            }


    vec_1a = gen_vec_from_dict(state_1a_string_coeff)
    vec_1b = gen_vec_from_dict(state_1b_string_coeff)
    vec_2a = gen_vec_from_dict(state_2a_string_coeff)
    vec_2b = gen_vec_from_dict(state_2b_string_coeff)
    assert np.isclose(np.linalg.norm(vec_1a), 1.), f"the vectors are not normalized {np.linalg.norm(vec_1a)}"
    assert np.isclose(np.linalg.norm(vec_1b), 1.), f"the vectors are not normalized {np.linalg.norm(vec_1b)}"
    assert np.isclose(np.linalg.norm(vec_2a), 1.), f"the vectors are not normalized {np.linalg.norm(vec_2a)}"
    assert np.isclose(np.linalg.norm(vec_2b), 1.), f"the vectors are not normalized {np.linalg.norm(vec_2b)}"

    list_of_initial_states = [vec_1a, vec_2a]
    list_of_target_states = [vec_1b, vec_2b]



    # In terms of cubic connectivity, we should consider the circuit
    # 45 -- 67
    # |      |
    # 01 -- 23

    # The site  0,  1,  2,  3,  4,  5,  6,  7 correspond to
    #           1u  1d  2u  2d  3u  3d  4u  4d
    # Need to define a brickwork circuit acting
    # [(0,1), (2,3), (4,5), (6,7)] and 
    # [(0,2), (1,3), (4,6), (5,7)]
    # [(0,4), (1,5), (2,6), (3,7)]


    # (i) complex
    # (ii) real
    # (iii) complex + [(1,2), (3,4), (5,6), (7,0)]
    # (iv) real + [(1,2), (3,4), (5,6), (7,0)]

    # seq = int(sys.argv[1])
    seq = 4
    depth = int(sys.argv[1])


    Id = np.eye(4)

    list_of_indices = [
                       (0, 1), (2, 3), (4, 5), (6, 7),
                       (0, 2), (2, 4), (4, 6), (0, 6),
                       (0, 1), (2, 3), (4, 5), (6, 7),
                       (1, 3), (3, 5), (5, 7), (1, 7),
                       ]
    # if seq == 4:
    #     list_of_indices = list_of_indices + [(1, 2), (3, 4), (5, 6), (0, 7),]


    pair_of_indices_and_Us = []
    for depth_idx in range(depth):
        for indices in list_of_indices:
            # pair_of_indices_and_Us.append((indices, misc.get_random_u1_2q_gate()))
            pair_of_indices_and_Us.append((indices, gate.U1UnitaryGate(scale=1e-1)))

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

    with open(f"circuit_sequence_{seq}_complex_depth{depth}.pickle", "wb") as f:
        pickle.dump(C.pairs_of_indices_and_Us, f)

    import matplotlib.pyplot as plt
    plt.plot(np.real(list_of_bottom_states[0].state_vector), 'bo-', label='vec_1b')
    plt.plot(vec_1b, 'bs', label='True vec_1b', mfc='none')
    plt.plot(np.real(list_of_bottom_states[1].state_vector), 'ko-', label='vec_2b')
    plt.plot(vec_2b, 'ks', label='True vec_2b', mfc='none')
    plt.legend()
    plt.show()

    for pair in C.pairs_of_indices_and_Us:
        print(pair[0], "\n", pair[1].reshape([4, 4]) / pair[1][0,0,0,0])


