import numpy as np
import sys
sys.path.append("..")
import circuit
np.set_printoptions(precision=3)

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

def get_random_u1_2q_gate(using_complex=True):
    """
    Generate a random U1 2-qubit unitary gate.
    """
    if using_complex:
        M = np.random.rand(2, 2) + 1j * np.random.rand(2, 2) - 0.5 - 0.5j
    else:
        M = np.random.rand(2, 2) - 0.5

    M = M * 1e-4 + np.eye(2)
    Q, R = np.linalg.qr(M)

    if using_complex:
        two_qubit_unitary = np.zeros((4, 4), dtype=np.complex128)
    else:
        two_qubit_unitary = np.zeros((4, 4), dtype=np.float64)

    two_qubit_unitary[0, 0] = 1.
    two_qubit_unitary[1:3, 1:3] = Q

    if using_complex:
        scale = np.random.rand(1) * np.pi * 2
        phase = np.exp(1j * scale)
        two_qubit_unitary[3, 3] = phase.item() 
    else:
        two_qubit_unitary[3, 3] = 1.

    return two_qubit_unitary

def gen_vec_from_dict(string_coeff_dict):
    """
    Generate a vector from a dictionary of string coefficients.
    The keys of the dictionary are strings representing the states,
    and the values are the coefficients for those states.
    """
    one_key = next(iter(string_coeff_dict))
    L = len(one_key)
    vec = np.zeros(2**L, dtype=np.float64)
    for key, coeff in string_coeff_dict.items():
        index = int(key, 2)
        vec[index] = coeff
    return vec






if __name__ == "__main__":

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


    vec_1a = gen_vec_from_dict(state_1a_string_coeff)
    vec_1b = gen_vec_from_dict(state_1b_string_coeff)
    vec_2a = gen_vec_from_dict(state_2a_string_coeff)
    vec_2b = gen_vec_from_dict(state_2b_string_coeff)


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


    Id = np.eye(4)

    list_of_indices = [(0, 1), (2, 3), (4, 5), (6, 7),
                       (0, 2), (1, 3), (4, 6), (5, 7),
                       (0, 4), (1, 5), (2, 6), (3, 7),
                       (1, 2), (3, 4), (5, 6), (7, 0),
                       ]

    depth = int(sys.argv[1])
    pair_of_indices_and_Us = []
    for depth_idx in range(depth):
        for indices in list_of_indices:
            pair_of_indices_and_Us.append((indices, get_random_u1_2q_gate()))

    C = circuit.GenericCircuit(pair_of_indices_and_Us)
    cost, list_of_bottom_states = C.polar_opt(list_of_target_states,
                                              list_of_initial_states,
                                              verbose=True,)

    for i in range(1000):
        cost, list_of_bottom_states = C.polar_opt(list_of_target_states,
                                                  list_of_initial_states,
                                                  list_of_bottom_states=list_of_bottom_states,
                                                  verbose=True,
                                                  )

    with open(f"sequence_4_real_depth{depth}.txt", "a") as f:
        # wrtie the cost to 5 decimal places
        f.write(str(cost) + "\n")

    import matplotlib.pyplot as plt
    plt.plot(np.real(list_of_bottom_states[0].state_vector), 'bo-', label='vec_1b')
    plt.plot(vec_1b, 'bs', label='True vec_1b', mfc='none')
    plt.plot(np.real(list_of_bottom_states[1].state_vector), 'ko-', label='vec_2b')
    plt.plot(vec_2b, 'ks', label='True vec_2b', mfc='none')
    plt.legend()
    plt.show()

    for pair in C.pairs_of_indices_and_Us:
        print(pair[0], "\n", pair[1].reshape([4, 4]) / pair[1][0,0,0,0])


