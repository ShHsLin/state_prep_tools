import scipy
import sys
import numpy as np
import pickle
np.set_printoptions(precision=3, suppress=True, linewidth=200)

sys.path.append("..")
from utils.common_setup import *
import circuit

debug = False


# The site  0,  1,  2,  3,  4,  5,  6,  7 correspond to
#           1u  1d  2u  2d  3u  3d  4u  4d
site_idx_to_str = {
    0: '0',
    1: '1',
    2: '2',
    3: '3',
}


if __name__ == "__main__":
    filename = sys.argv[1]

    print("================  Id | XX+YY | XY-YX |  ZZ  |  Z1  |  Z2  ==============")
    data = pickle.load(open(filename, 'rb'))
    if type(data) == circuit.QubitCircuit:
        pairs_of_indices_and_Us = data.pairs_of_indices_and_Us
    else:
        pairs_of_indices_and_Us = data

    for idx in range(len(pairs_of_indices_and_Us)):
        U = pairs_of_indices_and_Us[idx][1]

        U = U.reshape([4, 4])
        # phase = U[0, 0]
        # U = U / phase
        if debug:
            # print("phase: ", phase)
            print("U: ", U)

        H = scipy.linalg.logm(U) / -1.j


        c0 = np.trace(H @ np.eye(4)) / 4.
        c1 = np.trace(H @ hopping) / 2.
        c2 = np.trace(H @ current) / 2.
        c3 = np.trace(H @ ZZ) / 4.
        c4 = np.trace(H @ Z1) / 4.
        c5 = np.trace(H @ Z2) / 4.
        coefficients = np.array([c0, c1, c2, c3, c4, c5])
        coefficients = np.real_if_close(coefficients, 1e-10)

        indices = pairs_of_indices_and_Us[idx][0]
        if np.allclose(coefficients, [-1.571, 1.571, 0., 0., 0.785, 0.785], rtol=1e-3):
            print("--- fSWAP ----")
            continue
        else:
            print("coefficients: ", coefficients, ) # "between:", site_idx_to_str[indices[0]], "and", site_idx_to_str[indices[1]])

        list_of_operators = np.array([np.eye(4), hopping, current, ZZ, Z1, Z2])
        H_reconstructed = np.tensordot(coefficients, list_of_operators, [[0], [0]])

        if debug:
            print("H: ", H)
            print("H_reconstructd:", H_reconstructed)

        U_reconstruct = scipy.linalg.expm(-1.j * H_reconstructed)
        assert np.allclose(U_reconstruct, U), "Reconstructed U does not match original U"
        # print("U=", U,)


    print("================  Id |  hop  |  cur  | n1n2  |  n1  | n2  ==============")

    print(" my lattice setup")
    print(
    """
    3u,3d   4u,4d
    site3   site4
     | >  --- | >

      |        |
      |        |

     | >  --- | >
    site1,   site2
    1u,1d    2u,2d
    """)


    # [ 0.581571186916119 -1.69986963665876  -1.303083781148004  0.581579211369547 -1.730677042356878  0.567526644071236]
    hh=0.582 * np.eye(4) -1.7 * hopping +  -1.303 * current +   0.5816 * ZZ + -1.73* Z1 + 0.5675 * Z2 
    print(hh)
    print(scipy.linalg.expm(-1.j * hh))

