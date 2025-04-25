import pickle
import numpy as np
import scipy
import sys
np.set_printoptions(precision=3, suppress=True, linewidth=200)

X = np.array([[0., 1.], [1., 0.]])
Y = np.array([[0., -1.j], [1.j, 0.]])
Z = np.array([[1., 0.], [0., -1.]])

XX = np.kron(X, X)
YY = np.kron(Y, Y)
XY = np.kron(X, Y)
YX = np.kron(Y, X)
ZZ = np.kron(Z, Z)
Z1 = np.kron(Z, np.eye(2))
Z2 = np.kron(np.eye(2), Z)

hop = (XX + YY) / 2.
current = (XY - YX) / 2.

debug = False

if debug:
    print("XX + YY", hop)
    print("XY - YX", current)
    print("ZZ", ZZ)
    print("Z1", Z1)
    print("Z2", Z2)

# print("XX - YY", (XX - YY) / 2.)
# print("XY + YX", (XY + YX) / 2.)

# The site  0,  1,  2,  3,  4,  5,  6,  7 correspond to
#           1u  1d  2u  2d  3u  3d  4u  4d
site_idx_to_str = {
    0: '1u',
    1: '1d',
    2: '2u',
    3: '2d',
    4: '3u',
    5: '3d',
    6: '4u',
    7: '4d'
}


depth = int(sys.argv[1])

print("================  Id | XX+YY | XY-YX |  ZZ  |  Z1  |  Z2  ==============")
pairs_of_indices_and_Us = pickle.load(open(f'circuit_sequence_3_complex_depth{depth}.pickle', 'rb'))
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
    c1 = np.trace(H @ hop) / 2.
    c2 = np.trace(H @ current) / 2.
    c3 = np.trace(H @ ZZ) / 4.
    c4 = np.trace(H @ Z1) / 4.
    c5 = np.trace(H @ Z2) / 4.
    coefficients = np.array([c0, c1, c2, c3, c4, c5])
    coefficients = np.real_if_close(coefficients, 1e-10)

    indices = pairs_of_indices_and_Us[idx][0]
    print("coefficients: ", coefficients, "between:", site_idx_to_str[indices[0]], "and", site_idx_to_str[indices[1]])

    list_of_operators = np.array([np.eye(4), hop, current, ZZ, Z1, Z2])
    H_reconstructed = np.tensordot(coefficients, list_of_operators, [[0], [0]])

    if debug:
        print("H: ", H)
        print("H_reconstructd:", H_reconstructed)

    U_reconstruct = scipy.linalg.expm(-1.j * H_reconstructed)
    assert np.allclose(U_reconstruct, U), "Reconstructed U does not match original U"

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
import pdb;pdb.set_trace()
