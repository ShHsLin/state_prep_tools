import numpy as np
import scipy.sparse

X = np.array([[0., 1.], [1., 0.]])
Y = np.array([[0., -1.j], [1.j, 0.]])
Z = np.array([[1., 0.], [0., -1.]])

Id2 = np.eye(2)
f_creation = np.array([[0, 0],
                       [1, 0]])
f_annihilation = np.array([[0, 1],
                           [0, 0]])

str_to_op = {'X': X,
             'Y': Y,
             'Z': Z,
             'I': Id2,
             'C': f_creation,
             'A': f_annihilation,
             }


XX = np.kron(X, X)
YY = np.kron(Y, Y)
XY = np.kron(X, Y)
YX = np.kron(Y, X)
ZZ = np.kron(Z, Z)
Z1 = np.kron(Z, np.eye(2))
Z2 = np.kron(np.eye(2), Z)

hopping = (XX + YY) / 2.
current = (XY - YX) / 2.


def nice_print(vec, tolerance=1e-10):
    """
    Print the vector in a nice format
    """
    n_qubits = int(np.log2(len(vec)))    
    for i in range(len(vec)):
        if vec[i] != 0 and abs(vec[i]) > tolerance:
            print(f"{vec[i]} |{i:0{n_qubits}b}>")
    print("")

def niceprint(v, tol=1e-4):
    for j in range(len(v)):
        if abs(v[j])>tol:
            ket = '{0:b}'.format(j).zfill(int(np.log2(len(v))))
            st = ' '.join(a+b for a,b in zip(ket[::2], ket[1::2]))
            if v[j]>0:
                sign = '+'
            else:
                sign = ''
            #print('{}{:.4f} |{}>'.format(sign, np.real(v[j]),st))
            print('{}{:.16f} |{}>'.format(sign, v[j], st))

def pauli_to_sparse_op(pauli_string):
    full_sparse_op = scipy.sparse.eye(1)
    for p_str in pauli_string:
        full_sparse_op = scipy.sparse.kron(full_sparse_op, str_to_op[p_str])
        full_sparse_op.eliminate_zeros()

    return full_sparse_op

