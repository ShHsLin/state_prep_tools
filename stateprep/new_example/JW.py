import numpy as np
from common_setup import *
"""
Operator [ C1^\dagger C_2 + C_2^\dagger C_1]
Matrix represenation:
= (X1-iY1)/2 Z2 (X2+iY2)/2 + (X2-iY2)/2 (X1+iY1)/2 Z2
= (X1-iY1)/2 (iY2+X2)/2 + (-iY2+X2)/2 (X1+iY1)/2
= c1dagger (c2) + c1 c2dagger

"""

"""
C^\dagger_1 C_3 + C_3^\dagger C_1
= (X1-iY1)/2 Z2 Z3 (X3+iY3)/2 + (X3-iY3)/2 (X1+iY1)/2 Z2 Z3
= (X1-iY1)/2 Z2 (iY3+X3)/2 + (X1+iY1)/2 Z2 (-iY3+X3)/2
= c1dagger Z c3 + c1 Z c3dagger

or [double check]
= FSWAP(1,2) (c2dagger c3 + c2 c3dagger) FSWAP(1,2)


TODO:
    learn about the compact encoding
"""

XZX = np.kron(X, np.kron(Z, X))
YZY = np.kron(Y, np.kron(Z, Y))
Op = XZX + YZY

FSWAP = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, -1]]).reshape([2, 2, 2, 2])
XX_YY = np.kron(X, X) + np.kron(Y, Y)
XX_YY = XX_YY.reshape([2, 2, 2, 2])

#[p1*, (p2*), p1, p2], [p2*, p3*, (p2), p3]
Op2 = np.tensordot(FSWAP, XX_YY, [[1], [2]])
#[p1*, p2*, (p1), (p2)] [(p1*), p1, p2, (p2*), p3*, p3]
Op2 = np.tensordot(FSWAP, Op2, [[2, 3], [0, 3]])
#[p1*, p2*, p1, p2, p3*, p3]
Op2 = np.transpose(Op2, [0, 1, 4, 2, 3, 5])
Op2 = Op2.reshape([8, 8])
assert np.allclose(Op, Op2)



def generate_JW_Hamiltonian(N=4):
    """
    Generate free fermion Hamiltonian:
    H = \sum_{i=1}^{N-1} (c_i^\dagger c_{i+1} + c_{i+1}^\dagger c_i)
    """
    hop = XX_YY.real / 2
    hop = hop.reshape([4, 4])
    H = np.zeros((2**N, 2**N))
    for i in range(N-1):
        H += np.kron(np.eye(2**i), np.kron(hop, np.eye(2**(N-i-2)))) 

    H = H + (np.kron(X, np.kron(np.eye(2**(N-2)), X)).real / 2.)
    H = H + (np.kron(Y, np.kron(np.eye(2**(N-2)), Y)).real / 2.)
    return -H


H_ff = generate_JW_Hamiltonian(4)
E, V = np.linalg.eigh(H_ff)
print(E)


print("vector norm: ", np.linalg.norm(V[:, 0]))
V0 = V[:, 0]
print("E0 :", V0.T @ H_ff @ V0)
for v in V0.real:
    print("%.4f" % v, end=", ")

print("\n")
nice_print(V[:, 0])
print("sin(pi/4) / 2 = ", np.sin(np.pi/4) / 2)
print("sin(pi/2) / 2 = ", np.sin(np.pi/2) / 2)
print("sin(3pi/4) / 2 = ", np.sin(3*np.pi/4) / 2)


cd = np.array([[0, 0], [1, 0]])
c = np.array([[0, 1], [0, 0]])

assert np.allclose(cd, (X - 1j * Y) / 2)
assert np.allclose(c, (X + 1j * Y) / 2)

cd1cd2 = np.kron(cd, np.kron(cd, np.eye(4)))
cd1cd3 = np.kron(cd, np.kron(Z, np.kron(cd, np.eye(2))))
cd1cd4 = np.kron(cd, np.kron(Z, np.kron(Z, cd)))
vac = np.zeros(16)
vac[0] = 1
print("cd1cd2 @ vac")
nice_print(cd1cd2 @ vac)
print("cd1cd3 @ vac")
nice_print(cd1cd3 @ vac)
print("cd1cd4 @ vac")
nice_print(cd1cd4 @ vac)




