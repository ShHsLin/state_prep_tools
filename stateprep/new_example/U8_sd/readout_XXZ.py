import numpy as np
from common_setup import *


H_eff = np.zeros([4, 4], dtype=np.complex128)

for i in range(8):
    for j in range(8):
        H_eff += np.load('plaquette_hubbard_H_eff_{i}_{j}.npy'.format(i=i, j=j))

H_eff = H_eff + H_eff.T.conj()
H_eff = H_eff / 2.
H_eff = np.real_if_close(H_eff)

print(H_eff)


c0 = np.trace(H_eff @ np.eye(4)) / 4.
c1 = np.trace(H_eff @ hop) / 2.
c2 = np.trace(H_eff @ current) / 2.
c3 = np.trace(H_eff @ ZZ) / 4.
c4 = np.trace(H_eff @ Z1) / 4.
c5 = np.trace(H_eff @ Z2) / 4.
print("---- coefficients ----")
print("Identity: ", c0)
print("Hopping: ", c1)
print("Current: ", c2)
print("ZZ: ", c3)
print("Z1: ", c4)
print("Z2: ", c5)
print("---- coefficients ----")
