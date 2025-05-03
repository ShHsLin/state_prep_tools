import numpy as np
vec_0a = np.array([0, 0, 0, 1.])
vec_0b = np.array([0, 0, 0, -1.j])
vec_1a = np.array([0, 0., 1., 0])
vec_1b = np.array([0, 1./np.sqrt(2), -1/np.sqrt(2), 0])
M = np.tensordot(vec_0b.conj(), vec_0a, [[], []]) + np.tensordot(vec_1b.conj(), vec_1a, [[], []]) 
U, S, Vd=np.linalg.svd(M)
ideal_unitary = (U@Vd).conj()


import sys
sys.path.append("..")
import circuit
C = circuit.GenericCircuit([((0, 1), np.eye(4))])

list_of_initial_states = [vec_0a, vec_1a]
list_of_target_states = [vec_0b, vec_1b]
C.polar_opt(list_of_target_states, list_of_initial_states, True)
print(C.pairs_of_indices_and_Us[0][1].reshape([4, 4]))

