'''
>>> import numpy as np
>>> vec_0a = np.array([1./np.sqrt(2), 0, 0, 1./np.sqrt(2)])
>>> vec_0b = np.array([1., 0, 0, 0])
>>> vec_1a = np.array([0., 1., 0, 0])
>>> vec_1b = np.array([0, 0, 1, 0])
>>> M = np.tensordot(vec_0b.conj(), vec_0a, [[], []]) + np.tensordot(vec_1b.conj(), vec_1a, [[], []]) 
>>> U, S, Vd=np.linalg.svd(M); ideal_unitary = U@Vd
>>> ideal_unitary @ vec_1a
array([0., 0., 1., 0.])
>>> ideal_unitary @ vec_0a
array([1., 0., 0., 0.])
'''
