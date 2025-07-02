

```
>>> import numpy as np
>>> vec_1a = np.array([0, 0, 0, 1.])
>>> vec_1b = np.array([0, 0, 0, -1.j])
>>> vec_2a = np.array([0, 0., 1., 0])
>>> vec_2b = np.array([0, 1./np.sqrt(2), -1/np.sqrt(2), 0])
>>> M = np.tensordot(vec_1b.conj(), vec_1a, [[], []]) + np.tensordot(vec_2b.conj(), vec_2a, [[], []])
>>> U, S, Vd=np.linalg.svd(M)
>>> ideal_unitary = (U@Vd).conj()
>>> print(ideal_unitary)
[[ 1.        -0.j  0.        -0.j  0.        -0.j  0.        -0.j]
 [ 0.        -0.j  0.70710678-0.j  0.70710678-0.j  0.        -0.j]
 [ 0.        -0.j  0.70710678-0.j -0.70710678-0.j  0.        -0.j]
 [ 0.        -0.j  0.        -0.j  0.        -0.j  0.        -1.j]]
>>> ideal_unitary @ vec_1a
array([0.+0.j, 0.+0.j, 0.+0.j, 0.-1.j])
>>> ideal_unitary @ vec_2a
array([ 0.        +0.j,  0.70710678+0.j, -0.70710678+0.j,  0.        +0.j])
```
