import numpy as np

# class StateVector(np.ndarray):
#     def __new__(cls, input_array):
#         # Convert the input to a numpy array
#         obj = np.asarray(input_array).view(cls)
#         # Set the system size of the state vector
#         obj.L = int(np.rint(np.log2(obj.size)))
#         assert 2**obj.L == obj.size, "State vector size must be a power of 2."
#         return obj
# 
#     def __array_finalize__(self, obj):
#         if obj is None: return
#         self.L = getattr(obj, 'L', None)

class StateVector(object):
    def __init__(self, vector):
        self.state_vector = np.array(vector).flatten()
        self.L = int(np.rint(np.log2(self.state_vector.size)))
        assert 2**self.L == self.state_vector.size, "State vector size must be a power of 2."

    def set_state(self, state):
        if len(state) != self.size:
            raise ValueError("State size does not match vector size.")
        self.state = np.array(state)

    def get_state(self):
        return self.state

    def __repr__(self):
        return f"StateVector({self.state_vector})"

    def __getitem__(self, idx):
        return self.state_vector[idx]

    def copy(self):
        return StateVector(self.state_vector.copy())

    def apply_gate(self, gate, indices):
        '''
        Apply a gate to the state, starting from site idx.
        '''
        if type(indices) == int:
            indices = [indices]

        assert len(gate.shape) // 2 == len(indices), "Gate and indices dimensions do not match."
        if len(indices) == 1:
            self.apply_gate_old(gate, indices[0])
        elif len(indices) == 2:
            idx0 = indices[0]
            idx1 = indices[1]
            assert idx0 != idx1, "Indices must be different."
            if idx0 > idx1:
                idx0, idx1 = idx1, idx0
                gate = np.transpose(gate, [1, 0, 3, 2])  ## swap the two indices

            theta = np.reshape(self.state_vector, [(2**idx0), 2, 2**(idx1-idx0-1), 2, 2**(self.L-(idx1+1))])
            theta = np.tensordot(gate, theta, [[2, 3], [1, 3]])  #[i, j, left, mid, right]
            self.state_vector = np.transpose(theta, [2, 0, 3, 1, 4]).flatten()

            # Create a new state vector and return
            # return StateVector(np.transpose(theta, [2, 0, 3, 1, 4]).flatten())
        else:
            raise ValueError("Only 1 or 2 indices are supported.")

    def apply_gate_old(self, gate, idx):
        '''
        Apply a gate to the state, starting from site idx.
        '''
        num_sites = len(gate.shape) // 2
        gate = np.reshape(gate, (2**num_sites, 2**num_sites))
        gate_dim = gate.shape[0]

        theta = np.reshape(self.state_vector, [(2**idx), gate_dim, 2**(self.L-(idx+num_sites))])
        theta = np.tensordot(gate, theta, [1, 1])  ## [ij] [..., j, ...] --> [i, ..., ...]
        self.state_vector = (np.transpose(theta, [1, 0, 2])).flatten()

    def exp_val(self, operator, indices):
        '''
        Calculate the expectation value of an operator.
        '''
        ket = self.copy()
        ket.apply_gate(operator, indices)
        return np.dot(np.conj(self.state_vector), ket.state_vector)


def overlap(state1, state2):
    '''
    Calculate the overlap between two state vectors.
    '''
    return np.dot(np.conj(state1.state_vector), state2.state_vector)
