import numpy as np
from exact_sim import StateVector
import misc

# General circuit
class GenericCircuit():
    def __init__(self, pairs_of_indices_and_Us):
        '''
        The general circuit is given by a list of pairs of indices and
        unitaries, where the indices are the indices of the qubits that
        the unitaries act on.

        Examples:
        A sequential circuit is defined by
        [((0, 1), U_0), ((1, 2), U_1), ...]

        A brickwork circuit is defined by
        [((0, 1), U_0), ((2, 3), U_1), ((4, 5), U_2), ...]

        A more general circuit can be defined by
        [((0, 1), U_0), ((1, 3), U_1), ((0, 2), U_2), ((2, 3), U_3), ...]
        which represent a circuit acting on 4 qubits on a square lattice.
        '''
        self.pairs_of_indices_and_Us = pairs_of_indices_and_Us
        self.num_gates = len(pairs_of_indices_and_Us)
        self.num_qubits = max([max(pair) for pair, _ in pairs_of_indices_and_Us]) + 1

    def polar_opt(self, list_of_target_states, list_of_initial_states,
                  list_of_bottom_states=None,
                  verbose=False):
        '''
        We find the unitary that maximizes the overlap between the
        target states and the initial states.
        U = ArgMax_U Re[\\sum <vec_target_i | U |vec_initial_i>]

        Perform a sweep from top of the circuit down to the bottom,
        and from bottom of the circuit to the top.
        The algorithm is as follows:
        We utilize the intermediate states including the list of
        top states and the list of bottom states.

        When sweeping from top to bottom, we remove one unitary from
        the current circuit (bottom_states) at a time by applying the
        conjugated unitary to the current states.
        Combining with the target states (top), we form the environment.
        Once a new unitary is obtained, it is then applied to the
        "bra", i.e., the top states.

        When sweeping from bottom to top, we remove at a time one
        unitary from the top states resulting from the previous sweep.
        We remove the unitary by applying the conjugated unitary to the
        top states. Combining with the current state (bottom), we form
        the environment. Once a new unitary is obtained, it is then
        applied to the "ket", i.e., the bottom states.

        In this way, we do not store the intermediate state
        representation. We avoid it by removing the unitary
        iteratively.
        There is no disadvantage in doing so, as the simulation is
        exact.

        Parameters
        ----------
            list_of_target_states: List[np.ndarray]
            the target states

            list_of_initial_states: List[np.ndarray]
            the initial states

            verbose: bool
            whether to print out the error
        '''
        # Preparing the bottom states
        num_states = len(list_of_target_states)
        if list_of_bottom_states is None:
            list_of_bottom_states = [StateVector(init_state) for init_state in list_of_initial_states]
            for gate_idx in range(self.num_gates):
                gate = self.pairs_of_indices_and_Us[gate_idx][1]
                indices = self.pairs_of_indices_and_Us[gate_idx][0]
                gate = np.reshape(gate, [2, 2, 2, 2])
                for state_idx in range(len(list_of_bottom_states)):
                    list_of_bottom_states[state_idx].apply_gate(gate, indices)
        else:
            print('The bottom states are given.')
            assert len(list_of_bottom_states) == num_states
            

        # Preparing the top states
        list_of_top_states = [StateVector(target_state) for target_state in list_of_target_states]

        # Definte the objective function
        def get_cost(num_states, list_of_top_states, list_of_bottom_states):
            cost = num_states
            for state_idx in range(num_states):
                top_vec = list_of_top_states[state_idx].state_vector
                bottom_vec = list_of_bottom_states[state_idx].state_vector
                cost -= np.real(np.dot(top_vec.conj(), bottom_vec))

            return cost

        # Get the initial cost
        cost = get_cost(num_states, list_of_top_states, list_of_bottom_states)
        if verbose:
            print('Initial error:', cost)

        # We now sweep from top to bottom
        for gate_idx in range(self.num_gates-1, -1, -1):
            remove_gate = self.pairs_of_indices_and_Us[gate_idx][1]
            remove_indices = self.pairs_of_indices_and_Us[gate_idx][0]
            remove_gate_conj = remove_gate.reshape([4, 4]).T.conj()
            remove_gate_conj = remove_gate_conj.reshape([2, 2, 2, 2])

            for state_idx in range(num_states):
                list_of_bottom_states[state_idx].apply_gate(remove_gate_conj, remove_indices)

            # Now the bottom states are the states without remove_gate.
            # We can now variational find the optimal gate to update.

            new_gate = var_gate_exact_list_of_states(list_of_top_states, remove_indices,
                                                     list_of_bottom_states,)
            self.pairs_of_indices_and_Us[gate_idx] = (remove_indices, new_gate)

            new_gate_conj = new_gate.reshape([4, 4]).T.conj()
            new_gate_conj = new_gate_conj.reshape([2, 2, 2, 2])
            for state_idx in range(num_states):
                list_of_top_states[state_idx].apply_gate(new_gate_conj, remove_indices)
                
        cost = get_cost(num_states, list_of_top_states, list_of_bottom_states)
        if verbose:
            print('Sweep down to the bottom. The intermediate error:', cost)

        for state_idx in range(num_states):
            assert np.allclose(list_of_bottom_states[state_idx].state_vector,
                               list_of_initial_states[state_idx])

        # We now sweep from bottom to top
        for gate_idx in range(self.num_gates):
            remove_gate = self.pairs_of_indices_and_Us[gate_idx][1]
            remove_indices = self.pairs_of_indices_and_Us[gate_idx][0]

            for state_idx in range(num_states):
                list_of_top_states[state_idx].apply_gate(remove_gate, remove_indices)

            # This remove the gate from top_state
            # Because <\phi | U_{ij} | \psi> = inner( U_{ij}^\dagger |\phi>, |\psi> )
            # applying U would remove the U_{ij}^\dagger, and
            # partial_Tr[ |\phi>, |\psi> ] = Env

            # Now the top states are the states without remove_gate.
            # We can now variational find the optimal gate to update.

            new_gate = var_gate_exact_list_of_states(list_of_top_states, remove_indices,
                                                     list_of_bottom_states,)
            self.pairs_of_indices_and_Us[gate_idx] = (remove_indices, new_gate)

            for state_idx in range(num_states):
                list_of_bottom_states[state_idx].apply_gate(new_gate, remove_indices)

        cost = get_cost(num_states, list_of_top_states, list_of_bottom_states)
        if verbose:
            print('Sweep up to the top. The intermediate error:', cost)

        return cost, list_of_bottom_states


def var_gate_exact(top_state, indices, bottom_state):
    '''
    Given the top state and the bottom state, we find the gate that
    argmax_{gate} <top_state | gate | bottom_state>.

    Parameters
    ----------
        top_state: np.ndarray
        the top state (without complex conjugation)

        indices: (int, int)
        the indices of the gate starts acting on

        bottom_state: np.ndarray
        the bottom state
    '''
    idx0, idx1 = indices
    assert idx0 != idx1
    if idx0 > idx1:
        SWAP = True
        idx0, idx1 = idx1, idx0
    else:
        SWAP = False

    L = top_state.L
    top_theta = np.reshape(top_state.state_vector, [2 ** idx0, 2, 2 ** (idx1-(idx0+1)), 2, 2 ** (L-(idx1+1))])
    bottom_theta = np.reshape(bottom_state.state_vector, [2 ** idx0, 2, 2 ** (idx1-(idx0+1)), 2, 2 ** (L-(idx1+1))])
    # [left, i, mid, j, right]

    # We want to find the gate that maximizes the overlap
    # <top_state | gate | bottom_state>
    M = np.tensordot(top_theta.conj(), bottom_theta, axes=([0, 2, 4], [0, 2, 4]))
    M = np.reshape(M, [4, 4])
    # [ ..., upper_p, ...], [ ..., lower_p, ...] -> [upper_p, lower_p]

    U, _, Vd = misc.svd(M, full_matrices=False)
    new_gate = np.dot(U, Vd).conj()
    new_gate = new_gate.reshape([2, 2, 2, 2])
    if SWAP:
        new_gate = np.transpose(new_gate, [1, 0, 3, 2])

    return new_gate

def var_gate_exact_old(top_state, idx, bottom_state):
    '''
    Given the top state and the bottom state, we find the gate that
    argmax_{gate} <top_state | gate | bottom_state>.

    Parameters
    ----------
        top_state: np.ndarray
        the top state (without complex conjugation)

        idx: int
        the index of the gate starts acting on

        bottom_state: np.ndarray
        the bottom state
    '''
    L = top_state.L
    top_theta = np.reshape(top_state.state_vector, [(2 ** idx), 4, 2 ** (L - idx - 2)])
    bottom_theta = np.reshape(bottom_state.state_vector, [2 ** idx, 4, 2 ** (L - idx - 2)])

    # We want to find the gate that maximizes the overlap
    # <top_state | gate | bottom_state>
    M = np.tensordot(top_theta.conj(), bottom_theta, axes=([0, 2], [0, 2]))
    # [ ..., upper_p, ...], [ ..., lower_p, ...] -> [upper_p, lower_p]

    U, _, Vd = misc.svd(M, full_matrices=False)
    new_gate = np.dot(U, Vd).conj()
    new_gate = new_gate.reshape([2, 2, 2, 2])
    return new_gate

def var_gate_exact_list_of_states(top_states, indices, bottom_states):
    '''
    Given the top state and the bottom state, we find the gate that
    argmax_{gate} <top_state | gate | bottom_state>.

    Parameters
    ----------
        top_states: List[np.ndarray]
            the list of top states (without complex conjugation)

        indices: (int, int)
        the indices of the gate starts acting on

        bottom_states: List[np.ndarray]
            the list of bottom states
    '''
    idx0, idx1 = indices
    assert idx0 != idx1
    if idx0 > idx1:
        SWAP = True
        idx0, idx1 = idx1, idx0
    else:
        SWAP = False

    assert len(top_states) == len(bottom_states)

    L = top_states[0].L
    sum_of_Ms = np.zeros([4, 4], dtype=np.complex128)
    for state_idx in range(len(top_states)):
        top_state = top_states[state_idx]
        bottom_state = bottom_states[state_idx]

        top_theta = np.reshape(top_state.state_vector, [2 ** idx0, 2, 2 ** (idx1-(idx0+1)), 2, 2 ** (L-(idx1+1))])
        bottom_theta = np.reshape(bottom_state.state_vector, [2 ** idx0, 2, 2 ** (idx1-(idx0+1)), 2, 2 ** (L-(idx1+1))])
        # [left, i, mid, j, right]

        # We want to find the gate that maximizes the overlap
        # <top_state | gate | bottom_state>
        M = np.tensordot(top_theta.conj(), bottom_theta, axes=([0, 2, 4], [0, 2, 4]))
        M = np.reshape(M, [4, 4])
        # [ ..., upper_p, ...], [ ..., lower_p, ...] -> [upper_p, lower_p]
        sum_of_Ms = sum_of_Ms + M

    U, _, Vd = misc.svd(sum_of_Ms, full_matrices=False)
    new_gate = (U @ Vd).conj()
    new_gate = new_gate.reshape([2, 2, 2, 2])
    if SWAP:
        new_gate = np.transpose(new_gate, [1, 0, 3, 2])

    return new_gate


