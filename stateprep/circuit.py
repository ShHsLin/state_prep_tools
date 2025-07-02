import numpy as np
import scipy
import stateprep.utils.misc as misc
from stateprep.exact_sim import StateVector
from stateprep.gate import U1UnitaryGate

# Base class for circuits
class Circuit():
    def __init__(self, pairs_of_indices_and_Us, trainable=None):
        '''
        The circuit is given by a list of pairs of indices and
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
        if trainable is None:
            self.trainable = [True] * self.num_gates
        else:
            assert type(trainable) == list
            assert len(trainable) == self.num_gates
            self.trainable = trainable

    def get_params(self):
        raise NotImplementedError('get_params is not implemented.')

    def set_params(self, params):
        raise NotImplementedError('set_params is not implemented.')


# Fermionic circuit
class FermionicCircuit(Circuit):
    def __init__(self, pairs_of_indices_and_Us, trainable=None):
        super().__init__(pairs_of_indices_and_Us, trainable)
        self.fSWAP = np.array([[1, 0, 0, 0],
                               [0, 0, 1, 0],
                               [0, 1, 0, 0],
                               [0, 0, 0, -1]],).reshape([2, 2, 2, 2])

    def export_QubitCircuit(self):
        '''
        Decorate the fermionic gates with fermionic swap gates.
        The resulting circuit is a qubit circuit.
        '''
        qubit_cirucit_pairs_of_indices_and_Us = []
        qubit_circuit_trainable = []

        for idx, indices_U in enumerate(self.pairs_of_indices_and_Us):
            indices, U = indices_U
            assert indices[0] < indices[1]
            if indices[1] - indices[0] == 1:
                qubit_cirucit_pairs_of_indices_and_Us.append(indices_U)
                qubit_circuit_trainable.append(self.trainable[idx])
            else:
                # We need to apply fSWAP gates
                # Apply the fSWAP gate to move the left qubit
                for move_idx in range(indices[0], indices[1]-1):
                    pair = ((move_idx, move_idx+1), self.fSWAP)
                    qubit_cirucit_pairs_of_indices_and_Us.append(pair)
                    qubit_circuit_trainable.append(False)

                # Apply the unitary
                pair = ((indices[1]-1, indices[1]), U)
                qubit_cirucit_pairs_of_indices_and_Us.append(pair)
                qubit_circuit_trainable.append(self.trainable[idx])

                # Apply the fSWAP gate to move back the left qubit
                for move_idx in range(indices[1]-1, indices[0], -1):
                    pair = ((move_idx-1, move_idx), self.fSWAP)
                    qubit_cirucit_pairs_of_indices_and_Us.append(pair)
                    qubit_circuit_trainable.append(False)

        # Now we have a qubit circuit
        qubit_circuit = QubitCircuit(qubit_cirucit_pairs_of_indices_and_Us,
                                     qubit_circuit_trainable)
        return qubit_circuit

    def print_fermionic_params(self):
        '''
        Print the parameters of the fermionic circuit.
        '''

        print("=================     Id   | XX+YY  | XY-YX |  ZZ  |  Z1  |  Z2  ==============")

        for idx, indices_U in enumerate(self.pairs_of_indices_and_Us):
            indices, U = indices_U

            if self.trainable[idx]:
                print(f"Trainable gate[{idx}]: {U.decompose_fermionic_gate()} between {indices[0]} and {indices[1]}")
            else:
                print(f"Fixed gate {idx}: {indices} {U.decopmose_fermionic_gate()}")

        print("=================     Id   |  hop   |  cur  | n1n2 |  n1  |  n2  ==============")



# Quantum circuit that acts on qubits
class QubitCircuit(Circuit):
    def __init__(self, pairs_of_indices_and_Us, trainable=None):
        super().__init__(pairs_of_indices_and_Us, trainable)

    def to_state_vector(self, init_state=None):
        '''
        Given the initial state, we apply the circuit to the initial
        state and return the final state.

        Parameters
        ----------
            init_state: np.ndarray
            the initial state

        Returns
        -------
            np.ndarray
            the final state
        '''
        if init_state is None:
            init_state = np.zeros([2 ** self.num_qubits])
            init_state[0] = 1

        iter_state = StateVector(init_state)

        # Apply the circuit to the initial state
        for gate_idx in range(self.num_gates):
            gate = self.pairs_of_indices_and_Us[gate_idx][1]
            indices = self.pairs_of_indices_and_Us[gate_idx][0]
            gate = np.reshape(gate, [2, 2, 2, 2])
            iter_state.apply_gate(gate, indices)

        return iter_state.state_vector

    def get_params(self):
        '''
        Get the parameters of the circuit.

        Returns
        -------
            params: List[np.ndarray]
            the parameters of the circuit
        '''
        params = []
        for idx, indices_U in enumerate(self.pairs_of_indices_and_Us):
            indices, U = indices_U
            if self.trainable[idx]:
                params.append(U.get_parameters())

        return params

    def set_params(self, params):
        '''
        Set the parameters of the circuit.

        Parameters
        ----------
            params: List[np.ndarray]
            the parameters of the circuit
        '''
        if params.ndim == 1:
            params = params.reshape(self.trainable.count(True), -1)
            assert params.shape[1] in [6, 16]

        assert len(params) == self.trainable.count(True)
        params_idx = 0
        for idx, indices_U in enumerate(self.pairs_of_indices_and_Us):
            indices, U = indices_U
            if self.trainable[idx]:
                U.set_parameters(params[params_idx])
                params_idx += 1

        assert params_idx == len(params)
        return

    def export_FermionicCircuit(self):
        '''
        Given the qubit circuit implementing an underlying fermionic
        circuit with fSWAP gates, we now remove the fSWAP gates and
        return the fermionic circuit.

        Parameters
        ----------
            None

        Returns
        -------
            FermionicCircuit
            the fermionic circuit
        '''
        new_pairs_of_indices_and_Us = []
        new_trainable = []
        idx = 0

        fSWAP = np.array([[1, 0, 0, 0],
                          [0, 0, 1, 0],
                          [0, 1, 0, 0],
                          [0, 0, 0, -1]],).reshape([2, 2, 2, 2])

        while idx < self.num_gates:
            indices, U = self.pairs_of_indices_and_Us[idx]
            is_trainable = self.trainable[idx]

            if not is_trainable and np.allclose(U, fSWAP):
                # Possibly beginning of fSWAP sequence
                start = indices[0]
                # forward fSWAPs
                forward = []
                while idx < self.num_gates and not self.trainable[idx] and \
                      np.allclose(self.pairs_of_indices_and_Us[idx][1], fSWAP):
                    forward.append(self.pairs_of_indices_and_Us[idx][0])
                    idx += 1


                # Unitary application
                if idx >= self.num_gates or (not self.trainable[idx]):
                    raise ValueError("Malformed fSWAP sequence: expected trainable unitary after forward swaps")

                unitary_indices, U_actual = self.pairs_of_indices_and_Us[idx]
                if type(U_actual) == np.ndarray:
                    U_actual = U1UnitaryGate(U_actual.reshape([4, 4]))
                    # [TODO] we should actualy define a fermionic gate class

                is_unitary_trainable = self.trainable[idx]
                idx += 1

                # backward fSWAPs
                backward = []
                while idx < self.num_gates and not self.trainable[idx] and \
                      np.allclose(self.pairs_of_indices_and_Us[idx][1], fSWAP):
                    backward.append(self.pairs_of_indices_and_Us[idx][0])
                    idx += 1
                    if len(backward) == len(forward):
                        break

                # Check that forward and backward swaps are mirror images
                if forward != list(reversed(backward)):
                    raise ValueError("Inconsistent fSWAP pattern: forward and backward swaps don't match")

                # Reconstruct fermionic pair
                left = start
                right = unitary_indices[1]
                # print("adding gates :", idx, "unitary indices = ", (left, right))
                new_pairs_of_indices_and_Us.append(((left, right), U_actual))
                new_trainable.append(is_unitary_trainable)
            else:
                # print("adding gates :", idx, "unitary indices = ", indices)
                # No fSWAP: must be adjacent gate
                if indices[1] - indices[0] != 1:
                    raise ValueError(f"Unexpected gate without fSWAP at non-adjacent qubits: {indices}")

                if type(U) == np.ndarray:
                    U = U1UnitaryGate(U.reshape([4, 4]))
                    # [TODO] we should actualy define a fermionic gate class

                new_pairs_of_indices_and_Us.append((indices, U))
                new_trainable.append(is_trainable)
                idx += 1

        new_circ = FermionicCircuit(new_pairs_of_indices_and_Us, new_trainable)
        return new_circ

    def get_energy(self, H, init_state=None):
        '''
        Given the Hamiltonian, we apply the circuit to the initial
        state and return the final state.

        Parameters
        ----------
            H: np.ndarray
            the Hamiltonian

        Returns
        -------
            energy: np.float
        '''
        state_vec = self.to_state_vector(init_state)
        return state_vec.conj() @ H @ state_vec

    def get_energy_gradient(self, H, init_state=None):
        '''
        Given the Hamiltonian, we apply the circuit to the initial
        state and return the gradient with respect to the energy.

        Parameters
        ----------
            H: np.ndarray
            the Hamiltonian

        Returns
        -------
            energy_gradient: np.ndarray
            the energy gradient
        '''
        state_vec = self.to_state_vector(init_state)
        bottom_vec = StateVector(H @ state_vec)
        top_vec = StateVector(state_vec)  # not yet conjugated

        E = top_vec.state_vector.conj() @ bottom_vec.state_vector

        list_of_envs = [None] * self.num_gates
        for gate_idx in range(self.num_gates-1, -1, -1):
            indices, U = self.pairs_of_indices_and_Us[gate_idx]

            idx0, idx1 = indices
            if idx0 > idx1:
                SWAP = True
                idx0, idx1 = idx1, idx0
            else:
                SWAP = False

            top_theta = np.reshape(top_vec.state_vector,
                                   [(2**idx0), 2, 2**(idx1-idx0-1), 2, 2**(self.num_qubits-(idx1+1))])
            bottom_theta = np.reshape(bottom_vec.state_vector,
                                      [(2**idx0), 2, 2**(idx1-idx0-1), 2, 2**(self.num_qubits-(idx1+1))])
            # [left, i, mid, j, right]
            env = np.tensordot(top_theta.conj(), bottom_theta, axes=([0, 2, 4], [0, 2, 4]))
            if SWAP:
                env = np.transpose(env, [1, 0, 3, 2])

            list_of_envs[gate_idx] = env

            Ud = U.T.conj().reshape([2, 2, 2, 2])
            U = np.reshape(U, [2, 2, 2, 2])

            # Contract Ud to |bottom>
            bottom_vec.apply_gate(Ud, indices)
            # Contract Ud to |top>
            top_vec.apply_gate(Ud, indices)

            # This is because we have the structure
            # <top | bottom> = < new_top | Ud | bottom> = < new_top | new_bottom>
            # So <top| = <new_top| Ud, and Ud |bottom> = |new_bottom>
            # |top> = U |new_top>
            # Ud |top> = |new_top>

        assert np.isclose(E, top_vec.state_vector.conj() @ bottom_vec.state_vector)

        # The gradient is given by the derivative of the energy with
        # respect to the parameters
        # dE/dp = dE/dU * dU/dp
        grads = []
        for gate_idx in range(self.num_gates):
            env = list_of_envs[gate_idx].reshape([4, 4])
            U = self.pairs_of_indices_and_Us[gate_idx][1]
            # U(i,[j]) env_([j],i)
            dU_mat = U.T @ env
            U_grad = U.get_gradient(dU_mat)
            grads.append(U_grad)

        return grads

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
            # print('The bottom states are given.')
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

            if self.trainable[gate_idx]:
                # We only update the trainable gates
                new_gate = var_gate_exact_list_of_states(list_of_top_states, remove_indices,
                                                         list_of_bottom_states,)
                self.pairs_of_indices_and_Us[gate_idx] = (remove_indices, new_gate)
            else:
                new_gate = self.pairs_of_indices_and_Us[gate_idx][1]

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

            if self.trainable[gate_idx]:
                new_gate = var_gate_exact_list_of_states(list_of_top_states, remove_indices,
                                                         list_of_bottom_states,)
                self.pairs_of_indices_and_Us[gate_idx] = (remove_indices, new_gate)
            else:
                new_gate = self.pairs_of_indices_and_Us[gate_idx][1]

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
    [TODO] Remove the code. The code is old and only supported
    indices of the form (i, i+1).

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

    U1 = False
    if U1:
        masked_sum_of_Ms = np.zeros([4, 4], dtype=np.complex128)
        masked_sum_of_Ms[0, 0] = sum_of_Ms[0, 0]
        masked_sum_of_Ms[1:3, 1:3] = sum_of_Ms[1:3, 1:3]
        masked_sum_of_Ms[3, 3] = sum_of_Ms[3, 3]

        masked_sum_of_Ms[0, 3] = sum_of_Ms[0, 3]
        masked_sum_of_Ms[3, 0] = sum_of_Ms[3, 0]

        U, _, Vd = misc.svd(masked_sum_of_Ms, full_matrices=False)
        new_gate = (U @ Vd).conj()
        new_gate = new_gate.reshape([2, 2, 2, 2])
    else:
        U, _, Vd = misc.svd(sum_of_Ms, full_matrices=False)
        new_gate = (U @ Vd).conj()
        new_gate = new_gate.reshape([2, 2, 2, 2])

    if SWAP:
        new_gate = np.transpose(new_gate, [1, 0, 3, 2])

    return new_gate



