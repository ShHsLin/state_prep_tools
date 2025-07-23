import numpy as np
import scipy
import copy
import stateprep.utils.misc as misc
from stateprep.exact_sim import StateVector
from stateprep.gate import U1UnitaryGate, UnitaryGate, FermionicGate, FreeFermionGate

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

        self.num_trainable_gates = self.trainable.count(True)

    def get_params(self):
        raise NotImplementedError('get_params is not implemented.')

    def set_params(self, params):
        raise NotImplementedError('set_params is not implemented.')

    def save_pairs_of_indices_and_Us(self, filename):
        '''
        Save the pairs of indices and unitaries to a file.
        We cast the unitaries back to np.ndarray.
        '''
        import pickle
        with open(filename, "wb") as f:
            pairs_of_indices_and_Us = [(idx, np.array(U)) for idx, U in self.pairs_of_indices_and_Us]
            pickle.dump(pairs_of_indices_and_Us, f)

    def copy(self):
        """Create a deep copy of the circuit."""
        copied_pairs = [(tuple(indices), copy.deepcopy(U)) for indices, U in self.pairs_of_indices_and_Us]
        copied_trainable = list(self.trainable)
        return self.__class__(copied_pairs, trainable=copied_trainable)


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
        qubit_circuit_pairs_of_indices_and_Us = []
        qubit_circuit_trainable = []

        for idx, indices_U in enumerate(self.pairs_of_indices_and_Us):
            indices, U = indices_U
            assert indices[0] < indices[1]
            if indices[1] - indices[0] == 1:
                qubit_circuit_pairs_of_indices_and_Us.append(indices_U)
                qubit_circuit_trainable.append(self.trainable[idx])
            else:
                # We need to apply fSWAP gates
                # Apply the fSWAP gate to move the left qubit
                for move_idx in range(indices[0], indices[1]-1):
                    pair = ((move_idx, move_idx+1), self.fSWAP)
                    qubit_circuit_pairs_of_indices_and_Us.append(pair)
                    qubit_circuit_trainable.append(False)

                # Apply the unitary
                pair = ((indices[1]-1, indices[1]), U)
                qubit_circuit_pairs_of_indices_and_Us.append(pair)
                qubit_circuit_trainable.append(self.trainable[idx])

                # Apply the fSWAP gate to move back the left qubit
                for move_idx in range(indices[1]-1, indices[0], -1):
                    pair = ((move_idx-1, move_idx), self.fSWAP)
                    qubit_circuit_pairs_of_indices_and_Us.append(pair)
                    qubit_circuit_trainable.append(False)

        # Now we have a qubit circuit
        qubit_circuit = QubitCircuit(qubit_circuit_pairs_of_indices_and_Us,
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
                print(f"Fixed gate {idx}: {indices} {U.decompose_fermionic_gate()}")

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
                    # [TODO] We should implement a function to identify the class
                    # [TODO] We should implement test assert in the gate class to avoid wrongly assign class

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
                    # [TODO] We should implement a function to identify the class
                    # [TODO] We should implement test assert in the gate class to avoid wrongly assign class

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

            if not self.trainable[gate_idx]:
                pass
                # If the gate is not trainable, we still need
                # to apply the gate to the state vectors
            else:
                ####################################################
                # Compute the environment for the gate
                ####################################################
                env = misc.get_env(top_vec.state_vector, bottom_vec.state_vector, indices, self.num_qubits)
                list_of_envs[gate_idx] = env

            Ud = U.T.conj().reshape([2, 2, 2, 2])
            U = np.reshape(U, [2, 2, 2, 2])

            bottom_vec.apply_gate(Ud, indices)  # Contract Ud to |bottom>
            top_vec.apply_gate(Ud, indices)  # Contract Ud to |top>

            # This is because we have the structure
            # <top | bottom> = < new_top | Ud | bottom> = < new_top | new_bottom>
            # So <top| = <new_top| Ud, and Ud |bottom> = |new_bottom>
            # |top> = U |new_top>
            # Ud |top> = |new_top>

        assert np.isclose(E, top_vec.state_vector.conj() @ bottom_vec.state_vector)

        # The gradient is given by the derivative of the energy with
        # respect to the parameters dE / dp = dE/d(Ud) * d(Ud)/dp
        grads = []
        for gate_idx in range(self.num_gates):
            if not self.trainable[gate_idx]:
                continue

            env = list_of_envs[gate_idx].reshape([4, 4])
            U = self.pairs_of_indices_and_Us[gate_idx][1]
            # U(i,[j]) env_([j],i)
            dUd_mat = U.T @ env
            ## cost = f(Ud) = Tr (Ud @ dUd_mat)
            U_grad = U.get_gradient(dUd_mat)
            grads.append(U_grad)

        assert len(grads) == self.num_trainable_gates, \
            f"Expected {self.num_trainable_gates} gradients, got {len(grads)}"

        return grads

def transform_to_nearest_neighbor_qubit_circuit(qubit_circuit):
    '''
    Transform a generic qubit circuit, which may have unitaries
    acting on non-neighboring qubits, to a nearest neighbor qubit
    circuit. This is done by applying SWAP gates to move the unitaries
    to nearest neighbor pairs.

    Parameters
    ----------
        qubit_circuit: QubitCircuit
        the qubit circuit

    Returns
    -------
        QubitCircuit
        the nearest neighbor qubit circuit
    '''
    SWAP = np.array([[1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1]],).reshape([2, 2, 2, 2])

    if not isinstance(qubit_circuit, QubitCircuit):
        raise TypeError("Input must be a QubitCircuit instance.")

    if qubit_circuit.num_qubits < 2:
        # No need to transform if there are less than 2 qubits
        return qubit_circuit

    # Begin with an empty list of pairs and trainable flags
    new_pairs_of_indices_and_Us = []
    new_trainable = []

    for idx, (indices, U) in enumerate(qubit_circuit.pairs_of_indices_and_Us):
        assert indices[0] < indices[1], "Indices must be in ascending order."

        if indices[1] - indices[0] == 1:
            # If the unitary is already nearest neighbor, keep it
            new_pairs_of_indices_and_Us.append((indices, U))
            new_trainable.append(qubit_circuit.trainable[idx])
        else:
            # We need to apply SWAP gates to move the left qubit
            for move_idx in range(indices[0], indices[1]-1):
                new_pairs_of_indices_and_Us.append(((move_idx, move_idx+1), SWAP))
                new_trainable.append(False)

            # Apply the unitary
            new_pairs_of_indices_and_Us.append(((indices[1]-1, indices[1]), U))
            new_trainable.append(qubit_circuit.trainable[idx])

            # Apply the SWAP gates to move back the left qubit
            for move_idx in range(indices[1]-1, indices[0], -1):
                new_pairs_of_indices_and_Us.append(((move_idx-1, move_idx), SWAP))
                new_trainable.append(False)

    # Now we have a nearest neighbor qubit circuit
    nearest_neighbor_circuit = QubitCircuit(new_pairs_of_indices_and_Us, new_trainable)
    return nearest_neighbor_circuit

def removing_SWAP_from_nearest_neighbor_qubit_circuit(qubit_circuit):
    '''
    Simplify a nearest neighbor qubit circuit by removing the SWAP gates
    that are from tansforming a generic qubit circuit to a nearest neighbor
    qubit circuit. This function is similar to the export_FermionicCircuit
    function, but we pull out the SWAP gates here.

    Parameters
    ----------
        qubit_circuit: QubitCircuit
        the nearest neighbor qubit circuit

    Returns
    -------
        QubitCircuit
        the simplified nearest neighbor qubit circuit
    '''
    if not isinstance(qubit_circuit, QubitCircuit):
        raise TypeError("Input must be a QubitCircuit instance.")

    new_pairs_of_indices_and_Us = []
    new_trainable = []
    idx = 0
    SWAP = np.array([[1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1]],).reshape([2, 2, 2, 2])

    while idx < qubit_circuit.num_gates:
        indices, U = qubit_circuit.pairs_of_indices_and_Us[idx]
        is_trainable = qubit_circuit.trainable[idx]

        if not is_trainable and np.allclose(U, SWAP):
            # Possibly beginning of SWAP sequence
            start = indices[0]
            # forward SWAPs
            forward = []
            while idx < qubit_circuit.num_gates and not qubit_circuit.trainable[idx] and \
                  np.allclose(qubit_circuit.pairs_of_indices_and_Us[idx][1], SWAP):
                forward.append(qubit_circuit.pairs_of_indices_and_Us[idx][0])
                idx += 1

            # Unitary application
            if idx >= qubit_circuit.num_gates or (not qubit_circuit.trainable[idx]):
                raise ValueError("Malformed SWAP sequence: expected trainable unitary after forward swaps")

            unitary_indices, U_actual = qubit_circuit.pairs_of_indices_and_Us[idx]
            if type(U_actual) == np.ndarray:
                U_actual = UnitaryGate(U_actual.reshape([4, 4]))

            is_unitary_trainable = qubit_circuit.trainable[idx]
            idx += 1

            # backward SWAPs
            backward = []
            while idx < qubit_circuit.num_gates and not qubit_circuit.trainable[idx] and \
                  np.allclose(qubit_circuit.pairs_of_indices_and_Us[idx][1], SWAP):
                backward.append(qubit_circuit.pairs_of_indices_and_Us[idx][0])
                idx += 1
                if len(backward) == len(forward):
                    break

            # Check that forward and backward swaps are mirror images
            if forward != list(reversed(backward)):
                raise ValueError("Inconsistent SWAP pattern: forward and backward swaps don't match")

            # Reconstruct pair
            left = start
            right = unitary_indices[1]
            new_pairs_of_indices_and_Us.append(((left, right), U_actual))
            new_trainable.append(is_unitary_trainable)
        else:
            # No SWAP: must be adjacent gate
            if indices[1] - indices[0] != 1:
                raise ValueError(f"Unexpected gate without SWAP at non-adjacent qubits: {indices}")
            if type(U) == np.ndarray:
                U = UnitaryGate(U.reshape([4, 4]))

            new_pairs_of_indices_and_Us.append((indices, U))
            new_trainable.append(is_trainable)
            idx += 1

    new_circ = QubitCircuit(new_pairs_of_indices_and_Us, new_trainable)
    return new_circ



if __name__ == "__main__":
    # Example usage
    pairs = [((0, 5), U1UnitaryGate(np.eye(4))),
             ((1, 2), U1UnitaryGate(np.eye(4))),
             ((0, 2), U1UnitaryGate(np.eye(4)))]
    circuit = QubitCircuit(pairs)
    print("------")
    for pair in circuit.pairs_of_indices_and_Us:
        print(pair[0], type(pair[1]))

    print("------")

    print("------")
    transformed_circuit = transform_to_nearest_neighbor_qubit_circuit(circuit)
    for pair in transformed_circuit.pairs_of_indices_and_Us:
        print(pair[0], type(pair[1]))
    print("------")


    print("------")
    simplified_circuit = removing_SWAP_from_nearest_neighbor_qubit_circuit(transformed_circuit)
    for pair in simplified_circuit.pairs_of_indices_and_Us:
        print(pair[0], type(pair[1]))
    print("------")


    # print(circuit.to_state_vector())
    # print(circuit.get_params())
    # circuit.set_params(np.random.rand(len(circuit.get_params()), 6))
    # print(circuit.to_state_vector())
