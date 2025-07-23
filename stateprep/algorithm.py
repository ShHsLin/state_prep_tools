import numpy as np
import scipy
import stateprep.utils.misc as misc
from stateprep.exact_sim import StateVector

"""
Here we provide high-level functions for the optimization of
qubit circuits. The functions are designed to be used with
QubitCircuit objects.
Currently, we provide the functions utilizing
state vector simulation. We plan to add also matrix-product
states and other simulation methods.

In principle, due to the different structure and nature
of the simulation. We provide hard coded code for the
different methods.
"""

def fidelity_maximization(qubit_circuit,
                          target_state,
                          initial_state=None,
                          method='polar_decomposition',
                          num_steps=1000,
                          verbose=False,
                          ):
    # One target state. Calling one of the basis transformation functions.
    if method == 'polar_decomposition':
        qubit_circuit, info = basis_transformation_with_polar_decomposition(qubit_circuit,
                                                                            [target_state, ],
                                                                            [initial_state, ],
                                                                            num_steps=num_steps,
                                                                            verbose=verbose)
    elif method == 'gradient_descent':
        qubit_circuit, info = basis_transformation_with_gradient_descent(qubit_circuit,
                                                                         [target_state,],
                                                                         [initial_state,],
                                                                         num_steps=num_steps,
                                                                         verbose=verbose,
                                                                         )
    else:
        raise ValueError("Unknown method: {}".format(method))

    return qubit_circuit, info

def basis_transformation_with_polar_decomposition(qubit_circuit,
                                                  list_of_target_states,
                                                  list_of_initial_states,
                                                  num_steps=1000,
                                                  verbose=False,
                                                  ):
    """
    Perform basis transformation using polar decomposition on a given qubit circuit.
    Parameters
    ----------
    qubit_circuit : QubitCircuit
        The qubit circuit to optimize.
    list_of_target_states : List[np.ndarray]
        The target states to map to.
    list_of_initial_states : List[np.ndarray]
        The initial states to map from.
    num_steps : int, optional
    verbose : bool, optional

    Returns
    -------
    qubit_circuit : QubitCircuit
        The optimized qubit circuit after basis transformation.
    info : dict
        Information about the optimization process, including convergence status and cost.
    """
    C = qubit_circuit.copy()

    cost, list_of_bottom_states = polar_opt(C,
                                            list_of_target_states,
                                            list_of_initial_states,
                                            verbose=verbose,)

    converged = False
    for i in range(num_steps):
        cost, list_of_bottom_states = polar_opt(C,
                                                list_of_target_states,
                                                list_of_initial_states,
                                                list_of_bottom_states=list_of_bottom_states,
                                                verbose=verbose,
                                                )
        if cost < 1e-10:
            converged = True
            break

    info = {'converged': converged, 'cost': cost, 'num_steps': i}
    return C, info

def basis_transformation_with_gradient_descent(qubit_circuit,
                                               list_of_target_states,
                                               list_of_initial_states,
                                               num_steps=1000,
                                               verbose=False,
                                               ):
    iter_circ = qubit_circuit.copy()
    iter_circ_params = iter_circ.get_concatenated_params()

    def f_and_g(params):
        iter_circ.set_params(params)
        cost, grads = get_fidelity_gradient(iter_circ,
                                            list_of_target_states,
                                            list_of_initial_states,
                                            verbose=verbose)
        g = np.concatenate([grad.flatten() for grad in grads])
        return cost, g


    result = scipy.optimize.minimize(f_and_g,
                                     iter_circ_params,
                                     method='L-BFGS-B',
                                     options={'disp': verbose},
                                     jac=True)

    iter_circ_params = result.x
    iter_circ.set_params(iter_circ_params)
    return iter_circ, result

def energy_minimization_with_polar_decomposition(qubit_circuit,
                                                 Hamiltonian,
                                                 verbose=False,
                                                 ):
    """
    Perform energy minimization using polar decomposition on a given qubit circuit.
    Parameters
    ----------
    qubit_circuit : QubitCircuit
        The qubit circuit to optimize.
    Hamiltonian : scipy.sparse.csc_matrix
    verbose : bool, optional

    Returns
    -------
    qubit_circuit : QubitCircuit
        The optimized qubit circuit after energy minimization.
    """
    raise NotImplementedError("This function is not implemented yet.")

def energy_minimization_with_gradient_descent(qubit_circuit,
                                              Hamiltonian,
                                              init_vec=None,
                                              method='L-BFGS-B',
                                              verbose=False,
                                              ):
    """
    Perform energy minimization using gradient descent on a given qubit circuit.
    Parameters
    ----------
    qubit_circuit : QubitCircuit
        The qubit circuit to optimize.
    Hamiltonian : scipy.sparse.csc_matrix
    method : str, optional
    verbose : bool, optional

    Returns
    -------
    qubit_circuit : QubitCircuit
        The optimized qubit circuit after energy minimization.
    """
    iter_circ = qubit_circuit.copy()
    iter_circ_params = iter_circ.get_concatenated_params()
    H = Hamiltonian
    E = iter_circ.get_energy(H, init_vec)

    def f(params):
        iter_circ.set_params(params)
        E = iter_circ.get_energy(H, init_vec)
        return E

    def f_and_g(params):
        iter_circ.set_params(params)
        E = iter_circ.get_energy(H, init_vec)
        grads = iter_circ.get_energy_gradient(H, init_vec)
        g = np.concatenate([grad.flatten() for grad in grads])
        return E.real, g

    # result = scipy.optimize.minimize(f, iter_circ_params, method='L-BFGS-B', options={'disp': True})
    result = scipy.optimize.minimize(f_and_g, iter_circ_params,
                                     method='L-BFGS-B',
                                     options={'disp': verbose},
                                     jac=True)
    iter_circ_params = result.x
    iter_circ.set_params(iter_circ_params)
    return iter_circ

def get_fidelity_gradient(qubit_circuit,
                          list_of_target_states,
                          list_of_initial_states,
                          verbose=False):
    '''
    We compute the gradient of the unitary parameters in temrs of
    the sum of fidelity of the target states and the initial states.
    U = ArgMax_U [ \\sum |<vec_target_i | U |vec_initial_i>|^2 ]

    d cost      d cost    d Ud
    ------  =   ------ *  ----
    d p         d Ud      d p
    where p are the parameters of the unitary.
    The first term is
    v1: - sum_i < target_state_i | bottom_state_i > * < d(Ud) bottom_state_i | target_state_i >
    v2: - sum_i < target_state_i | dU bottom_state_i > * < bottom_state_i | target_state_i >

    We implement v2 now.

    . We perform a sweep from bottom to top to collect all bottom states.
    . |bottom_state_i> = U |initial_state_i>
    . We compute <vec_target_i | bottom_state_i> for each i.
    . We perform a sweep from top back tot bottom to get all the
      gradient environments.
    . We compute the gradients for all the Us.

    The algorithm is as follows:
    We utilize the intermediate states including the list of
    top states and the list of bottom states.

    When sweeping from top to bottom, we remove one unitary from
    the current circuit (bottom_states) at a time by applying the
    conjugated unitary to the current states.
    Combining with the target states (top), we form the environment.
    Once a new unitary is obtained, it is then applied to the
    "bra", i.e., the top states.

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

    list_of_bottom_states = [StateVector(initial_state) for initial_state in list_of_initial_states]
    for gate_idx in range(qubit_circuit.num_gates):
        gate = qubit_circuit.pairs_of_indices_and_Us[gate_idx][1]
        indices = qubit_circuit.pairs_of_indices_and_Us[gate_idx][0]
        gate = np.reshape(gate, [2, 2, 2, 2])
        for state_idx in range(len(list_of_bottom_states)):
            list_of_bottom_states[state_idx].apply_gate(gate, indices)

    # Preparing the top states
    list_of_top_states = [StateVector(target_state) for target_state in list_of_target_states]

    overlap_list = []
    cost = num_states
    for state_idx in range(num_states):
        top_vec = list_of_top_states[state_idx].state_vector
        bottom_vec = list_of_bottom_states[state_idx].state_vector
        # < target_state_i | bottom_state_i >
        overlap = np.dot(top_vec.conj(), bottom_vec)
        overlap_list.append(overlap)
        cost -= np.square(np.abs(overlap))

    if verbose:
        print('Error:', cost)

    list_of_envs_wo_U = [None] * qubit_circuit.num_gates
    # We now sweep from top to bottom
    for gate_idx in range(qubit_circuit.num_gates-1, -1, -1):
        remove_gate = qubit_circuit.pairs_of_indices_and_Us[gate_idx][1]
        remove_indices = qubit_circuit.pairs_of_indices_and_Us[gate_idx][0]
        remove_gate_conj = remove_gate.reshape([4, 4]).T.conj()
        remove_gate_conj = remove_gate_conj.reshape([2, 2, 2, 2])

        for state_idx in range(num_states):
            list_of_bottom_states[state_idx].apply_gate(remove_gate_conj, remove_indices)

        # Now the bottom states are the states without remove_gate.
        # We can now get the environment without the remove_gate.

        # --------------------------------------------------------------------
        if not qubit_circuit.trainable[gate_idx]:
            # we don't need to compute the environment
            pass
        else:
            weighted_sum_env = np.zeros([2, 2, 2, 2], dtype=np.complex128)
            for state_idx in range(num_states):
                env = misc.get_env(list_of_top_states[state_idx].state_vector,
                                   list_of_bottom_states[state_idx].state_vector,
                                   remove_indices,
                                   qubit_circuit.num_qubits)
                weighted_sum_env += (-1 * overlap_list[state_idx].conj() * env)

            list_of_envs_wo_U[gate_idx] = weighted_sum_env
        # --------------------------------------------------------------------

        for state_idx in range(num_states):
            list_of_top_states[state_idx].apply_gate(remove_gate_conj, remove_indices)

    for state_idx in range(num_states):
        assert np.allclose(list_of_bottom_states[state_idx].state_vector,
                           list_of_initial_states[state_idx])

    # The gradient is given by the derivative of the cost with
    # respect to the parameters dC / dp = dC/d(U) * d(U)/dp
    grads = []
    for gate_idx in range(qubit_circuit.num_gates):
        if not qubit_circuit.trainable[gate_idx]:
            continue

        env = list_of_envs_wo_U[gate_idx].reshape([4, 4])
        # Currently, we have f = sum_ij env_ij, U_ij
        U = qubit_circuit.pairs_of_indices_and_Us[gate_idx][1]
        # The convention for input is
        # cost = f(Ud) = Tr (Ud @ dUd_mat)
        U_grad = U.get_gradient(env.T.conj())
        grads.append(U_grad)

    assert len(grads) == qubit_circuit.num_trainable_gates, \
        f"Expected {qubit_circuit.num_trainable_gates} gradients, got {len(grads)}"

    return cost, grads

def polar_opt(qubit_circuit,
              list_of_target_states,
              list_of_initial_states,
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
        list_of_bottom_states = [StateVector(initial_state) for initial_state in list_of_initial_states]
        for gate_idx in range(qubit_circuit.num_gates):
            gate = qubit_circuit.pairs_of_indices_and_Us[gate_idx][1]
            indices = qubit_circuit.pairs_of_indices_and_Us[gate_idx][0]
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
    for gate_idx in range(qubit_circuit.num_gates-1, -1, -1):
        remove_gate = qubit_circuit.pairs_of_indices_and_Us[gate_idx][1]
        remove_indices = qubit_circuit.pairs_of_indices_and_Us[gate_idx][0]
        remove_gate_conj = remove_gate.reshape([4, 4]).T.conj()
        remove_gate_conj = remove_gate_conj.reshape([2, 2, 2, 2])

        for state_idx in range(num_states):
            list_of_bottom_states[state_idx].apply_gate(remove_gate_conj, remove_indices)

        # Now the bottom states are the states without remove_gate.
        # We can now variational find the optimal gate to update.

        if qubit_circuit.trainable[gate_idx]:
            # We only update the trainable gates
            new_gate = var_gate_exact_list_of_states(list_of_top_states, remove_indices,
                                                     list_of_bottom_states,)
            qubit_circuit.pairs_of_indices_and_Us[gate_idx] = (remove_indices, new_gate)
        else:
            new_gate = qubit_circuit.pairs_of_indices_and_Us[gate_idx][1]

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
    for gate_idx in range(qubit_circuit.num_gates):
        remove_gate = qubit_circuit.pairs_of_indices_and_Us[gate_idx][1]
        remove_indices = qubit_circuit.pairs_of_indices_and_Us[gate_idx][0]

        for state_idx in range(num_states):
            list_of_top_states[state_idx].apply_gate(remove_gate, remove_indices)

        # This remove the gate from top_state
        # Because <\phi | U_{ij} | \psi> = inner( U_{ij}^\dagger |\phi>, |\psi> )
        # applying U would remove the U_{ij}^\dagger, and
        # partial_Tr[ |\phi>, |\psi> ] = Env

        # Now the top states are the states without remove_gate.
        # We can now variational find the optimal gate to update.

        if qubit_circuit.trainable[gate_idx]:
            new_gate = var_gate_exact_list_of_states(list_of_top_states, remove_indices,
                                                     list_of_bottom_states,)
            qubit_circuit.pairs_of_indices_and_Us[gate_idx] = (remove_indices, new_gate)
        else:
            new_gate = qubit_circuit.pairs_of_indices_and_Us[gate_idx][1]

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

        top_theta = np.reshape(top_state.state_vector,
                               [2 ** idx0, 2, 2 ** (idx1-(idx0+1)), 2, 2 ** (L-(idx1+1))])
        bottom_theta = np.reshape(bottom_state.state_vector,
                                  [2 ** idx0, 2, 2 ** (idx1-(idx0+1)), 2, 2 ** (L-(idx1+1))])
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


