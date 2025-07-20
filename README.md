# Classical tools for quantum state preparation and simulation

We provide a set of classical tools for quantum state preparation and simulation.
The emphasis and assumption are on "classical", in the sense that at least one of
the classical representation of the simulation should be efficiently computable.

The repository consists part of the [code](https://github.com/ShHsLin/qutepy-archive) from the TUM group on finding state prep circuits.
It utilizes the tensor networks and exact state vector simulation with fidelity optimization
to find the state preparation circuits for the target states.

The tools are planned to include more features including but not only:
- finding state prep circuits approximating the target state
- finding state prep circuits minimizing the energy
- finding a basis transformation circuit that maps a given stet of initial
  states to a given set of target states.

To this end, the goal is to use different possible backend for the simulaiton.
Depending on the setup and application, different simulation methods should
be used. The factor affecting the choice of the simulation method include:
Geometry, dimenionality, system size, application, etc.
- state vector simulation
- tensor network simulation
- sparse pauli operator simulation
- sparse pauli vector simulation
- sparse fermion vector simulation

