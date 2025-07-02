# Classical tools for quantum state preparation and simulation

We provide a set of classical tools for quantum state preparation and simulation.
The emphasis and assumption are on "classical",
in the sense that at least one of the classical representation of the simulation
should be efficiently computable.

The repository consists part of the [code](https://github.com/ShHsLin/qutepy-archive) from the TUM group on finding state prep circuits.
It utilizes the tensor networks and exact state vector simulation with fidelity optimization
to find the state preparation circuits for the target states.

The tools are now expanded to (plan to) include more features including but not only:
- state prep circuit found by energy minimization and more general geometry
- adding sparse pauli simulation as another simulation tools
- finding circuit to perform given basis transformation

