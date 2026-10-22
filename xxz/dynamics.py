"""Time evolution with QuTiP (closed or with local dephasing)."""
import numpy as np
import qutip as qt

from .model import collapse_operators, hamiltonian, total_spin

# Tight tolerances: the QFI is obtained by finite differences in lambda (step 1e-4),
# so the solver error must be far below 1e-4 * (target accuracy).
SOLVER_OPTIONS = {"atol": 1e-13, "rtol": 1e-11, "nsteps": 1_000_000, "store_states": True}


def evolve(psi0, tlist, J=1.0, delta=0.5, gamma=0.0, lam=0.0, generator="x"):
    """Density matrices rho(t) for H(lambda) = H_XXZ + lambda * S_generator and dephasing rate gamma.

    Returns a list of Qobj density matrices, one per time in `tlist`.
    """
    H = hamiltonian(J, delta)
    if lam != 0.0:
        H = H + lam * total_spin(generator)
    rho0 = psi0 if psi0.isoper else psi0.proj()
    tlist = np.asarray(tlist, float)
    # QuTiP's solvers take rho0 to be the state at tlist[0]; here rho0 is always the state at t = 0.
    start_at_zero = tlist[0] == 0.0
    grid = tlist if start_at_zero else np.concatenate([[0.0], tlist])
    states = qt.mesolve(H, rho0, grid, collapse_operators(gamma), options=SOLVER_OPTIONS).states
    return states if start_at_zero else states[1:]
