"""Entanglement and mixedness measures for two-qubit states."""
import numpy as np
import qutip as qt


def concurrence(rho):
    """Wootters concurrence of a two-qubit state (QuTiP implementation)."""
    return float(np.real(qt.concurrence(rho if rho.isoper else rho.proj())))


def negativity(rho):
    """Sum of the absolute values of the negative eigenvalues of the partial transpose on qubit B."""
    rho = rho if rho.isoper else rho.proj()
    eigs = np.linalg.eigvalsh(qt.partial_transpose(rho, [0, 1]).full())
    return float(np.sum(np.abs(eigs[eigs < 0])))


def purities(rho):
    """(P_A, P_B): purity Tr(rho_A^2), Tr(rho_B^2) of the reduced states."""
    rho = rho if rho.isoper else rho.proj()
    pa = float(np.real((rho.ptrace(0) * rho.ptrace(0)).tr()))
    pb = float(np.real((rho.ptrace(1) * rho.ptrace(1)).tr()))
    return pa, pb


def concurrence_dephasing_closed(state_key, t, gamma, J=1.0):
    """Closed-form concurrence under local dephasing (gamma < 4J); see the report, Sec. 5."""
    omega_r = np.sqrt(16 * J**2 - gamma**2)
    c01 = 4 * J / omega_r * np.exp(-gamma * t) * np.abs(np.sin(omega_r * t))
    return {"state1": 0 * c01, "state2": c01, "state3": 0.5 * c01, "state4": np.exp(-2 * gamma * t) + 0 * c01}[state_key]
