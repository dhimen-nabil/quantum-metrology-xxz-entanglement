"""Quantum Fisher information (QFI) and Cramér-Rao bound."""
import numpy as np
import qutip as qt

from .dynamics import evolve
from .model import detuning, hamiltonian, total_spin


def qfi_mixed(rho, drho, tol=1e-9):
    """F_Q = 2 sum_{p_i+p_j>0} |<i|d rho|j>|^2 / (p_i + p_j) (eigen-decomposition of rho)."""
    p, U = np.linalg.eigh(rho.full())
    d = U.conj().T @ drho.full() @ U
    s = p[:, None] + p[None, :]
    mask = s > tol
    return float(np.sum(2 * np.abs(d[mask]) ** 2 / s[mask]))


def qfi_series(psi0, tlist, J=1.0, delta=0.5, gamma=0.0, generator="x", eps=1e-4):
    """QFI of lambda at lambda=0 for every t in `tlist`, central finite difference of step eps."""
    r0 = evolve(psi0, tlist, J, delta, gamma, 0.0, generator)
    rp = evolve(psi0, tlist, J, delta, gamma, +eps, generator)
    rm = evolve(psi0, tlist, J, delta, gamma, -eps, generator)
    return np.array([qfi_mixed(a, (b - c) / (2 * eps)) for a, b, c in zip(r0, rp, rm)])


def qfi_commuting(psi0, t, generator="z"):
    """F_Q = 4 t^2 Var(V) when [H, V] = 0 (V = S_z)."""
    return 4 * t**2 * float(qt.variance(total_spin(generator), psi0))


def qfi_duhamel(psi0, t, J=1.0, delta=0.5, generator="x"):
    """Closed system, any generator: F_Q = 4 Var(G(t)), G(t) = int_0^t e^{iHs} V e^{-iHs} ds (pure state)."""
    E, U = np.linalg.eigh(hamiltonian(J, delta).full())
    Vm = U.conj().T @ total_spin(generator).full() @ U
    w = E[:, None] - E[None, :]
    safe = np.where(np.abs(w) < 1e-12, 1.0, w)
    G = np.where(np.abs(w) < 1e-12, t * Vm, Vm * (np.exp(1j * w * t) - 1) / (1j * safe))
    c = U.conj().T @ psi0.full().ravel()
    m1, m2 = c.conj() @ G @ c, c.conj() @ G @ G @ c
    return float(np.real(4 * (m2 - abs(m1) ** 2)))


def qfi_sx_closed(state_key, t, J=1.0, delta=0.5):
    """Closed forms for V = S_x: 8 sin^2(w t/2)/w^2 for |00>,|01>; twice that for |psi+>."""
    w = detuning(J, delta)
    base = 8 * np.sin(w * t / 2) ** 2 / w**2
    if state_key in ("state1", "state2"):
        return base
    if state_key == "state4":
        return 2 * base
    raise ValueError("closed form only for |00>, |01>, |psi+>")


def crb(F, nu=1000):
    """Quantum Cramér-Rao bound 1/sqrt(nu F); NaN where F is (numerically) zero."""
    F = np.asarray(F, float)
    out = np.full_like(F, np.nan)
    ok = F > 1e-9
    out[ok] = 1.0 / np.sqrt(nu * F[ok])
    return out
