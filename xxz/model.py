"""Two-qubit Heisenberg XXZ model: Hamiltonian, generators, initial states, noise.

Conventions
-----------
* hbar = 1 and Pauli matrices (not spin-1/2 operators), so the characteristic
  frequency of the exchange term is 4J.
* Computational basis ordering {|00>, |01>, |10>, |11>}.
* H = J (sx*sx + sy*sy + Delta * sz*sz).
"""
import numpy as np
import qutip as qt

J_DEFAULT = 1.0
DELTA_DEFAULT = 0.5

SX, SY, SZ, I2 = qt.sigmax(), qt.sigmay(), qt.sigmaz(), qt.qeye(2)


def hamiltonian(J=J_DEFAULT, delta=DELTA_DEFAULT):
    """H = J (sx⊗sx + sy⊗sy + Delta sz⊗sz)."""
    return J * (qt.tensor(SX, SX) + qt.tensor(SY, SY) + delta * qt.tensor(SZ, SZ))


def total_spin(axis):
    """Total spin component S_a = (sigma_a ⊗ I + I ⊗ sigma_a)/2 for axis in {'x','y','z'}."""
    s = {"x": SX, "y": SY, "z": SZ}[axis]
    return 0.5 * (qt.tensor(s, I2) + qt.tensor(I2, s))


def detuning(J=J_DEFAULT, delta=DELTA_DEFAULT):
    """omega = E_{psi+} - E_{00} = 2J(1 - Delta): the frequency that S_x probes."""
    return 2.0 * J * (1.0 - delta)


def collapse_operators(gamma):
    """Local pure dephasing in the z basis, rate gamma per qubit (Lindblad operators sqrt(gamma/2) sigma_z^(k))."""
    if gamma == 0:
        return []
    return [np.sqrt(gamma / 2) * qt.tensor(SZ, I2), np.sqrt(gamma / 2) * qt.tensor(I2, SZ)]


def initial_states():
    """The four initial states, as an ordered list of (key, plain label, latex label, ket)."""
    k0, k1 = qt.basis(2, 0), qt.basis(2, 1)
    k00, k01, k10 = qt.tensor(k0, k0), qt.tensor(k0, k1), qt.tensor(k1, k0)
    return [
        ("state1", "|00>", r"$|00\rangle$", k00),
        ("state2", "|01>", r"$|01\rangle$", k01),
        ("state3", "(|00>+|01>)/sqrt2", r"$(|00\rangle+|01\rangle)/\sqrt{2}$", (k00 + k01).unit()),
        ("state4", "|psi+>", r"$|\psi_+\rangle$", (k01 + k10).unit()),
    ]
