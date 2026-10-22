"""Numerical validation of every analytic formula and of the QuTiP implementation.

Run from the repository root:  python scripts/validate.py
Exit status is non-zero if any check exceeds its tolerance.
"""
import os
import sys

import numpy as np
import qutip as qt
from scipy.linalg import expm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from xxz import dynamics, measures, metrology, model  # noqa: E402

J, DELTA = 1.0, 0.5
STATES = model.initial_states()
failures = []


def check(name, value, tol):
    ok = value <= tol
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {value:.2e} (tolerance {tol:.0e})")
    if not ok:
        failures.append(name)


# ---------------------------------------------------------------------------
# Independent reference: plain NumPy/SciPy Liouvillian (no QuTiP in the solver)
# ---------------------------------------------------------------------------
def reference_rho(psi0, t, gamma, lam, generator="x"):
    k = np.kron
    sx, sy, sz = (m.full() for m in (model.SX, model.SY, model.SZ))
    I2, I4, I16 = np.eye(2), np.eye(4), np.eye(16)
    H = J * (k(sx, sx) + k(sy, sy) + DELTA * k(sz, sz))
    s = {"x": sx, "z": sz}[generator]
    H = H + lam * 0.5 * (k(s, I2) + k(I2, s))
    L = -1j * (k(I4, H) - k(H.T, I4))                       # column-stacking vec
    if gamma > 0:
        for Z in (k(sz, I2), k(I2, sz)):
            L = L + (gamma / 2) * (k(Z.T, Z) - I16)
    psi = psi0.full().ravel()
    v = np.outer(psi, psi.conj()).reshape(-1, order="F")
    return (expm(L * t) @ v).reshape((4, 4), order="F")


print("== 1. QuTiP mesolve vs independent NumPy/SciPy Liouvillian ==")
worst = 0.0
tl = np.array([0.0, 0.8, 3.3, 7.1])
for _, _, _, psi in STATES:
    for g in (0.0, 0.1, 0.5):
        r = dynamics.evolve(psi, tl, J, DELTA, g, 0.03, "x")
        for i, t in enumerate(tl):
            worst = max(worst, np.abs(r[i].full() - reference_rho(psi, t, g, 0.03)).max())
check("max |rho_QuTiP - rho_reference|", worst, 1e-8)

print("\n== 2. Closed system: analytic C, N, P vs numerics (Sec. 3) ==")
t2 = np.linspace(0, 2 * np.pi, 400)
C_formula = {"state1": 0 * t2, "state2": np.abs(np.sin(4 * J * t2)), "state3": 0.5 * np.abs(np.sin(4 * J * t2)), "state4": 1 + 0 * t2}
P_formula = {"state1": 1 + 0 * t2, "state2": 1 - 0.5 * np.sin(4 * J * t2) ** 2, "state3": 1 - np.sin(4 * J * t2) ** 2 / 8, "state4": 0.5 + 0 * t2}
wC = wN = wP = wPC = 0.0
for key, _, _, psi in STATES:
    rhos = dynamics.evolve(psi, t2, J, DELTA)
    C = np.array([measures.concurrence(r) for r in rhos])
    N = np.array([measures.negativity(r) for r in rhos])
    PA = np.array([measures.purities(r)[0] for r in rhos])
    wC = max(wC, np.abs(C - C_formula[key]).max())
    wN = max(wN, np.abs(N - C / 2).max())
    wP = max(wP, np.abs(PA - P_formula[key]).max())
    wPC = max(wPC, np.abs(PA - (1 - C**2 / 2)).max())
check("max |C_numeric - C_analytic|", wC, 1e-6)
check("max |N - C/2|", wN, 1e-6)
check("max |P_numeric - P_analytic|", wP, 1e-6)
check("max |P - (1 - C^2/2)|", wPC, 1e-6)
d_worst = 0.0
for d in (0.0, 1.0, 2.0):                                    # Delta must not change C(t)
    r = dynamics.evolve(STATES[2][3], t2, J, d)
    d_worst = max(d_worst, np.abs(np.array([measures.concurrence(x) for x in r]) - C_formula["state3"]).max())
check("Delta-independence of C(t) (Delta = 0, 1, 2)", d_worst, 1e-6)

print("\n== 3. Commuting generator V = S_z: F_Q = 4 t^2 Var(V) ==")
t3 = np.linspace(0.1, 6.0, 40)
w3 = 0.0
for key, _, _, psi in STATES:
    f_num = metrology.qfi_series(psi, t3, J, DELTA, 0.0, "z")
    f_ana = np.array([metrology.qfi_commuting(psi, t) for t in t3])
    w3 = max(w3, np.abs(f_num - f_ana).max())
check("max |F_Q(finite difference) - 4 t^2 Var(S_z)|", w3, 1e-5)

print("\n== 4. Non-commuting generator V = S_x, closed system (Sec. 5.1) ==")
t4 = np.linspace(0.0, 4 * np.pi, 400)
fd = {k: metrology.qfi_series(p, t4, J, DELTA, 0.0, "x") for k, _, _, p in STATES}
du = {k: np.array([metrology.qfi_duhamel(p, t) for t in t4]) for k, _, _, p in STATES}
check("max |F_Q(finite difference) - F_Q(Duhamel)|, all states", max(np.abs(fd[k] - du[k]).max() for k in fd), 1e-5)
check("max |F_Q(Duhamel) - closed form|, |00>, |01>, |psi+>",
      max(np.abs(du[k] - metrology.qfi_sx_closed(k, t4, J, DELTA)).max() for k in ("state1", "state2", "state4")), 1e-10)
check("max |F_Q(|psi+>) - 2 F_Q(|00>)|", np.abs(du["state4"] - 2 * du["state1"]).max(), 1e-10)
check("max |F_Q(|01>) - F_Q(|00>)|", np.abs(du["state2"] - du["state1"]).max(), 1e-10)
nrm = np.linalg.norm((model.hamiltonian(J, DELTA) * model.total_spin("x") - model.total_spin("x") * model.hamiltonian(J, DELTA)).full(), 2)
print(f"        ||[H, S_x]|| (spectral norm) = {nrm:.4f}, nonzero for Delta != 1")

print("\n== 5. Mixed-state QFI under dephasing: three independent estimators ==")


def qfi_pinv(r0, dr, rcond=1e-9):
    R = r0.full()
    M = np.kron(R, np.eye(4)) + np.kron(np.eye(4), R.T)       # row-major vec(R X + X R)
    v = dr.full().reshape(-1)
    return float(np.real(2 * v.conj() @ np.linalg.pinv(M, rcond=rcond, hermitian=True) @ v))


def qfi_fidelity(psi, t, g, delta=0.02):
    a = dynamics.evolve(psi, [0.0, t], J, DELTA, g, -delta / 2, "x")[-1]
    b = dynamics.evolve(psi, [0.0, t], J, DELTA, g, +delta / 2, "x")[-1]
    return 8 * (1 - qt.fidelity(a, b)) / delta**2             # F_Q = 8(1 - sqrt-fidelity)/delta^2 as delta -> 0


w_pinv = w_fid = 0.0
eps = 1e-4
for _, _, _, psi in STATES:
    for g in (0.1, 0.5):
        for t in (0.7, 2.0, 4.5, 7.0):
            tl = [0.0, t]
            r0 = dynamics.evolve(psi, tl, J, DELTA, g, 0.0, "x")[-1]
            rp = dynamics.evolve(psi, tl, J, DELTA, g, +eps, "x")[-1]
            rm = dynamics.evolve(psi, tl, J, DELTA, g, -eps, "x")[-1]
            dr = (rp - rm) / (2 * eps)
            a = metrology.qfi_mixed(r0, dr)
            w_pinv = max(w_pinv, abs(a - qfi_pinv(r0, dr)) / max(1e-6, abs(a)))
            w_fid = max(w_fid, abs(a - qfi_fidelity(psi, t, g)) / max(1e-3, abs(a)))
check("relative difference, eigen-decomposition vs superoperator pseudo-inverse", w_pinv, 1e-8)
check("relative difference, eigen-decomposition vs fidelity-based estimate", w_fid, 5e-3)

print("\n== 6. Closed-form concurrence under dephasing vs Wootters (QuTiP) ==")
w6 = 0.0
t6 = np.linspace(0.2, 12, 15)
for key, _, _, psi in STATES:
    for g in (0.05, 0.1, 0.2, 0.5):
        rhos = dynamics.evolve(psi, t6, J, DELTA, g, 0.0)
        C = np.array([measures.concurrence(r) for r in rhos])
        w6 = max(w6, np.abs(C - measures.concurrence_dephasing_closed(key, t6, g, J)).max())
check("max |C_Wootters - C_closed form|", w6, 1e-6)
rhos = dynamics.evolve(STATES[3][3], np.linspace(0, 20, 200), J, DELTA, 0.5, 0.0)
cmin = min(measures.concurrence(r) for r in rhos)
print(f"        min over t in [0,20] of C(|psi+>) at gamma = 0.5: {cmin:.3e} (> 0: no sudden death within the window)")

print("\n== 7. Long-time plateau: F_Q(t=300) vs the empirical 2/(omega^2 + gamma^2) ==")
w7 = 0.0
for d in (0.0, 0.25, 0.5, 0.75):
    om = model.detuning(J, d)
    for g in (0.1, 0.5):
        for k in ("state1", "state2", "state4"):
            psi = dict((s[0], s[3]) for s in STATES)[k]
            val = metrology.qfi_series(psi, np.array([0.0, 300.0]), J, d, g, "x")[-1]
            w7 = max(w7, abs(val - 2 / (om**2 + g**2)) / (2 / (om**2 + g**2)))
check("max relative deviation from 2/(omega^2+gamma^2), |00>, |01>, |psi+>", w7, 1e-3)

print()
if failures:
    print("FAILED:", failures)
    sys.exit(1)
print("All checks passed.")
