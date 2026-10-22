"""Closed system (Sections 3-4 of the report): entanglement, purity, QFI and Cramér-Rao bound
for the four initial states, with the commuting generator V = S_z.

Usage (from the repository root):  python scripts/closed_system_figures.py [output_dir]
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from xxz import dynamics, measures, model  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "figures"
os.makedirs(OUT, exist_ok=True)

J, DELTA, N_SHOTS = 1.0, 0.5, 1000
tlist = np.linspace(0, 2 * np.pi, 500)
V = model.total_spin("z")


def save(fig, key, kind):
    fig.savefig(os.path.join(OUT, f"{key}_{kind}.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, f"{key}_{kind}.pdf"), bbox_inches="tight")
    plt.close(fig)


def entanglement_and_purity(psi0, key, label):
    rhos = dynamics.evolve(psi0, tlist, J, DELTA)           # closed system: gamma = 0
    C = np.array([measures.concurrence(r) for r in rhos])
    Neg = np.array([measures.negativity(r) for r in rhos])
    P = np.array([measures.purities(r) for r in rhos])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tlist, C, label="Concurrence C(t)")
    ax.plot(tlist, Neg, "--", label="Negativity N(t)")
    ax.set(xlabel=r"Time $t$ ($\hbar=1$)", ylabel="Entanglement measure", ylim=(-0.05, 1.05),
           title=f"{label}: Entanglement Dynamics")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save(fig, key, "entanglement")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tlist, P[:, 0], label=r"Purity $P_A$")
    ax.plot(tlist, P[:, 1], "--", label=r"Purity $P_B$")
    ax.set(xlabel=r"Time $t$ ($\hbar=1$)", ylabel="Purity", ylim=(0, 1.05), title=f"{label}: Reduced-State Purity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save(fig, key, "purity")


def metrology(variance_V, key, label):
    FQ = 4 * tlist**2 * variance_V                          # F_Q = 4 t^2 Var(V) because [H, V] = 0
    valid = FQ > 0
    bound = np.full_like(tlist, np.nan)
    bound[valid] = 1 / np.sqrt(N_SHOTS * FQ[valid])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tlist, FQ, linewidth=2, label=r"$F_Q(t)$")
    ax.set(xlabel=r"Time $t$", ylabel=r"Quantum Fisher Information $F_Q$", title=f"{label}: Quantum Fisher Information")
    ax.grid(True, alpha=0.3)
    ax.legend()
    if variance_V == 0:
        ax.text(0.5, 0.75, r"$F_Q(t)=0$ for all $t$", transform=ax.transAxes, ha="center")
    fig.tight_layout()
    save(fig, key, "QFI")

    fig, ax = plt.subplots(figsize=(8, 5))
    if np.any(valid):
        ax.plot(tlist[valid], bound[valid], linewidth=2, label=r"$\delta\lambda_{\min}$")
        ax.legend()
    else:
        ax.text(0.5, 0.6, "No finite bound: F_Q = 0\nThis state cannot estimate lambda\nwith the chosen generator V.",
                transform=ax.transAxes, ha="center", va="center")
    ax.set(xlabel=r"Time $t$", ylabel=r"Uncertainty $\delta\lambda_{\min}$", title=f"{label}: Cramér–Rao Bound")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save(fig, key, "CRB")


if __name__ == "__main__":
    for key, _, latex, psi0 in model.initial_states():
        label = f"Initial state {latex}"
        entanglement_and_purity(psi0, key, label)
        metrology(float(qt.variance(V, psi0)), key, label)   # variance computed, not hard-coded
    print("written:", sorted(os.listdir(OUT)))
