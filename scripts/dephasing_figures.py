"""Non-commuting generator V = S_x and local dephasing (Section 5 of the report).

Produces the figures figures/nc_*.{png,pdf} and the LaTeX tables tables/nc_*.tex.
Usage (from the repository root):  python scripts/dephasing_figures.py [root_dir]
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from xxz import measures, metrology, model  # noqa: E402

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
FIG, TAB = os.path.join(ROOT, "figures"), os.path.join(ROOT, "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

J, DELTA, NSHOTS = 1.0, 0.5, 1000
omega = model.detuning(J, DELTA)
STATES = [(key, latex, psi) for key, _, latex, psi in model.initial_states()]
PSI = {key: psi for key, _, psi in STATES}

tg = np.linspace(0.0, 4 * np.pi, 800)
GAMMAS = [0.0, 0.05, 0.1, 0.2]                     # curves in the figures
GAMMAS_TAB = [0.0, 0.05, 0.1, 0.2, 0.5]            # columns in the tables
RAMP = {0.0: "#0d366b", 0.05: "#1c5cab", 0.1: "#3987e5", 0.2: "#86b6ef"}   # blue ordinal ramp
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})


def save(fig, key):
    fig.savefig(os.path.join(FIG, key + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(FIG, key + ".pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------- QFI on the time grid (QuTiP mesolve + finite difference in lambda) ----------
F = {(key, g): metrology.qfi_series(PSI[key], tg, J, DELTA, g, "x") for key, _, _ in STATES for g in GAMMAS_TAB}

# ---------- per-state QFI and Cramér-Rao figures ----------
for idx, (key, label, _) in enumerate(STATES, start=1):
    fig, ax = plt.subplots(figsize=(8, 5))
    for g in GAMMAS:
        ax.plot(tg, F[(key, g)], color=RAMP[g], lw=2, label=rf"$\gamma={g:g}$")
    ax.set(xlabel=r"Time $t$ ($J=1$, $\hbar=1$)", ylabel=r"Quantum Fisher information $F_Q(t)$",
           title=rf"Initial state {label}: QFI for $V=S_x$ with local dephasing", xlim=(0, 4 * np.pi), ylim=(0, None))
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    save(fig, f"nc_state{idx}_QFI")

    fig, ax = plt.subplots(figsize=(8, 5))
    for g in GAMMAS:
        ax.plot(tg, metrology.crb(F[(key, g)], NSHOTS), color=RAMP[g], lw=2, label=rf"$\gamma={g:g}$")
    ax.set(ylim=(0, 0.05), xlim=(0, 4 * np.pi), xlabel=r"Time $t$ ($J=1$, $\hbar=1$)",
           ylabel=r"Uncertainty $\delta\lambda_{\min}$ ($N=1000$)",
           title=rf"Initial state {label}: Cramér–Rao bound with local dephasing")
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    save(fig, f"nc_state{idx}_CRB")

# ---------- concurrence under dephasing (closed forms; validated against QuTiP in validate.py) ----------
fig, axs = plt.subplots(2, 2, figsize=(9, 6.2), sharex=True, sharey=True)
for ax, (key, label, _) in zip(axs.ravel(), STATES):
    for g in GAMMAS:
        ax.plot(tg, measures.concurrence_dephasing_closed(key, tg, g, J), color=RAMP[g], lw=2, label=rf"$\gamma={g:g}$")
    ax.set_title(rf"Initial state {label}", fontsize=11)
    ax.set(xlim=(0, 4 * np.pi), ylim=(-0.03, 1.05))
for ax in axs[1]:
    ax.set_xlabel(r"Time $t$")
for ax in axs[:, 0]:
    ax.set_ylabel(r"Concurrence $C(t)$")
axs[0, 0].text(0.5, 0.18, r"$C(t)=0$ for all $t$ and all $\gamma$", transform=axs[0, 0].transAxes, ha="center", fontsize=10)
axs[0, 0].legend(loc="center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 0.6))
fig.tight_layout()
save(fig, "nc_concurrence")

# ---------- comparison: product vs entangled, advantage vs gamma ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
ax1.plot(tg, F[("state1", 0.0)], color=BLUE, lw=2, ls="-", label=r"$|00\rangle$, $\gamma=0$")
ax1.plot(tg, F[("state4", 0.0)], color=ORANGE, lw=2, ls="-", label=r"$|\psi_+\rangle$, $\gamma=0$")
ax1.plot(tg, F[("state1", 0.1)], color=BLUE, lw=2, ls="--", label=r"$|00\rangle$, $\gamma=0.1$")
ax1.plot(tg, F[("state4", 0.1)], color=ORANGE, lw=2, ls="--", label=r"$|\psi_+\rangle$, $\gamma=0.1$")
ax1.set(xlabel=r"Time $t$", ylabel=r"$F_Q(t)$", title="Product vs entangled state", xlim=(0, 4 * np.pi), ylim=(0, 22))
ax1.legend(loc="upper center", ncol=2, fontsize=9)

gs = np.linspace(0.0, 0.5, 26)
tt = tg[1:]
Rpeak, Rrate = [], []
for g in gs:
    f0 = metrology.qfi_series(PSI["state1"], tg, J, DELTA, g, "x")[1:]
    f4 = metrology.qfi_series(PSI["state4"], tg, J, DELTA, g, "x")[1:]
    Rpeak.append(f4.max() / f0.max())
    Rrate.append((f4 / tt).max() / (f0 / tt).max())
ax2.plot(gs, Rpeak, color=BLUE, lw=2, label=r"$\max_t F_Q$")
ax2.plot(gs, Rrate, color=ORANGE, lw=2, ls="--", label=r"$\max_t F_Q/t$")
ax2.axhline(1.0, color="0.45", lw=1, ls=":")
ax2.text(0.497, 1.02, "no advantage", color="0.35", fontsize=9, ha="right")
ax2.set(xlabel=r"Dephasing rate $\gamma$", ylabel=r"Advantage ratio  $\psi_+$ / $|00\rangle$",
        title="Entanglement advantage vs. dephasing", ylim=(0.9, 2.1))
ax2.legend(loc="upper right", fontsize=9)
fig.tight_layout()
save(fig, "nc_comparison")

# ---------- LaTeX tables ----------
names_tab = [("state1", r"$\ket{00}$"), ("state2", r"$\ket{01}$"),
             ("state3", r"$(\ket{00}+\ket{01})/\sqrt2$"), ("state4", r"$\ket{\psi_+}$")]
head = " & ".join(rf"$\gamma={g:g}$" for g in GAMMAS_TAB)


def write_table(name, header, rows, colspec):
    with open(os.path.join(TAB, name), "w") as fh:
        fh.write(r"\begin{tabular}{@{}" + colspec + r"@{}}" + "\n\\toprule\n" + header + r"\\" + "\n\\midrule\n")
        for row in rows:
            fh.write(row + r"\\" + "\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")


write_table("nc_peak.tex", r"Initial state & " + head,
            [lab + " & " + " & ".join(f"{F[(n, g)].max():.2f}" for g in GAMMAS_TAB) for n, lab in names_tab],
            "l" + "c" * len(GAMMAS_TAB))
write_table("nc_crb.tex", r"Initial state & " + head,
            [lab + " & " + " & ".join(f"{1 / np.sqrt(NSHOTS * F[(n, g)].max()):.4f}" for g in GAMMAS_TAB) for n, lab in names_tab],
            "l" + "c" * len(GAMMAS_TAB))

adv_rows = []
for g in GAMMAS_TAB:
    i = int(round(g / 0.02))
    frac = float(np.mean(F[("state4", g)][1:] > F[("state1", g)][1:]))
    adv_rows.append(f"{g:g} & {Rpeak[i]:.2f} & {Rrate[i]:.2f} & {100 * frac:.0f}\\%")
write_table("nc_adv.tex", r"$\gamma$ & $\max_t F_Q$ ratio & $\max_t F_Q/t$ ratio & time fraction with $F_{\psi_+}>F_{00}$",
            adv_rows, "lccc")

plateau_rows = []
for g in [0.05, 0.1, 0.2, 0.5]:
    vals = [metrology.qfi_series(psi, np.array([0.0, 300.0]), J, DELTA, g, "x")[-1] for _, _, psi in STATES]
    plateau_rows.append(f"{g:g} & " + " & ".join(f"{v:.3f}" for v in vals) + f" & {2 / (omega ** 2 + g ** 2):.3f}")
write_table("nc_plateau.tex", r"$\gamma$ & $\ket{00}$ & $\ket{01}$ & $(\ket{00}+\ket{01})/\sqrt2$ & $\ket{\psi_+}$ & $2/(\omega^2+\gamma^2)$",
            plateau_rows, "lccccc")

print("peak QFI (rows: states, cols: gamma =", GAMMAS_TAB, ")")
for n, _ in names_tab:
    print(" ", n, [round(float(F[(n, g)].max()), 3) for g in GAMMAS_TAB])
print("advantage ratios (max F, max F/t):", {g: (round(Rpeak[int(round(g / 0.02))], 3), round(Rrate[int(round(g / 0.02))], 3)) for g in GAMMAS_TAB})
