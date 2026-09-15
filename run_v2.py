# -*- coding: utf-8 -*-
"""Figures du PoC v2 (loi de puissance certifiee). Sorties : figv2_*.png."""

import numpy as np
np.seterr(all="ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bean_slab_certified import (
    BeanSlabCertified, bean_Q_per_cycle, bean_loop_meanH_down,
    aligned_schedule, _run_schedule, certified_cycle_loss,
)
from powerlaw_slab_certified import (
    MU0, PowerLawSlabCertified, run_powerlaw_cycles, loss_last_cycle,
)

C_ANA = "#d62728"
CMAP = plt.cm.viridis


def main():
    beta = 0.7
    ns = [5, 10, 20, 40, 80]
    runs, Qs, errs = {}, {}, {}
    for n_exp in ns:
        m = PowerLawSlabCertified(n_exp=n_exp, N=800)
        h = run_powerlaw_cycles(m, beta, steps_per_cycle=200, n_cycles=3)
        Q, err_sp, drift = loss_last_cycle(m, h)
        runs[n_exp] = (m, h)
        Qs[n_exp], errs[n_exp] = Q, err_sp
        print(f"n = {n_exp:3d} : Q = {Q:8.1f} +/- {err_sp:7.1f} (spatial), "
              f"derive periodicite = {drift:.2e}")
    m0 = runs[ns[0]][0]
    Hp = m0.Hp
    Hm = beta * Hp
    QBean = float(bean_Q_per_cycle(beta, Hp))

    # ------------------------------------------- figA : famille de cycles
    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    S = 200
    for i, n_exp in enumerate(ns):
        m, h = runs[n_exp]
        Ha = h["Ha"][-S - 1:]
        mM = (np.array(h["meanH"][-S - 1:]) - Ha) / Hp
        ax.plot(Ha / Hp, mM, "-", color=CMAP(i / (len(ns) - 1)), lw=1.5,
                label=f"n = {n_exp}")
    ha_d = np.linspace(Hm, -Hm, 300)
    md = (bean_loop_meanH_down(ha_d, Hm, Hp) - ha_d) / Hp
    ax.plot(ha_d / Hp, md, "--", color=C_ANA, lw=2.0, label="Bean (analytique)")
    ax.plot(-ha_d / Hp, -md, "--", color=C_ANA, lw=2.0)
    ax.set_xlabel(r"$H_a/H_p$")
    ax.set_ylabel(r"$\langle M\rangle/H_p$")
    ax.set_title(rf"Loi de puissance : cycles vs modele de Bean ($\beta$ = {beta}, 50 Hz)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("figv2_cycles.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------- figB : Q(n) et brackets
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    qn = MU0 * Hp**2
    ax.axhline(QBean / qn, color=C_ANA, ls="--", lw=1.6,
               label=r"Bean ($n \to \infty$), analytique")
    ax.errorbar(ns, [Qs[n] / qn for n in ns], yerr=[errs[n] / qn for n in ns],
                fmt="o-", color="#1f77b4", capsize=5, ms=5, lw=1.4,
                label="loi de puissance, bracket spatial certifie")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("exposant n de la loi E-J")
    ax.set_ylabel(r"$Q_v/(\mu_0 H_p^2)$ par cycle")
    ax.set_title("Pertes AC : convergence vers la limite de Bean")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig("figv2_bean_limit.png", dpi=150)
    plt.close(fig)

    # ---------------------------- figC : robustesse en n de l'effectivite
    effs = []
    for n_exp in ns:
        mc = PowerLawSlabCertified(n_exp=n_exp, N=200)
        tau = 1.0 / (50.0 * 200)
        H = np.zeros(mc.N + 1)
        for k in range(1, 11):
            hb = Hm * np.sin(2 * np.pi * 50.0 * tau * k)
            H, _ = mc.prox_step(H, hb, tau)
        g = H
        hb = Hm * np.sin(2 * np.pi * 50.0 * tau * 11)
        Hc, _ = mc.prox_step(g, hb, tau)
        _, errc, _ = mc.certified_gap(Hc, g, hb, tau)
        mf = PowerLawSlabCertified(n_exp=n_exp, N=1600)
        gf = np.interp(mf.x, mc.x, g)
        Hf, _ = mf.prox_step(gf, hb, tau, tol=1e-14)
        actual = mf.norm_L2(np.interp(mf.x, mc.x, Hc) - Hf)
        effs.append(errc / max(actual, 1e-30))
        print(f"effectivite n = {n_exp:3d} : {effs[-1]:.2f}")
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(ns, effs, "o-", color="#1f77b4", lw=1.5, ms=6)
    ax.axhline(1.0, color="#7f7f7f", ls=":", label="effectivite ideale = 1")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_ylim(0, max(effs) * 1.3)
    ax.set_xlabel("exposant n")
    ax.set_ylabel("effectivite = borne certifiee / erreur reelle")
    ax.set_title("ROBUSTESSE EN n du certificat primal-dual (pas prox)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("figv2_robustesse.png", dpi=150)
    plt.close(fig)

    # ------------------------------- figD : profils J/Jc, depassement de Jc
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for i, n_exp in enumerate([10, 20, 80]):
        m, h = runs[n_exp]
        kq = len(h["Ha"]) - 100                 # zero de Ha (vitesse maximale)
        J = -np.diff(h["profiles"][kq]) / m.h
        xm = 0.5 * (m.x[:-1] + m.x[1:]) * 1e3
        ax.plot(xm, J / m.Jc, "-", color=CMAP(i / 2), lw=1.4, label=f"n = {n_exp}")
    ax.axhline(1.0, color=C_ANA, ls="--", lw=1.2, label=r"$\pm J_c$ (Bean)")
    ax.axhline(-1.0, color=C_ANA, ls="--", lw=1.2)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel(r"$J/J_c$")
    ax.set_title(r"Profils de courant a $H_a = 0$ (vitesse max) : fluage au-dela de $J_c$")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("figv2_profils_J.png", dpi=150)
    plt.close(fig)

    print(f"Q_Bean = {QBean:.1f} J/m^3")
    print("figures ecrites : figv2_cycles.png figv2_bean_limit.png "
          "figv2_robustesse.png figv2_profils_J.png")


if __name__ == "__main__":
    main()
