# -*- coding: utf-8 -*-
"""Figures du PoC pertes AC certifiees (slab de Bean). Sorties : fig1..fig4 PNG."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bean_slab_certified import (
    MU0, BeanSlabCertified, bean_Q_per_cycle, bean_virgin_meanH,
    bean_loop_meanH_down, bean_profile_virgin, bean_profile_after_reversal,
    aligned_schedule, sine_schedule, _run_schedule, certified_cycle_loss,
)

C_NUM, C_ANA, C_B = "#1f77b4", "#d62728", "#7f7f7f"


def main():
    # ------------------------------------------------------------------ runs
    beta0, N0 = 0.7, 200
    m = BeanSlabCertified(N=N0)
    seq, i0 = aligned_schedule(m, beta0)
    h = _run_schedule(m, seq, mode="A")
    Hm, Hp = beta0 * m.Hp, m.Hp

    # ---------------------------------------------------- fig 1 : profils
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    picks = [i0 - 1, i0 + (len(seq) - i0) // 8, i0 + (len(seq) - i0) // 4,
             i0 + 3 * (len(seq) - i0) // 8, i0 + (len(seq) - i0) // 2 - 1]
    for k in picks:
        Ha_k = h["Ha"][k]
        ax.plot(m.x * 1e3, h["profiles"][k] / Hp, "-", color=C_NUM, lw=1.4)
        if k == i0 - 1:
            ana = bean_profile_virgin(m.x, Hm, m.a, m.Jc)
        else:
            ana = bean_profile_after_reversal(m.x, Hm, Ha_k, m.a, m.Jc)
        ax.plot(m.x * 1e3, ana / Hp, "--", color=C_ANA, lw=1.0)
        ax.annotate(f"$H_a/H_p$ = {Ha_k / Hp:+.2f}", (m.x[2] * 1e3,
                    h["profiles"][k][2] / Hp), fontsize=8,
                    textcoords="offset points", xytext=(6, 3))
    ax.plot([], [], "-", color=C_NUM, label="solveur certifie (P1 + dual)")
    ax.plot([], [], "--", color=C_ANA, label="analytique de Bean")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel(r"$H/H_p$")
    ax.set_title(rf"Slab de Bean, profils sur la descente ($\beta$ = {beta0})")
    ax.legend(loc="lower center", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig1_profils.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------- fig 2 : cycle <M>(Ha)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    Ha, mH = h["Ha"], h["meanH"]
    Mn = (mH - Ha) / Hp
    ax.plot(Ha / Hp, Mn, "-", color=C_NUM, lw=1.6,
            label="solveur certifie")
    ha_d = np.linspace(Hm, -Hm, 200)
    md = (bean_loop_meanH_down(ha_d, Hm, Hp) - ha_d) / Hp
    ax.plot(ha_d / Hp, md, "--", color=C_ANA, lw=1.2, label="analytique (descente)")
    ax.plot(-ha_d / Hp, -md, ":", color=C_ANA, lw=1.2, label="analytique (montee)")
    ha_v = np.linspace(0, Hm, 100)
    ax.plot(ha_v / Hp, (bean_virgin_meanH(ha_v, Hp) - ha_v) / Hp, "-.",
            color=C_B, lw=1.0, label="premiere aimantation")
    ax.set_xlabel(r"$H_a/H_p$")
    ax.set_ylabel(r"$\langle M\rangle/H_p$")
    ax.set_title(rf"Cycle d'hysteresis ($\beta$ = {beta0}), aire = pertes/cycle")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig2_cycle.png", dpi=150)
    plt.close(fig)

    # --------------------------------------- fig 3 : LE bracket certifie (Q)
    betas = [0.4, 0.7, 1.0, 1.4]
    Qs, errs = [], []
    for b in betas:
        mb = BeanSlabCertified(N=200)
        sq, ib = aligned_schedule(mb, b)
        hb = _run_schedule(mb, sq, mode="A", check_oracle=False)
        Q, err, _, _ = certified_cycle_loss(mb, hb, ib)
        Qs.append(Q)
        errs.append(err)
        print(f"mode A beta={b}: Q = {Q:.1f} +/- {err:.1f} J/m^3")
    mB = BeanSlabCertified(N=400)
    sqB, iB = sine_schedule(mB, 0.7, nsteps_per_branch=120)
    hB = _run_schedule(mB, sqB, mode="B", check_oracle=False)
    QB, errB, _, _ = certified_cycle_loss(mB, hB, iB)
    print(f"mode B beta=0.7: Q = {QB:.1f} +/- {errB:.1f} J/m^3")

    qn = MU0 * Hp**2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.4),
                                   gridspec_kw={"width_ratios": [3, 2]})
    bb = np.linspace(0.05, 1.6, 300)
    ax1.plot(bb, bean_Q_per_cycle(bb, Hp) / qn, "-", color=C_ANA, lw=1.4,
             label="analytique de Bean")
    ax1.errorbar(betas, np.array(Qs) / qn, yerr=np.array(errs) / qn, fmt="o",
                 color=C_NUM, capsize=5, ms=5, lw=1.6,
                 label="bracket GARANTI, mode A")
    ax1.errorbar([0.72], [QB / qn], yerr=[errB / qn], fmt="s", mfc="none",
                 color=C_B, capsize=4, ms=6, lw=1.2,
                 label="bracket garanti, mode B (generique)")
    ax1.set_xlabel(r"$\beta = H_m/H_p$")
    ax1.set_ylabel(r"$Q_v\,/\,(\mu_0 H_p^2)$  par cycle")
    ax1.set_title("Pertes AC : encadrement certifie vs analytique")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(bottom=-0.05)

    rel = 2 * np.array(errs) / np.array([float(bean_Q_per_cycle(b, Hp)) for b in betas])
    ax2.bar([f"{b}" for b in betas], 100 * rel, color=C_NUM, width=0.55)
    for i, r in enumerate(rel):
        ax2.text(i, 100 * r + 0.02, f"{100 * r:.2f} %", ha="center", fontsize=9)
    ax2.set_xlabel(r"$\beta$")
    ax2.set_ylabel("largeur relative du bracket [%]")
    ax2.set_title("Finesse de l'encadrement (mode A)")
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig("fig3_bracket.png", dpi=150)
    plt.close(fig)

    # ------------------------------------ fig 4 : convergence du mode B
    Ns = [100, 200, 400, 800]
    ws = []
    for N in Ns:
        mN = BeanSlabCertified(N=N)
        sN, iN = sine_schedule(mN, 0.7, nsteps_per_branch=120)
        hN = _run_schedule(mN, sN, mode="B", check_oracle=False)
        _, eN, _, _ = certified_cycle_loss(mN, hN, iN)
        ws.append(2 * eN)
        print(f"mode B N={N}: largeur bracket = {2 * eN:.0f} J/m^3")
    hs = 2 * 1e-3 / np.array(Ns)
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.loglog(hs, ws, "o-", color=C_NUM, label="largeur du bracket, mode B")
    ref = ws[0] * (hs / hs[0])
    ax.loglog(hs, ref, "--", color=C_B, label=r"pente $O(h)$")
    ax.axhline(2 * errs[1], color=C_ANA, ls=":",
               label=f"mode A (N=200) : {2 * errs[1]:.0f} J/m$^3$")
    ax.set_xlabel("h [m]")
    ax.set_ylabel("largeur du bracket [J/m$^3$]")
    ax.set_title(r"Convergence de l'encadrement generique ($\beta$ = 0.7)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig("fig4_convergence.png", dpi=150)
    plt.close(fig)
    print("figures ecrites : fig1_profils.png fig2_cycle.png "
          "fig3_bracket.png fig4_convergence.png")


if __name__ == "__main__":
    main()
