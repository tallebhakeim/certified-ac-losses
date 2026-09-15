# -*- coding: utf-8 -*-
"""English figures for the SuST manuscript (fig_en_*.png), including the
concept figure. Recomputes everything from the certified modules."""

import numpy as np
np.seterr(all="ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from bean_slab_certified import (
    MU0, BeanSlabCertified, bean_Q_per_cycle, bean_loop_meanH_down,
    bean_virgin_meanH, aligned_schedule, sine_schedule, _run_schedule,
    certified_cycle_loss,
)
from powerlaw_slab_certified import (
    PowerLawSlabCertified, run_powerlaw_cycles,
)
from ben_v3 import ben_certificate, certified_loss_v3

C_ANA, C_NUM, C_B = "#d62728", "#1f77b4", "#7f7f7f"
CMAP = plt.cm.viridis
BETA = 0.7


# ------------------------------------------------------------- concept figure
def concept_figure():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.2),
                                   gridspec_kw={"width_ratios": [2, 3]})
    # (a) slab geometry with Bean-like profile
    ax1.set_xlim(-1.45, 1.45)
    ax1.set_ylim(-0.18, 1.18)
    ax1.axis("off")
    ax1.set_title("(a)  slab in an AC applied field", fontsize=10)
    ax1.axvline(-1, color="k", lw=1.4)
    ax1.axvline(1, color="k", lw=1.4)
    ax1.axhline(0, color="k", lw=0.5, alpha=0.4)
    x = np.linspace(-1, 1, 200)
    prof = np.maximum(0.8 - 1.6 * np.minimum(x + 1, 1 - x), 0.0)
    ax1.plot(x, prof, color=C_NUM, lw=2.0)
    ax1.fill_between(x, 0, prof, color=C_NUM, alpha=0.15)
    for xa, dx in ((-1.35, 0.22), (1.35, -0.22)):
        ax1.annotate("", xy=(xa + dx, 0.82), xytext=(xa, 0.82),
                     arrowprops=dict(arrowstyle="->", color=C_ANA, lw=1.6))
    ax1.text(-1.36, 0.93, r"$H_a(t)$", color=C_ANA, fontsize=10)
    ax1.text(1.06, 0.93, r"$H_a(t)$", color=C_ANA, fontsize=10, ha="left")
    ax1.text(0.0, 0.55, r"$H(x,t)$", color=C_NUM, fontsize=10, ha="center")
    ax1.annotate(r"front", xy=(-0.5, 0.02), xytext=(-0.42, 0.3),
                 fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.9))
    ax1.text(0.0, -0.155, r"$|J| \leq J_c$ (Bean)  or  $E \propto |J|^{n-1}J$",
             fontsize=9, ha="center")
    ax1.text(-1, -0.05, r"$-a$", fontsize=9, ha="right")
    ax1.text(1, -0.05, r"$+a$", fontsize=9, ha="left")

    # (b) certification pipeline
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis("off")
    ax2.set_title("(b)  the certificate: both field laws exact, "
                  "the material law relaxed", fontsize=10)

    def box(xc, yc, w, h, lines, fc):
        ax2.add_patch(FancyBboxPatch((xc - w / 2, yc - h / 2), w, h,
                      boxstyle="round,pad=0.12", fc=fc, ec="k", lw=0.9))
        ax2.text(xc, yc, "\n".join(lines), ha="center", va="center", fontsize=8.6)

    def arrow(p, q):
        ax2.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13,
                                      lw=1.1, color="k"))

    box(2.5, 8.6, 4.4, 1.7, ["Ampere's law: EXACT",
                             r"trajectory $H_v$,  $J = -\partial_x H_v$"], "#dce6f1")
    box(7.5, 8.6, 4.4, 1.7, ["Faraday's law: EXACT",
                             r"reconstructed field  $\partial_x s = \mu_0\,\partial_t H_v$"], "#dce6f1")
    box(5.0, 5.9, 6.4, 1.7, ["constitutive law E(J): RELAXED",
                             r"Fenchel-Young residual  $\Phi(J) + \Phi^*(s) - sJ \geq 0$"], "#fde9d9")
    box(5.0, 3.2, 7.6, 1.7, ["Theorem 1 (Brezis-Ekeland-Nayroles majorant)",
        r"$\frac{\mu_0}{2}\,\|H_v(t)-u(t)\|^2_{L^2} \;\leq\; \iint \mathrm{FY}\,\,dx\,dt$   at every $t$"], "#e2efda")
    box(5.0, 0.9, 6.6, 1.4, ["certified loss bracket",
        r"$Q \in [\,Q_v - \epsilon,\; Q_v + \epsilon\,]$, guaranteed"], "#e2efda")
    arrow((2.5, 7.6), (3.6, 6.85))
    arrow((7.5, 7.6), (6.4, 6.85))
    arrow((5.0, 4.95), (5.0, 4.15))
    arrow((5.0, 2.25), (5.0, 1.7))
    fig.tight_layout()
    fig.savefig("fig_en_concept.png", dpi=150)
    plt.close(fig)
    print("fig_en_concept.png")


# --------------------------------------------------------------- Bean figure
def bean_figure():
    betas = [0.4, 0.7, 1.0, 1.4]
    Qs, errs = [], []
    Hp = None
    for b in betas:
        m = BeanSlabCertified(N=200)
        Hp = m.Hp
        sq, ib = aligned_schedule(m, b)
        h = _run_schedule(m, sq, mode="A", check_oracle=False)
        Q, err, _, _ = certified_cycle_loss(m, h, ib)
        Qs.append(Q)
        errs.append(err)
    mB = BeanSlabCertified(N=400)
    sqB, iB = sine_schedule(mB, BETA, nsteps_per_branch=120)
    hB = _run_schedule(mB, sqB, mode="B", check_oracle=False)
    QB, errB, _, _ = certified_cycle_loss(mB, hB, iB)

    qn = MU0 * Hp**2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.4),
                                   gridspec_kw={"width_ratios": [3, 2]})
    bb = np.linspace(0.05, 1.6, 300)
    ax1.plot(bb, bean_Q_per_cycle(bb, Hp) / qn, "-", color=C_ANA, lw=1.4,
             label="Bean, analytical")
    ax1.errorbar(betas, np.array(Qs) / qn, yerr=np.array(errs) / qn, fmt="o",
                 color=C_NUM, capsize=5, ms=5, lw=1.6,
                 label="GUARANTEED bracket, aligned mode")
    ax1.errorbar([0.72], [QB / qn], yerr=[errB / qn], fmt="s", mfc="none",
                 color=C_B, capsize=4, ms=6, lw=1.2,
                 label="guaranteed bracket, generic mode")
    ax1.set_xlabel(r"$\beta = H_m/H_p$")
    ax1.set_ylabel(r"$Q_v\,/\,(\mu_0 H_p^2)$  per cycle")
    ax1.set_title("AC losses: certified enclosure vs analytical")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(bottom=-0.05)
    rel = 2 * np.array(errs) / np.array([float(bean_Q_per_cycle(b, Hp)) for b in betas])
    ax2.bar([f"{b}" for b in betas], 100 * rel, color=C_NUM, width=0.55)
    for i, r in enumerate(rel):
        ax2.text(i, 100 * r + 0.02, f"{100 * r:.2f}%", ha="center", fontsize=9)
    ax2.set_xlabel(r"$\beta$")
    ax2.set_ylabel("relative bracket width [%]")
    ax2.set_title("Sharpness of the enclosure (aligned mode)")
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig("fig_en_bean_bracket.png", dpi=150)
    plt.close(fig)
    print("fig_en_bean_bracket.png")


# ---------------------------------------------------------- power-law figures
def powerlaw_figures():
    ns = [5, 10, 20, 40, 80]
    runs = {}
    for n_exp in ns:
        m = PowerLawSlabCertified(n_exp=n_exp, N=800)
        h = run_powerlaw_cycles(m, BETA, steps_per_cycle=200, n_cycles=3,
                                with_certificates=False)
        runs[n_exp] = (m, h)
    m0 = runs[ns[0]][0]
    Hp = m0.Hp
    Hm = BETA * Hp

    # loops
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
    ax.plot(ha_d / Hp, md, "--", color=C_ANA, lw=2.0, label="Bean (analytical)")
    ax.plot(-ha_d / Hp, -md, "--", color=C_ANA, lw=2.0)
    ax.set_xlabel(r"$H_a/H_p$")
    ax.set_ylabel(r"$\langle M\rangle/H_p$")
    ax.set_title(rf"Power law: steady loops vs the Bean model ($\beta$ = {BETA}, 50 Hz)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig_en_cycles.png", dpi=150)
    plt.close(fig)
    print("fig_en_cycles.png")

    # J profiles
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for i, n_exp in enumerate([10, 20, 80]):
        m, h = runs[n_exp]
        kq = len(h["Ha"]) - 100
        J = -np.diff(h["profiles"][kq]) / m.h
        xm = 0.5 * (m.x[:-1] + m.x[1:]) * 1e3
        ax.plot(xm, J / m.Jc, "-", color=CMAP(i / 2), lw=1.4, label=f"n = {n_exp}")
    ax.axhline(1.0, color=C_ANA, ls="--", lw=1.2, label=r"$\pm J_c$ (Bean)")
    ax.axhline(-1.0, color=C_ANA, ls="--", lw=1.2)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel(r"$J/J_c$")
    ax.set_title(r"Current profiles at $H_a = 0$ (maximal ramp rate): "
                 r"flux creep beyond $J_c$")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig_en_profiles_J.png", dpi=150)
    plt.close(fig)
    print("fig_en_profiles_J.png")

    # effectivity (n-robustness)
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
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(ns, effs, "o-", color=C_NUM, lw=1.5, ms=6)
    ax.axhline(1.0, color=C_B, ls=":", label="ideal effectivity = 1")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_ylim(0, max(effs) * 1.3)
    ax.set_xlabel("power-law exponent n")
    ax.set_ylabel("effectivity = certified bound / true error")
    ax.set_title("n-ROBUSTNESS of the primal-dual certificate (prox step)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("fig_en_robustness.png", dpi=150)
    plt.close(fig)
    print("fig_en_robustness.png")


# ------------------------------------------------------------- BEN v3 figures
def ben_figures():
    S, N = 800, 400
    ns = [5, 10, 20, 40, 80]
    m20 = PowerLawSlabCertified(n_exp=20, N=N)
    h20 = run_powerlaw_cycles(m20, BETA, steps_per_cycle=S, n_cycles=3,
                              with_certificates=True)
    _, eps20 = ben_certificate(m20, h20)
    t = h20["t"]

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.semilogy(t * 1e3, h20["E"], "-", color=C_B, lw=1.6,
                label="step-accumulated certificates (section 3)")
    ax.semilogy(t * 1e3, eps20, "-", color=C_NUM, lw=1.8,
                label="space-time BEN majorant (Theorem 1)")
    for c in (1, 2, 3):
        ax.axvline(c * 20.0, color="k", ls=":", lw=0.6, alpha=0.5)
    ax.set_xlabel("t [ms]  (three 50 Hz cycles)")
    ax.set_ylabel(r"certified bound on $\|H_v(t) - u(t)\|_{L^2}$")
    ax.set_title(f"BEN vs accumulation: factor {h20['E'][-1] / eps20[-1]:.0f} "
                 f"(n = 20, N = {N}, S = {S})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig("fig_en_epsilon.png", dpi=150)
    plt.close(fig)
    print("fig_en_epsilon.png")

    Qs, errs = {}, {}
    Qs[20], errs[20] = certified_loss_v3(m20, h20, eps20)
    for n_exp in ns:
        if n_exp == 20:
            continue
        m = PowerLawSlabCertified(n_exp=n_exp, N=N)
        h = run_powerlaw_cycles(m, BETA, steps_per_cycle=S, n_cycles=3,
                                with_certificates=False)
        _, eps = ben_certificate(m, h)
        Qs[n_exp], errs[n_exp] = certified_loss_v3(m, h, eps)
    Hp = m20.Hp
    QBean = float(bean_Q_per_cycle(BETA, Hp))
    qn = MU0 * Hp**2
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.axhline(QBean / qn, color=C_ANA, ls="--", lw=1.6,
               label=r"Bean ($n \to \infty$), analytical")
    ax.errorbar(ns, [Qs[n] / qn for n in ns], yerr=[errs[n] / qn for n in ns],
                fmt="o-", color=C_NUM, capsize=5, ms=5, lw=1.4,
                label="FULLY certified bracket (space + time + QoI)")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("power-law exponent n")
    ax.set_ylabel(r"$Q_v/(\mu_0 H_p^2)$ per cycle")
    ax.set_title("Certified AC losses against the continuum (Theorem 1)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig("fig_en_v3_bracket.png", dpi=150)
    plt.close(fig)
    print("fig_en_v3_bracket.png")


if __name__ == "__main__":
    concept_figure()
    bean_figure()
    powerlaw_figures()
    ben_figures()
    print("done.")
