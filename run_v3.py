# -*- coding: utf-8 -*-
"""Figures du PoC v3 (certificat espace-temps BEN). Sorties : figv3_*.png."""

import numpy as np
np.seterr(all="ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bean_slab_certified import bean_Q_per_cycle
from powerlaw_slab_certified import (
    MU0, PowerLawSlabCertified, run_powerlaw_cycles,
)
from ben_v3 import ben_certificate, certified_loss_v3

C_ANA, C_V2, C_V3 = "#d62728", "#7f7f7f", "#1f77b4"


def main():
    beta, S, N = 0.7, 800, 400
    ns = [5, 10, 20, 40, 80]

    # ------------------- flagship n=20 avec certificats v2 ET v3 -----------
    m20 = PowerLawSlabCertified(n_exp=20, N=N)
    h20 = run_powerlaw_cycles(m20, beta, steps_per_cycle=S, n_cycles=3,
                              with_certificates=True)
    slabs, eps20 = ben_certificate(m20, h20)
    t = h20["t"]

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.semilogy(t * 1e3, h20["E"], "-", color=C_V2, lw=1.6,
                label="v2 : accumulation par pas (croissance ~S)")
    ax.semilogy(t * 1e3, eps20, "-", color=C_V3, lw=1.8,
                label="v3 : majorant BEN espace-temps (Theoreme 1)")
    for c in (1, 2, 3):
        ax.axvline(c * 20.0, color="k", ls=":", lw=0.6, alpha=0.5)
    ax.set_xlabel("t [ms]  (3 cycles a 50 Hz)")
    ax.set_ylabel(r"borne certifiee sur $\|H_v(t) - u(t)\|_{L^2}$")
    ax.set_title(f"BEN vs accumulation : gain x{h20['E'][-1] / eps20[-1]:.0f} "
                 f"(n = 20, N = {N}, S = {S})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig("figv3_epsilon.png", dpi=150)
    plt.close(fig)
    print(f"n=20 : eps_BEN(T) = {eps20[-1]:.1f}, E_v2(T) = {h20['E'][-1]:.1f}, "
          f"gain x{h20['E'][-1] / eps20[-1]:.1f}")

    # ------------------- Q(n) avec brackets v3 -----------------------------
    Qs, errs = {}, {}
    Qs[20], errs[20] = certified_loss_v3(m20, h20, eps20)
    for n_exp in ns:
        if n_exp == 20:
            continue
        m = PowerLawSlabCertified(n_exp=n_exp, N=N)
        h = run_powerlaw_cycles(m, beta, steps_per_cycle=S, n_cycles=3,
                                with_certificates=False)
        _, eps = ben_certificate(m, h)
        Qs[n_exp], errs[n_exp] = certified_loss_v3(m, h, eps)
    for n_exp in ns:
        print(f"n = {n_exp:3d} : Q = {Qs[n_exp]:8.1f} +/- {errs[n_exp]:6.1f} "
              f"J/m^3 (certifie TOTAL, largeur {2 * errs[n_exp] / Qs[n_exp]:.1%})")

    Hp = m20.Hp
    QBean = float(bean_Q_per_cycle(beta, Hp))
    qn = MU0 * Hp**2
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.axhline(QBean / qn, color=C_ANA, ls="--", lw=1.6,
               label=r"Bean ($n \to \infty$), analytique")
    ax.errorbar(ns, [Qs[n] / qn for n in ns], yerr=[errs[n] / qn for n in ns],
                fmt="o-", color=C_V3, capsize=5, ms=5, lw=1.4,
                label="bracket v3 CERTIFIE TOTAL (espace + temps + QoI)")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("exposant n de la loi E-J")
    ax.set_ylabel(r"$Q_v/(\mu_0 H_p^2)$ par cycle")
    ax.set_title("Pertes AC certifiees face au continuum (Theoreme 1)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig("figv3_bracket.png", dpi=150)
    plt.close(fig)
    print(f"Q_Bean = {QBean:.1f} J/m^3")
    print("figures ecrites : figv3_epsilon.png figv3_bracket.png")


if __name__ == "__main__":
    main()
