# -*- coding: utf-8 -*-
"""Verification du PoC v3 : certificat espace-temps BEN. Style n/n PASS."""

import time
import numpy as np

np.seterr(all="ignore")

from bean_slab_certified import bean_Q_per_cycle
from powerlaw_slab_certified import (
    MU0, PowerLawSlabCertified, run_powerlaw_cycles, loss_last_cycle,
)
from ben_v3 import ben_certificate, certified_loss_v3

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def pwl_reference(n_exp, N_f, Ha_nodes, tau, nsub_t):
    """Chaine prox sur le MEME probleme (excitation affine par morceaux),
    sous-pas de temps nsub_t et maillage N_f : proxy de la solution exacte."""
    m = PowerLawSlabCertified(n_exp=n_exp, N=N_f)
    H = np.zeros(m.N + 1)
    tau_f = tau / nsub_t
    Ha_prev = 0.0
    mean_hist, Ha_hist = [], []
    for hb in Ha_nodes:
        for j in range(1, nsub_t + 1):
            hbj = Ha_prev + (hb - Ha_prev) * j / nsub_t
            H, _ = m.prox_step(H, hbj, tau_f)
            mean_hist.append(m.meanH(H))
            Ha_hist.append(hbj)
        Ha_prev = hb
    return m, H, np.array(Ha_hist), np.array(mean_hist)


def loop_Q(Ha, mH, S_nodes):
    f = Ha[-S_nodes - 1:] - mH[-S_nodes - 1:]
    dHa = np.diff(Ha[-S_nodes - 1:])
    return MU0 * float(np.sum(0.5 * (f[:-1] + f[1:]) * dHa))


def main():
    t0 = time.time()
    print("=" * 72)
    print("PoC v3 : certificat espace-temps BEN (Theoreme 1)")
    print("=" * 72)

    beta, S, Ncoarse = 0.7, 800, 400
    m = PowerLawSlabCertified(n_exp=20, N=Ncoarse)
    h = run_powerlaw_cycles(m, beta, steps_per_cycle=S, n_cycles=3)
    slabs, eps = ben_certificate(m, h)
    QBean = float(bean_Q_per_cycle(beta, m.Hp))

    # -- 1. structure du certificat -----------------------------------------
    ok1 = np.all(slabs >= 0) and np.all(np.diff(eps) >= -1e-15) \
        and np.all(np.isfinite(eps))
    check("1. slabs FY >= 0, eps(t) croissante et finie", ok1,
          f"eps(T) = {eps[-1]:.2f}, max slab = {np.max(slabs):.2e}")

    # -- 2. VALIDITE du Theoreme 1 : eps(T) >= erreur reelle ----------------
    mf, Hf, Ha_f, mH_f = pwl_reference(20, 2 * Ncoarse, h["Ha"], h["tau"], 4)
    Hc_on_f = np.interp(mf.x, m.x, h["profiles"][len(h["Ha"]) - 1])
    actual_T = mf.norm_L2(Hc_on_f - Hf)
    ok2 = eps[-1] >= actual_T
    check("2. Theoreme 1 VALIDE : eps(T) >= ||H_v(T) - u(T)|| (ref 2x/4x)",
          ok2, f"eps(T) = {eps[-1]:.2f} >= {actual_T:.2f} "
          f"(effectivite {eps[-1] / max(actual_T, 1e-30):.1f})")

    # -- 3. gain BEN vs accumulation v2 -------------------------------------
    E_v2 = h["E"][-1]
    ok3 = E_v2 / eps[-1] >= 2.0
    check("3. BEN bat l'accumulation par pas de la v2 (facteur >= 2)", ok3,
          f"v2 : {E_v2:.1f}, BEN : {eps[-1]:.1f}, gain x{E_v2 / eps[-1]:.1f}")

    # -- 4. bracket v3 ENTIEREMENT certifie contient la reference -----------
    Q3, err3 = certified_loss_v3(m, h, eps)
    Qref = loop_Q(Ha_f, mH_f, S * 4)
    ok4 = (Q3 - err3) <= Qref <= (Q3 + err3)
    check("4. bracket v3 contient Q de la reference sur-resolue", ok4,
          f"Q = {Q3:.1f} +/- {err3:.1f}, Q_ref = {Qref:.1f}")

    # -- 5. finesse du bracket (couvre AUSSI l'erreur en temps) -------------
    ok5 = 2 * err3 / Q3 < 0.20
    check("5. largeur relative du bracket v3 < 20 %", ok5,
          f"{2 * err3 / Q3:.1%} (v2 spatial seul donnait ~50 % sans couvrir le temps)")

    # -- 6. convergence du bracket en tau -----------------------------------
    m2 = PowerLawSlabCertified(n_exp=20, N=Ncoarse)
    h2 = run_powerlaw_cycles(m2, beta, steps_per_cycle=S // 2, n_cycles=3)
    _, eps2 = ben_certificate(m2, h2)
    _, err3_coarse = certified_loss_v3(m2, h2, eps2)
    ok6 = err3_coarse / err3 > 1.25
    check("6. bracket retreci par raffinement en temps (S/2 -> S)", ok6,
          f"err(S={S // 2}) = {err3_coarse:.1f} > err(S={S}) = {err3:.1f} "
          f"(ratio {err3_coarse / err3:.2f}, attendu ~sqrt(2))")

    # -- 7. robustesse en n du certificat BEN -------------------------------
    eps_n = {}
    for n_exp in (5, 20, 80):
        mn = PowerLawSlabCertified(n_exp=n_exp, N=Ncoarse)
        hn = run_powerlaw_cycles(mn, beta, steps_per_cycle=200, n_cycles=3,
                                 with_certificates=False)
        _, en = ben_certificate(mn, hn)
        eps_n[n_exp] = en[-1]
    vals = list(eps_n.values())
    ok7 = max(vals) / min(vals) < 20.0 and all(np.isfinite(vals))
    check("7. robustesse en n : eps(T) du meme ordre pour n = 5..80", ok7,
          "eps(T) = " + ", ".join(f"n={n}:{e:.1f}" for n, e in eps_n.items()))

    # -- 8. verdict Bean a n=80 avec bracket v3 -----------------------------
    m80 = PowerLawSlabCertified(n_exp=80, N=Ncoarse)
    h80 = run_powerlaw_cycles(m80, beta, steps_per_cycle=S, n_cycles=3,
                              with_certificates=False)
    _, eps80 = ben_certificate(m80, h80)
    Q80, err80 = certified_loss_v3(m80, h80, eps80)
    _, _, Ha80f, mH80f = pwl_reference(80, 2 * Ncoarse, h80["Ha"], h80["tau"], 4)
    Qref80 = loop_Q(Ha80f, mH80f, S * 4)
    ok8 = (Q80 - err80) <= Qref80 <= (Q80 + err80) \
        and abs(Q80 - QBean) / QBean < 0.05
    check("8. n=80 : bracket v3 contient la reference, Q a <5 % de Bean", ok8,
          f"Q = {Q80:.1f} +/- {err80:.1f}, Q_ref = {Qref80:.1f}, "
          f"Q/Q_Bean = {Q80 / QBean:.3f}")

    npass = sum(ok for _, ok in RESULTS)
    print("-" * 72)
    print(f"{npass}/{len(RESULTS)} PASS   ({time.time() - t0:.1f} s)")
    print("-" * 72)
    print(f"n=20 : Q = {Q3:.1f} +/- {err3:.1f} J/m^3 (bracket TOTAL : espace"
          f" + temps + QoI), Q_Bean = {QBean:.1f}")
    return npass == len(RESULTS)


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
