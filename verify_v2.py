# -*- coding: utf-8 -*-
"""Verification du PoC v2 : loi de puissance E~J^n certifiee par ecart
primal-dual (Bregman). Style n/n PASS."""

import time
import numpy as np

np.seterr(all="ignore")

from bean_slab_certified import bean_Q_per_cycle
from powerlaw_slab_certified import (
    MU0, PowerLawSlabCertified, run_powerlaw_cycles, loss_last_cycle,
)

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def one_step_setup(n_exp, N, beta=0.7, S=200):
    """Prepare un pas prox 'typique' (mi-montee du premier quart de cycle)."""
    m = PowerLawSlabCertified(n_exp=n_exp, N=N)
    tau = 1.0 / (50.0 * S)
    Hm = beta * m.Hp
    # quelques pas pour avoir un etat non trivial
    H = np.zeros(m.N + 1)
    for k in range(1, 11):
        hb = Hm * np.sin(2 * np.pi * 50.0 * tau * k)
        H, _ = m.prox_step(H, hb, tau)
    g = H
    hb = Hm * np.sin(2 * np.pi * 50.0 * tau * 11)
    return m, g, hb, tau


def main():
    t0 = time.time()
    print("=" * 72)
    print("PoC v2 : loi de puissance E ~ J^n, certificat par ecart primal-dual")
    print("=" * 72)

    # -- 1. paire conjuguee (Fenchel-Young) ---------------------------------
    m = PowerLawSlabCertified(n_exp=20, N=16)
    s = np.linspace(-1.6, 1.6, 41) * m.Jc
    fy_eq = m.Phi(s) + m.Phi_star(m.phi(s)) - m.phi(s) * s
    rng = np.random.default_rng(0)
    ss = rng.uniform(-1.5, 1.5, 200) * m.Jc
    gg = rng.uniform(-5, 5, 200) * m.Ec
    fy_rand = m.Phi(ss) + m.Phi_star(gg) - gg * ss
    ok1 = np.max(np.abs(fy_eq)) < 1e-10 * m.Ec * m.Jc and np.min(fy_rand) > -1e-12
    check("1. Fenchel-Young : egalite en sigma=phi(s), positivite", ok1,
          f"max|FY(s,phi(s))| = {np.max(np.abs(fy_eq)):.2e}")

    # -- 2. pas prox : residu de Newton -------------------------------------
    m, g, hb, tau = one_step_setup(20, 400)
    H, it = m.prox_step(g, hb, tau)
    s_el = np.diff(H) / m.h
    grad = MU0 / tau * m.M_dot(H - g)
    f = m.phi(s_el)
    grad[:-1] -= f
    grad[1:] += f
    res2 = np.max(np.abs(grad[1:m.N]))
    check("2. pas prox resolu (residu Newton libre ~ 0)", res2 < 1e-6,
          f"residu = {res2:.2e}, {it} iterations")

    # -- 3. identite du gap : F - D(sigma) et surestimation controlee -------
    gap, err, c_opt = m.certified_gap(H, g, hb, tau)
    FmD = m.F_primal(H, g, tau) - m.dual_value(c_opt, H, g, hb, tau)
    ok3 = gap >= FmD - 1e-10 * abs(m.F_primal(H, g, tau)) and FmD >= -1e-12 \
        and gap <= 3.0 * max(FmD, 1e-30) + 1e-10
    check("3. identite primal-dual : 0 <= F - D <= gap_sup <= 3 (F - D)", ok3,
          f"F - D = {FmD:.3e}, gap_sup = {gap:.3e}, ratio = {gap / max(FmD, 1e-30):.2f}")

    # -- 4-5. validite et robustesse en n du certificat ---------------------
    effs = []
    ok45 = True
    for n_exp in (5, 10, 20, 40, 80):
        mc, gc, hbc, tauc = one_step_setup(n_exp, 200)
        Hc, _ = mc.prox_step(gc, hbc, tauc)
        _, errc, _ = mc.certified_gap(Hc, gc, hbc, tauc)
        # reference sur-resolue : meme pas, maillage emboite 8x plus fin
        mf = PowerLawSlabCertified(n_exp=n_exp, N=200 * 8)
        gf = np.interp(mf.x, mc.x, gc)
        Hf, _ = mf.prox_step(gf, hbc, tauc, tol=1e-14)
        Hc_on_f = np.interp(mf.x, mc.x, Hc)
        actual = mf.norm_L2(Hc_on_f - Hf)
        eff = errc / max(actual, 1e-30)
        effs.append((n_exp, errc, actual, eff))
        ok45 &= errc >= actual
    check("4. borne certifiee VALIDE (>= erreur reelle vs maillage 8x)", ok45,
          "; ".join(f"n={n}: {e:.2e}>={a:.2e}" for n, e, a, _ in effs[:2]))
    eff_vals = [e for _, _, _, e in effs]
    ok5 = max(eff_vals) < 100.0 and max(eff_vals) / min(eff_vals) < 20.0
    check("5. ROBUSTESSE EN n : effectivite bornee, n = 5..80", ok5,
          "eff = " + ", ".join(f"{n}:{e:.1f}" for (n, _, _, e) in effs))

    # -- 6. run complet n=20 (N=800) : certificats sains --------------------
    m20 = PowerLawSlabCertified(n_exp=20, N=800)
    h20 = run_powerlaw_cycles(m20, 0.7, steps_per_cycle=200, n_cycles=3)
    ok6 = np.all(h20["gap"] >= 0) and np.all(np.isfinite(h20["E"])) \
        and int(np.median(h20["newton_it"])) < 40
    check("6. run 3 cycles n=20 : gaps >= 0 partout, chaine finie", ok6,
          f"max gap = {np.max(h20['gap']):.2e}, iters median/max = "
          f"{int(np.median(h20['newton_it']))}/{np.max(h20['newton_it'])}")

    # -- 7. quasi-periodicite mesuree ---------------------------------------
    S = h20["steps_per_cycle"]
    kend = len(h20["Ha"]) - 1
    d32 = m20.norm_L2(h20["profiles"][kend] - h20["profiles"][kend - S])
    d21 = m20.norm_L2(h20["profiles"][kend - S] - h20["profiles"][kend - 2 * S])
    scale = m20.norm_L2(np.full(m20.N + 1, m20.Hp))
    check("7. transitoire contracte (derive cycle 3-2 < cycle 2-1, petite)",
          d32 < d21 and d32 < 5e-3 * scale,
          f"d21 = {d21:.3e}, d32 = {d32:.3e} (echelle {scale:.1e})")

    # -- 8. Richardson en temps : O(tau) ------------------------------------
    Qs_tau = []
    for S_r in (50, 100, 200):
        mr = PowerLawSlabCertified(n_exp=20, N=400)
        hr = run_powerlaw_cycles(mr, 0.7, steps_per_cycle=S_r, n_cycles=3,
                                 with_certificates=False)
        Qr, _, _ = loss_last_cycle(mr, hr)
        Qs_tau.append(Qr)
    e1, e2 = abs(Qs_tau[1] - Qs_tau[0]), abs(Qs_tau[2] - Qs_tau[1])
    check("8. Richardson en tau : |Q(tau/2)-Q(tau)| decroit (~O(tau))",
          e2 < 0.8 * e1, f"{e1:.2f} -> {e2:.2f} J/m^3 (ratio {e2 / e1:.2f})")

    # -- 9. bracket spatial certifie du run n=20 ----------------------------
    Q20, err_sp20, drift20 = loss_last_cycle(m20, h20)
    check("9. partie spatiale certifiee du bracket raisonnable (< 80 %)",
          err_sp20 / Q20 < 0.8,
          f"Q = {Q20:.1f} +/- {err_sp20:.1f} J/m^3 (spatial, {err_sp20 / Q20:.1%})")

    # -- 10. limite de Bean : Q(n) suite de Cauchy, Q(80) proche de Bean ----
    QBean = float(bean_Q_per_cycle(0.7, m20.Hp))
    Qn = {}
    for n_exp in (5, 10, 20, 40, 80):
        mn = PowerLawSlabCertified(n_exp=n_exp, N=400)
        hn = run_powerlaw_cycles(mn, 0.7, steps_per_cycle=200, n_cycles=3,
                                 with_certificates=False)
        Qn[n_exp], _, _ = loss_last_cycle(mn, hn)
    ns = sorted(Qn)
    diffs = [abs(Qn[ns[i + 1]] - Qn[ns[i]]) for i in range(len(ns) - 1)]
    ok10 = all(diffs[i + 1] < diffs[i] for i in range(len(diffs) - 1)) \
        and abs(Qn[80] - QBean) / QBean < 0.05
    check("10. limite de Bean : Q(n) Cauchy (ecarts decroissants), "
          "Q(80) a <5 % de Q_Bean", ok10,
          "diffs = " + " > ".join(f"{d:.0f}" for d in diffs)
          + f" ; Q(80)/Q_Bean = {Qn[80] / QBean:.3f}")

    # -- 11. physique du fluage : depassement de Jc decroissant en n --------
    kpk = len(h20["Ha"]) - 1 - 3 * S // 4          # proche d'un pic de Ha
    Jmax = {}
    for n_exp in (10, 80):
        mn = PowerLawSlabCertified(n_exp=n_exp, N=400)
        hn = run_powerlaw_cycles(mn, 0.7, steps_per_cycle=200, n_cycles=1,
                                 with_certificates=False)
        kq = len(hn["Ha"]) // 4                     # premier pic
        Jmax[n_exp] = np.max(np.abs(np.diff(hn["profiles"][kq]) / mn.h)) / mn.Jc
    check("11. depassement de Jc : J_max/Jc > 1 et decroit avec n",
          Jmax[10] > 1.02 and Jmax[80] < Jmax[10],
          f"J_max/Jc : n=10 -> {Jmax[10]:.3f}, n=80 -> {Jmax[80]:.3f}")

    # -- 12. echelle du champ electrique coherente --------------------------
    S_ = h20["steps_per_cycle"]
    kq = len(h20["Ha"]) - S_ // 2                 # zero de Ha, |dHa/dt| maximal
    Emax = np.max(np.abs(m20.phi(np.diff(h20["profiles"][kq]) / m20.h)))
    Eop = MU0 * 2 * np.pi * 50.0 * 0.7 * m20.Hp * m20.a
    check("12. echelle de E : max|phi(H')| ~ mu0 w Hm a (facteur < 5)",
          0.2 * Eop < Emax < 5.0 * Eop,
          f"E_max = {Emax:.3e} V/m, mu0 w Hm a = {Eop:.3e} V/m")

    npass = sum(ok for _, ok in RESULTS)
    print("-" * 72)
    print(f"{npass}/{len(RESULTS)} PASS   ({time.time() - t0:.1f} s)")
    print("-" * 72)
    print(f"Q_Bean(beta=0.7) = {QBean:.1f} J/m^3")
    for n in sorted(Qn):
        print(f"  n = {n:3d} : Q = {Qn[n]:8.1f} J/m^3   (Q/Q_Bean = {Qn[n] / QBean:.3f})")
    return npass == len(RESULTS)


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
