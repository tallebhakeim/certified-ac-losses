# -*- coding: utf-8 -*-
"""Verification du PoC pertes AC certifiees (slab de Bean). Style n/n PASS."""

import time
import numpy as np

from bean_slab_certified import (
    MU0, BeanSlabCertified, bean_Q_per_cycle, bean_virgin_meanH,
    bean_loop_meanH_down, bean_profile_virgin, bean_profile_after_reversal,
    aligned_schedule, sine_schedule, _run_schedule, certified_cycle_loss,
)

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def run_aligned(beta, N=200):
    m = BeanSlabCertified(N=N)
    seq, i0 = aligned_schedule(m, beta)
    h = _run_schedule(m, seq, mode="A")
    return m, h, i0


def run_sine(beta, N, nsb):
    m = BeanSlabCertified(N=N)
    seq, i0 = sine_schedule(m, beta, nsteps_per_branch=nsb)
    h = _run_schedule(m, seq, mode="B")
    return m, h, i0


def main():
    t0 = time.time()
    print("=" * 72)
    print("PoC v1 : pertes AC certifiees, slab de Bean (formulation Prigozhin)")
    print("=" * 72)

    # -- 1. quadrature exacte P1 --------------------------------------------
    m = BeanSlabCertified(N=64)
    ok1 = abs(float(np.sum(m.w_int)) - 2 * m.a) < 1e-18
    rows = m.M_diag.copy()
    rows[:-1] += m.M_off
    rows[1:] += m.M_off
    ok1 &= np.allclose(rows, m.w_int, rtol=1e-14)
    check("1. quadrature/masse P1 exactes (somme = 2a, lignes M = poids)", ok1)

    # -- run de reference mode A, beta = 0.7 --------------------------------
    beta = 0.7
    m, h, i0 = run_aligned(beta, N=200)
    Hm, Hp = beta * m.Hp, m.Hp
    scale = m.norm_M(np.full(m.N + 1, Hp))

    # -- 2. profils numeriques vs analytique de Bean ------------------------
    Hpk = h["profiles"][i0 - 1]
    ana_pk = bean_profile_virgin(m.x, Hm, m.a, m.Jc)
    kmid = i0 + (len(h["Ha"]) - i0) // 4          # milieu de la descente
    Hmid = h["profiles"][kmid]
    ana_mid = bean_profile_after_reversal(m.x, Hm, h["Ha"][kmid], m.a, m.Jc)
    dev2 = max(np.max(np.abs(Hpk - ana_pk)), np.max(np.abs(Hmid - ana_mid)))
    check("2. profils H(x) = analytique Bean (pic + mi-descente)",
          dev2 < 1e-7 * Hp, f"L_inf = {dev2:.2e} A/m")

    # -- 3. chaine QP = oracle exact (rattrapage en forme fermee) -----------
    check("3. projections QP = oracle min/max exact",
          h["oracle_dev"] < 1e-7 * scale, f"dev_L2 = {h['oracle_dev']:.2e}")

    # -- 4. certificats : gaps >= 0, admissibilite stricte ------------------
    ok4 = np.all(h["gap_err"] >= 0) and h["min_margin"] >= 0
    check("4. certificats valides (gap >= 0, marge admissibilite >= 0)", ok4,
          f"max gap_err = {np.max(h['gap_err']):.2e}, marge min = {h['min_margin']:.2e}")

    # -- 5. courbe vierge DANS le tube certifie -----------------------------
    virg = slice(0, i0)
    ana_v = bean_virgin_meanH(h["Ha"][virg], Hp)
    eps_v = h["E"][virg] / np.sqrt(2 * m.a) + 1e-9 * Hp
    ok5 = np.all(np.abs(h["meanH"][virg] - ana_v) <= eps_v)
    check("5. courbe de premiere aimantation dans le tube certifie", ok5,
          f"ecart max = {np.max(np.abs(h['meanH'][virg] - ana_v)):.2e}")

    # -- 6. branche descendante vs analytique -------------------------------
    desc = slice(i0, i0 + (len(h["Ha"]) - i0) // 2)
    ana_d = bean_loop_meanH_down(h["Ha"][desc], Hm, Hp)
    dev6 = np.max(np.abs(h["meanH"][desc] - ana_d))
    check("6. branche descendante <H>(Ha) = analytique", dev6 < 1e-7 * Hp,
          f"L_inf = {dev6:.2e}")

    # -- 7-10. LE VERDICT : bracket certifie contient l'analytique ----------
    table = []
    for i, (b, Nb) in enumerate([(0.4, 200), (0.7, 200), (1.0, 200), (1.4, 200)]):
        mb, hb_, ib = run_aligned(b, N=Nb)
        Q, err, es, eq = certified_cycle_loss(mb, hb_, ib)
        Qana = float(bean_Q_per_cycle(b, mb.Hp))
        inside = (Q - err) <= Qana <= (Q + err)
        table.append((b, Q, err, es, eq, Qana, inside))
        check(f"{7 + i}. mode A beta={b} : Q_ana dans [Q-err, Q+err]", inside,
              f"Q = {Q:.1f} +/- {err:.1f} J/m^3, Q_ana = {Qana:.1f}, "
              f"largeur rel. = {2 * err / Qana:.2%}")

    # -- 11. resserrement : bracket mode A sous 1 % -------------------------
    b, Q, err, es, eq, Qana, _ = table[1]
    check("11. mode A beta=0.7 : largeur relative du bracket < 1 %",
          2 * err / Qana < 0.01, f"{2 * err / Qana:.3%} (espace {es:.2f} + quad {eq:.2f})")

    # -- 12. mode B (generique, sinus) : bracket garanti contient Q_ana -----
    mB, hB, iB = run_sine(0.7, N=400, nsb=120)
    QB, errB, esB, eqB = certified_cycle_loss(mB, hB, iB)
    QanaB = float(bean_Q_per_cycle(0.7, mB.Hp))
    okB = (QB - errB) <= QanaB <= (QB + errB) and hB["min_margin"] >= 0
    check("12. mode B (sinus, non aligne) : bracket garanti contient Q_ana", okB,
          f"Q = {QB:.1f} +/- {errB:.1f}, Q_ana = {QanaB:.1f}, "
          f"largeur rel. = {2 * errB / QanaB:.1%}")

    # -- 13. mode B : la largeur du bracket decroit avec h ------------------
    widths = []
    for N in (100, 200, 400):
        mN, hN, iN = run_sine(0.7, N=N, nsb=120)
        _, eN, _, _ = certified_cycle_loss(mN, hN, iN)
        widths.append(eN)
    ok13 = widths[0] > 1.2 * widths[1] > 1.2 * 1.2 * widths[2]
    check("13. mode B : convergence du bracket (h -> h/2 -> h/4)", ok13,
          "err = " + " > ".join(f"{w:.0f}" for w in widths))

    # -- 14. fermeture du cycle (etat au pic initial = etat au pic final) ---
    dev14 = m.norm_M(h["profiles"][i0 - 1] - h["profiles"][len(h["Ha"]) - 1])
    check("14. cycle ferme : etat(pic depart) = etat(pic arrivee)",
          dev14 < 1e-7 * scale, f"dev_L2 = {dev14:.2e}")

    npass = sum(ok for _, ok in RESULTS)
    print("-" * 72)
    print(f"{npass}/{len(RESULTS)} PASS   ({time.time() - t0:.1f} s)")
    print("-" * 72)
    print("beta    Q_num       +/-err      Q_analytique   inclus")
    for b, Q, err, es, eq, Qana, inside in table:
        print(f"{b:4.1f}  {Q:10.2f}  {err:10.2f}  {Qana:12.2f}   {'OUI' if inside else 'NON'}")
    return npass == len(RESULTS)


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
