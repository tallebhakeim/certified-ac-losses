# -*- coding: utf-8 -*-
"""Verification de la revision SuST (R1) : style n/n PASS.

Lit results_revision.json (produit par run_revision.py) et rejoue en direct
le test du Lemme 2 (comparaison, donnee de bord sinusoidale vs affine).
"""

import json
import time
import numpy as np

np.seterr(all="ignore")

from powerlaw_slab_certified import PowerLawSlabCertified
from revision_certificates import sine_defect

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def lemma2_check(n_exp=20, N=200, S=100, sub=8, beta=0.7, freq=50.0):
    """Chaine prox fine : sinus echantillonne finement vs affine par morceaux
    entre les noeuds grossiers. Le Lemme 2 (version discrete : Euler
    implicite d'un operateur monotone) impose max|H_s - H_p| <= D."""
    m = PowerLawSlabCertified(n_exp=n_exp, N=N)
    Hm = beta * m.Hp
    tau = 1.0 / (freq * S)
    tf = tau / sub
    Hs = np.zeros(N + 1)
    Hp_ = np.zeros(N + 1)
    worst = 0.0
    for k in range(S):
        a0 = Hm * np.sin(2 * np.pi * freq * k * tau)
        a1 = Hm * np.sin(2 * np.pi * freq * (k + 1) * tau)
        for j in range(1, sub + 1):
            t = k * tau + j * tf
            Hs, _ = m.prox_step(Hs, Hm * np.sin(2 * np.pi * freq * t), tf, tol=1e-13)
            Hp_, _ = m.prox_step(Hp_, a0 + (a1 - a0) * j / sub, tf, tol=1e-13)
            worst = max(worst, float(np.max(np.abs(Hs - Hp_))))
    return worst, sine_defect(Hm, freq, tau)


def main():
    t0 = time.time()
    with open("results_revision.json") as fh:
        R = json.load(fh)
    print("=" * 72)
    print("Revision SuST : Bean non aligne, sinusoide, effectivite sur la perte")
    print("=" * 72)

    # 1-3 : Bean non aligne, certificat BEN
    bb = R["bean_ben"]
    ok = all(r["contains"] for r in bb)
    check("1. Bean non aligne : tous les brackets BEN contiennent Q_Bean", ok,
          f"{sum(r['contains'] for r in bb)}/{len(bb)} configurations")
    for solver in ("qp", "nodal"):
        w = [r["width"] for r in bb if r["solver"] == solver]
        check(f"2{'a' if solver == 'qp' else 'b'}. largeur decroissante sous raffinement joint ({solver})",
              all(np.diff(w) < 0), " > ".join(f"{100 * x:.1f}%" for x in w))
    qp = [r for r in bb if r["solver"] == "qp"]
    gain = min(2 * r["err_v1B"] / r["QB"] / r["width"] for r in qp)
    check("3. BEN bat l'ancienne consistance par pas (mode B) d'un facteur >= 10",
          gain >= 10, f"gain minimal x{gain:.0f}")

    # 4-6 : loi de puissance, brackets du cycle 3
    pl = R["powerlaw"]
    okp = all(abs(r["Q"] - r["fac_Q_ref"]) <= r["err"] for r in pl)
    check("4. brackets (excitation affine) contiennent Q_ref pour n = 5..80", okp,
          ", ".join(f"n={r['n']}" for r in pl))
    oks = all(abs(r["Qs"] - r["fac_Q_ref"]) <= r["err_s"] for r in pl) \
        and all(r["err_s"] - r["err"] < 1e-3 * r["err"] for r in pl)
    check("5. bracket de la VRAIE sinusoide (Lemme 2) : surcout < 0.1 %", oks,
          ", ".join(f"n={r['n']}: +{100 * (r['err_s'] / r['err'] - 1):.3f}%" for r in pl))
    r20 = [r for r in pl if r["n"] == 20][0]
    cv = r20["curves"]
    okt = all(e >= a for e, a in zip(cv["eps"], cv["err"]))
    check("6. Theoreme 1 valide a TOUS les noeuds de temps (n = 20)", okt,
          f"min eps/err = {min(e / max(a, 1e-30) for e, a in zip(cv['eps'], cv['err'])):.1f}")

    # 7 : decomposition de l'effectivite sur la perte
    okf = all(abs(r["fac_A"] * r["fac_B"] * r["fac_C"] / r["fac_I_Q"] - 1) < 1e-9
              and r["fac_I_Q"] >= 1 for r in pl)
    check("7. effectivite sur la perte = A x B x C, et >= 1", okf,
          ", ".join(f"n={r['n']}: {r['fac_I_Q']:.0f}" for r in pl))

    # 8 : reference adequate
    rc = R["ref_check"]
    dref = abs(rc["Q_ref2"] - r20["fac_Q_ref"])
    check("8. reference fine adequate : |Q_ref - Q_ref2| < |Q_h - Q_ref| / 3",
          dref < r20["fac_dQ"] / 3, f"{dref:.2f} vs {r20['fac_dQ']:.2f} J/m^3")

    # 9 : etude en pas de temps (R2-Q6)
    ts = [r for r in R["timestep"] if r["n"] == 20 and r["N"] == 400]
    ts.sort(key=lambda r: r["S"])
    q = [r["Q"] for r in ts]
    ratio = (q[1] - q[2]) / (q[2] - q[3])
    old20 = ts[0]
    old80 = [r for r in R["timestep"] if r["n"] == 80 and r["N"] == 400 and r["S"] == 200][0]
    check("9. Q(S) ordre 1 ; anciens 2911.2 et 2944.6 (Table 2 v2) = S = 200",
          1.5 < ratio < 2.6 and abs(old20["Q"] - 2911.2) < 0.05
          and abs(old80["Q"] - 2944.6) < 0.05,
          f"ratio {ratio:.2f}, Q(S=200) = {old20['Q']:.2f} (n=20), {old80['Q']:.2f} (n=80)")

    # 10 : Jc(t) prescrit (R1.3)
    jc = R["jc_t"]
    check("10. Jc(t) prescrit : Theoreme 1 valide et bracket contient Q_ref",
          jc["contains"] and jc["eps_T"] >= jc["err_T"],
          f"eps(T) = {jc['eps_T']:.1f} >= {jc['err_T']:.2f} ; "
          f"Q = {jc['Q']:.1f} +/- {jc['err']:.1f}, Q_ref = {jc['Q_ref']:.1f}")

    # 11 : Lemme 2 en direct
    worst, D = lemma2_check()
    check("11. Lemme 2 (comparaison) : max|H_sinus - H_affine| <= D", worst <= D,
          f"{worst:.3e} <= D = {D:.3e} A/m")

    npass = sum(ok for _, ok in RESULTS)
    print("-" * 72)
    print(f"{npass}/{len(RESULTS)} PASS   ({time.time() - t0:.1f} s)")
    return npass == len(RESULTS)


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
