# -*- coding: utf-8 -*-
"""Revision SuST (SUST-107877) : tous les chiffres nouveaux du manuscrit R1.

Ecrit results_revision.json (lu par build_manuscript.js et
build_response.js : aucun nombre n'est recopie a la main) et les figures
fig_en_bean_ben.png, fig_en_loss_effectivity.png.

Discretisation UNIQUE pour la loi de puissance (R2-Q6) : N = 400, S = 800,
beta = 0.7, 50 Hz, 3 cycles. Duree ~ 15-25 min (processus paralleles).
"""

import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
np.seterr(all="ignore")

from bean_slab_certified import (
    MU0, BeanSlabCertified, bean_Q_per_cycle, sine_schedule, _run_schedule,
    certified_cycle_loss,
)
from powerlaw_slab_certified import PowerLawSlabCertified, run_powerlaw_cycles
from ben_v3 import ben_certificate
from revision_certificates import (
    bean_nodal_chain, ben_bean_certificate, bean_certified_loss_ben,
    certified_loss_cycle, certified_periodicity, pwl_reference_states,
    loss_from_states, loss_effectivity_factors,
)

BETA, FREQ, N, S, NCYC = 0.7, 50.0, 400, 800, 3
NS = [5, 10, 20, 40, 80]
OUT = "results_revision.json"


# ---------------------------------------------------------------- loi de puissance
def powerlaw_case(n_exp):
    """Run unifie + certificats + reference fine (excitation affine)."""
    m = PowerLawSlabCertified(n_exp=n_exp, N=N)
    t0 = time.process_time()
    h = run_powerlaw_cycles(m, BETA, freq=FREQ, steps_per_cycle=S,
                            n_cycles=NCYC, with_certificates=False)
    t_solve = time.process_time() - t0
    t0 = time.process_time()
    slabs, eps = ben_certificate(m, h)
    t_ben = time.process_time() - t0
    K = len(h["Ha"])
    c3 = certified_loss_cycle(m, h, eps, 3, BETA, FREQ)
    c2 = certified_loss_cycle(m, h, eps, 2, BETA, FREQ)
    per = certified_periodicity(m, h, eps)
    # courant a H_a = 0 (vitesse maximale), dernier cycle
    kq = K - S // 2
    J = np.abs(np.diff(h["profiles"][kq])) / m.h / m.Jc
    jmax_cycle = max(float(np.max(np.abs(np.diff(h["profiles"][k])))) / m.h / m.Jc
                     for k in range(K - S, K))
    # reference fine : meme probleme (affine par morceaux), 2x espace, 4x temps
    mref, ref = pwl_reference_states(n_exp, 2 * N, h["Ha"], h["tau"], 4)
    fac = loss_effectivity_factors(m, h, eps, mref, ref)
    # courbes temporelles (n = 20 seulement) pour la figure Q3
    curves = None
    if n_exp == 20:
        en = [mref.norm_L2(np.interp(mref.x, m.x, h["profiles"][k]) - ref[k])
              for k in range(K)]
        curves = dict(t=h["t"].tolist(), eps=eps.tolist(), err=en)
    return dict(n=n_exp, Q=c3["Q_p"], err=c3["err_p"], Qs=c3["Qhat_s"],
                err_s=c3["err_s"], D=c3["D"], Q2=c2["Q_p"], err2=c2["err_p"],
                periodicity=per, Ha_kq=float(h["Ha"][kq]),
                jmax_zero=float(np.max(J)), jmax_cycle=jmax_cycle,
                x_jmax_mm=float(0.5 * (m.x[np.argmax(J)] + m.x[np.argmax(J) + 1]) * 1e3),
                eps_T=float(eps[-1]), t_solve=t_solve, t_ben=t_ben,
                n_steps=K, **{"fac_" + k: v for k, v in fac.items()},
                curves=curves)


def timestep_study(args):
    """Q du dernier cycle en fonction de S (R2-Q6 : 2911.2 vs 2926.8)."""
    n_exp, Nx, Sx = args
    m = PowerLawSlabCertified(n_exp=n_exp, N=Nx)
    h = run_powerlaw_cycles(m, BETA, freq=FREQ, steps_per_cycle=Sx,
                            n_cycles=NCYC, with_certificates=False)
    c3 = certified_loss_cycle(m, h, np.zeros(len(h["Ha"])), 3, BETA, FREQ)
    return dict(n=n_exp, N=Nx, S=Sx, Q=c3["Q_p"], Qs=c3["Qhat_s"])


def cost_v2_gap(n_exp=20):
    """Cout du certificat par pas (v2) et du certificat BEN, n = 20."""
    m = PowerLawSlabCertified(n_exp=n_exp, N=N)
    t0 = time.process_time()
    h = run_powerlaw_cycles(m, BETA, freq=FREQ, steps_per_cycle=S,
                            n_cycles=NCYC, with_certificates=True)
    t_tot = time.process_time() - t0
    return dict(t_solve_plus_gap=t_tot, E_v2_T=float(h["E"][-1]))


def reference_check(n_exp=20):
    """Erreur de la reference elle-meme : ref (2x, 4x) contre ref (4x, 8x)."""
    m = PowerLawSlabCertified(n_exp=n_exp, N=N)
    h = run_powerlaw_cycles(m, BETA, freq=FREQ, steps_per_cycle=S,
                            n_cycles=NCYC, with_certificates=False)
    mref2, ref2 = pwl_reference_states(n_exp, 4 * N, h["Ha"], h["tau"], 8)
    return dict(n=n_exp, Q_ref2=loss_from_states(mref2, h["Ha"], ref2, S))


def jc_time_dependent(n_exp=20, H0_over_Hp=1.0):
    """R1.3 : Jc(t) = Jc0 / (1 + |H_a(t)| / H0) prescrit (champ applique,
    modele de Kim gele) : Phi(t, .) reste convexe, le Theoreme 1 s'applique."""
    m = PowerLawSlabCertified(n_exp=n_exp, N=N)
    tau = 1.0 / (FREQ * S)
    tg = tau * np.arange(1, NCYC * S + 1)
    Hm = BETA * m.Hp
    tm = tg - 0.5 * tau                                   # milieu de slab
    Jc_seq = m.Jc / (1.0 + np.abs(Hm * np.sin(2 * np.pi * FREQ * tm)) / (H0_over_Hp * m.Hp))
    h = run_powerlaw_cycles(m, BETA, freq=FREQ, steps_per_cycle=S,
                            n_cycles=NCYC, with_certificates=False, Jc_seq=Jc_seq)
    _, eps = ben_certificate(m, h, Jc_seq=Jc_seq)
    c3 = certified_loss_cycle(m, h, eps, 3, BETA, FREQ)
    mref, ref = pwl_reference_states(n_exp, 2 * N, h["Ha"], h["tau"], 4,
                                     Jc_seq=Jc_seq, Jc0=m.Jc)
    err_T = mref.norm_L2(np.interp(mref.x, m.x, h["profiles"][len(h["Ha"]) - 1]) - ref[-1])
    Q_ref = loss_from_states(mref, h["Ha"], ref, S)
    return dict(n=n_exp, Jc_min_over_Jc0=float(Jc_seq.min() / m.Jc),
                Q=c3["Q_p"], err=c3["err_p"], Q_ref=Q_ref,
                eps_T=float(eps[-1]), err_T=float(err_T),
                contains=bool(abs(c3["Q_p"] - Q_ref) <= c3["err_p"]))


def step_effectivities():
    """Effectivite du certificat PAR PAS (section 3, figure 4) : meme protocole
    que make_figures_en.py (N = 200 contre reference N = 1600, pas 11)."""
    out = {}
    Hm = BETA * PowerLawSlabCertified().Hp
    for n_exp in NS:
        mc = PowerLawSlabCertified(n_exp=n_exp, N=200)
        tau = 1.0 / (50.0 * 200)
        H = np.zeros(mc.N + 1)
        for k in range(1, 11):
            H, _ = mc.prox_step(H, Hm * np.sin(2 * np.pi * 50.0 * tau * k), tau)
        g = H
        hb = Hm * np.sin(2 * np.pi * 50.0 * tau * 11)
        Hc, _ = mc.prox_step(g, hb, tau)
        _, errc, _ = mc.certified_gap(Hc, g, hb, tau)
        mf = PowerLawSlabCertified(n_exp=n_exp, N=1600)
        Hf, _ = mf.prox_step(np.interp(mf.x, mc.x, g), hb, tau, tol=1e-14)
        out[str(n_exp)] = errc / mf.norm_L2(np.interp(mf.x, mc.x, Hc) - Hf)
    return out


# ------------------------------------------------------------------------ Bean
def bean_case(args):
    """Bean non aligne (sinus), certificat BEN ; solveur QP ou nodal."""
    Nb, spb, solver = args
    m = BeanSlabCertified(N=Nb)
    seq, i0 = sine_schedule(m, BETA, nsteps_per_branch=spb)
    t0 = time.process_time()
    if solver == "qp":
        hist = _run_schedule(m, seq, mode="A", check_oracle=False)
        profiles = [hist["profiles"][k] for k in range(len(seq))]
    else:
        profiles = bean_nodal_chain(m, seq)
    t_solve = time.process_time() - t0
    t0 = time.process_time()
    eta, eps = ben_bean_certificate(m, profiles)
    t_cert = time.process_time() - t0
    Q, err = bean_certified_loss_ben(m, seq, profiles, eps, i0)
    QB = float(bean_Q_per_cycle(BETA, m.Hp))
    out = dict(N=Nb, spb=spb, solver=solver, Q=Q, err=err, QB=QB,
               contains=bool(abs(Q - QB) <= err), width=2 * err / QB,
               eps_T=float(eps[-1]), t_solve=t_solve, t_cert=t_cert)
    if solver == "qp":
        # ancien mode B (v1) sur la meme configuration : consistance par pas
        hB = _run_schedule(m, seq, mode="B", check_oracle=False)
        QBv1, errBv1, _, _ = certified_cycle_loss(m, hB, i0)
        out.update(Q_v1B=QBv1, err_v1B=errBv1)
    return out


def main():
    t_start = time.time()
    res = {"params": dict(beta=BETA, freq=FREQ, N=N, S=S, n_cycles=NCYC)}
    Hp = PowerLawSlabCertified().Hp
    res["Q_Bean"] = float(bean_Q_per_cycle(BETA, Hp))
    bean_cfg = [(100, 80, "qp"), (200, 160, "qp"), (400, 320, "qp"),
                (800, 640, "qp"), (100, 80, "nodal"), (200, 160, "nodal"),
                (400, 320, "nodal"), (800, 640, "nodal"), (1600, 1280, "nodal"),
                (3200, 2560, "nodal")]
    ts_cfg = [(n, N, s) for n in (20, 80) for s in (200, 400, 800, 1600)] \
        + [(n, 800, 200) for n in NS]
    with ProcessPoolExecutor(max_workers=8) as ex:
        f_pl = [ex.submit(powerlaw_case, n) for n in NS]
        f_bean = [ex.submit(bean_case, c) for c in bean_cfg]
        f_ts = [ex.submit(timestep_study, c) for c in ts_cfg]
        f_cost = ex.submit(cost_v2_gap)
        f_ref = ex.submit(reference_check)
        f_jc = ex.submit(jc_time_dependent)
        res["powerlaw"] = [f.result() for f in f_pl]
        res["bean_ben"] = [f.result() for f in f_bean]
        res["timestep"] = [f.result() for f in f_ts]
        res["cost_v2"] = f_cost.result()
        res["ref_check"] = f_ref.result()
        res["jc_t"] = f_jc.result()
    res["wall_time_s"] = time.time() - t_start
    res["step_effectivity"] = step_effectivities()
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print("ecrit :", OUT, f"({res['wall_time_s']:.0f} s)")
    make_figures(res)


# --------------------------------------------------------------------- figures
def make_figures(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    C_ANA, C_NUM, C_B = "#d62728", "#1f77b4", "#7f7f7f"

    # Bean non aligne : BEN vs ancien mode B
    bb = res["bean_ben"]
    QB = res["Q_Bean"]
    qp = [r for r in bb if r["solver"] == "qp"]
    nod = [r for r in bb if r["solver"] == "nodal"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.3))
    ax1.loglog([r["N"] for r in qp], [100 * 2 * r["err_v1B"] / QB for r in qp],
               "s--", color=C_B, label="step-wise consistency bound (original mode B)")
    ax1.loglog([r["N"] for r in qp], [100 * r["width"] for r in qp], "o-",
               color=C_NUM, label="Theorem 1, QP projections")
    ax1.loglog([r["N"] for r in nod], [100 * r["width"] for r in nod], "^:",
               color="#2ca02c", label="Theorem 1, nodal catch-up")
    Ns = np.array([r["N"] for r in nod], float)
    ax1.loglog(Ns, 100 * nod[0]["width"] * Ns[0] / Ns, "k-", lw=0.6, alpha=0.5,
               label="slope -1")
    ax1.set_xlabel("elements N  (steps per branch = 0.8 N)")
    ax1.set_ylabel("relative bracket width [%]")
    ax1.set_title("Bean, non-aligned sinusoidal drive")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, which="both")
    ax2.axhline(1.0, color=C_ANA, ls="--", lw=1.4, label="Bean, analytical")
    ax2.errorbar([r["N"] for r in nod], [r["Q"] / QB for r in nod],
                 yerr=[r["err"] / QB for r in nod], fmt="o", color=C_NUM,
                 capsize=4, label="certified bracket (Theorem 1)")
    ax2.set_xscale("log")
    ax2.set_xlabel("elements N")
    ax2.set_ylabel(r"$Q\,/\,Q_{\rm Bean}$")
    ax2.set_title("Guaranteed enclosure under joint refinement")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig("fig_en_bean_ben.png", dpi=150)
    plt.close(fig)

    # effectivite sur la perte (n = 20)
    r20 = [r for r in res["powerlaw"] if r["n"] == 20][0]
    cv = r20["curves"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.2),
                                   gridspec_kw={"width_ratios": [3, 2]})
    t = np.array(cv["t"]) * 1e3
    ax1.semilogy(t, cv["eps"], "-", color=C_NUM, lw=1.6,
                 label=r"certified bound $\epsilon(t)$ (Theorem 1)")
    ax1.semilogy(t, np.maximum(cv["err"], 1e-6), "-", color=C_ANA, lw=1.0,
                 label=r"true error $\|H_v(t)-u(t)\|$ (reference)")
    for c in (1, 2, 3):
        ax1.axvline(20.0 * c, color="k", ls=":", lw=0.6, alpha=0.5)
    ax1.set_xlabel("t [ms]")
    ax1.set_ylabel("L2 error [A m$^{-1/2}$]")
    ax1.set_title("n = 20: bound versus true error")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, which="both")
    labels = ["A\nfield\neffectivity", "B\nCauchy-\nSchwarz", "C\nno sign\ncancellation"]
    vals = [r20["fac_A"], r20["fac_B"], r20["fac_C"]]
    ax2.bar(labels, vals, color=[C_NUM, "#ff7f0e", "#2ca02c"], width=0.55)
    for i, v in enumerate(vals):
        ax2.text(i, v * 1.08, f"{v:.1f}", ha="center", fontsize=9)
    ax2.set_yscale("log")
    ax2.set_ylim(1, max(vals) * 3)
    ax2.set_title(f"loss effectivity {r20['fac_I_Q']:.0f} = A x B x C")
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig("fig_en_loss_effectivity.png", dpi=150)
    plt.close(fig)
    print("figures : fig_en_bean_ben.png fig_en_loss_effectivity.png")


if __name__ == "__main__":
    import sys
    if "--extra" in sys.argv:          # complements fusionnes dans le JSON
        with open(OUT) as fh:
            res = json.load(fh)
        if "step_effectivity" not in res:
            res["step_effectivity"] = step_effectivities()
        if "E_v2_T" not in res["cost_v2"]:
            res["cost_v2"] = cost_v2_gap()
        with open(OUT, "w") as fh:
            json.dump(res, fh, indent=1)
        print(res["step_effectivity"])
    elif "--figures" in sys.argv:
        with open(OUT) as fh:
            make_figures(json.load(fh))
    else:
        main()
