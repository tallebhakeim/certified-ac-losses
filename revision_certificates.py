# -*- coding: utf-8 -*-
"""
Revision SuST (SUST-107877, R1/R2) : complements certifies du Theoreme 1.

Ce module ajoute, sans toucher aux certificats des sections 2 a 4 :

1) BEN POUR LE MODELE DE BEAN SUR MAILLAGE NON ALIGNE (R2-Q1).
   Le Theoreme 1 vaut pour Phi = indicatrice de {|p| <= Jc}, de conjuguee
   Phi*(s) = Jc |s|. Pour toute chaine admissible de rattrapage (|H'| <= Jc
   sur chaque profil, CL et CI exactes), l'interpolee lineaire en temps
   reste admissible (convexite), Phi(H_v') = 0, et le residu de Fenchel-Young
   par slab vaut
       eta_k = mu0 [ Jc Int |c + S_k| dx - Int (c + S_k) p_k dx ],
       S_k(x) = Int_{-a}^x (H^k - H^{k-1}) dy,  p_k = (H^k' + H^{k-1}')/2,
   independant du pas de temps (rate independence). La borne compare
   DIRECTEMENT a la solution continue en espace : plus aucun terme de
   consistance spatiale (le mode B de la v1 devient inutile). Comme Bean est
   rate-independent, la solution a excitation affine par morceaux coincide
   aux noeuds avec celle a excitation sinusoidale (reparametrage monotone
   du temps sur chaque branche, les extrema etant des noeuds) : le bracket
   certifie la VRAIE sinusoide.

2) STABILITE PAR RAPPORT A LA DONNEE DE BORD (R2-Q2), loi de puissance.
   Lemme 2 (comparaison) : si |H_a^s(t) - H_a^p(t)| <= D pour t <= t', et
   memes donnees initiales, alors |u_s - u_p| <= D partout sur [0, t'].
   Preuve : u_p + D est solution (seules les derivees interviennent), on
   teste la difference des equations par (u_s - u_p - D)^+, de trace nulle,
   et la monotonie de phi donne d/dt ||(.)^+||^2 <= 0. Pour l'interpolee
   affine d'une sinusoide, D = Hm (omega tau)^2 / 8. D'ou le bracket de la
   VRAIE sinusoide :
       |Q_s - Qhat_s| <= mu0 Sum_k |dHa_k| ( eps_{k+1} / sqrt(2a) + D ),
   avec Qhat_s = -mu0 Int <H_v> dHa_s/dt dt en forme fermee.

3) PERIODICITE (R2-Q2). Deux solutions de meme donnee de bord se
   contractent en L2 ; u(. + T) et u ayant la meme donnee periodique,
   d(t) = ||u(t + T) - u(t)|| est NON CROISSANTE, et certifiee par
   ||H_v(t + T) - H_v(t)|| +/- (eps(t + T) + eps(t)).

4) EFFECTIVITE SUR LA PERTE ET DECOMPOSITION DU CONSERVATISME (R2-Q3).
   Effectivite I_Q = borne / |Q_h - Q_ref| = A x B x C, avec
   A = effectivite du majorant sur le champ (ponderee par |dHa|),
   B = perte de Cauchy-Schwarz ||e|| / (sqrt(2a) |<e>|),
   C = absence de compensation des signes de <e> le long du cycle.
"""

import numpy as np
from scipy.optimize import minimize_scalar

from bean_slab_certified import MU0
from powerlaw_slab_certified import PowerLawSlabCertified


# ----------------------------------------------------------------------
# 1) Bean : chaines admissibles et certificat BEN
# ----------------------------------------------------------------------

def bean_nodal_chain(model, Ha_seq):
    """Rattrapage NODAL : ecretage conique des valeurs nodales.

    max/min de deux suites Jc-lipschitziennes aux noeuds : profil P1
    admissible (|H_{i+1} - H_i| <= Jc h). Solveur sans QP, sert a montrer
    que le certificat BEN ne depend pas du solveur.
    """
    d = np.minimum(model.x + model.a, model.a - model.x)
    H = np.zeros(model.N + 1)
    hb_old = 0.0
    profiles = []
    for hb in Ha_seq:
        if hb >= hb_old:
            H = np.maximum(H, hb - model.Jc * d)
        else:
            H = np.minimum(H, hb + model.Jc * d)
        H[0] = H[-1] = hb
        profiles.append(H.copy())
        hb_old = hb
    return profiles


def ben_bean_certificate(model, profiles, nsub=8):
    """Residus BEN par slab (eta_k >= 0) pour une chaine de Bean admissible.

    Verifie l'admissibilite (|H'| <= Jc), puis integre Jc |s| par une
    quadrature a surestimation garantie (corde + garde de courbure, comme en
    v2 : s est P2 par element). Retourne (eta, eps_nodes).
    """
    h, N = model.h, model.N
    xi = np.linspace(0.0, 1.0, nsub + 1)[None, :]
    hsub = h / nsub
    H_prev = np.zeros(N + 1)
    eta = np.empty(len(profiles))
    for k, Hk in enumerate(profiles):
        if np.max(np.abs(np.diff(Hk))) > model.Jc * h * (1.0 + 1e-12):
            raise ValueError("chaine non admissible au pas %d" % k)
        e = Hk - H_prev
        inc = 0.5 * h * (e[:-1] + e[1:])
        S0 = np.concatenate([[0.0], np.cumsum(inc)])
        p = 0.5 * (np.diff(H_prev) + np.diff(Hk)) / h
        eL, eR = e[:-1][:, None], e[1:][:, None]
        s0 = S0[:-1][:, None] + h * (eL * xi + 0.5 * (eR - eL) * xi**2)
        s0_mid = S0[:-1] + h * (0.5 * e[:-1] + 0.125 * (e[1:] - e[:-1]))
        guard = (0.125 * np.abs(e[1:] - e[:-1]) / h * hsub**2)[:, None]

        def eta_of_c(c):
            s = s0 + c
            int_s = h * (s[:, 0] + 4.0 * (s0_mid + c) + s[:, -1]) / 6.0
            av = np.abs(s) + guard
            int_abs = hsub * (0.5 * av[:, 0] + 0.5 * av[:, -1]
                              + np.sum(av[:, 1:-1], axis=1))
            return MU0 * (model.Jc * float(np.sum(int_abs)) - float(p @ int_s))

        lo, hi = -float(np.max(S0)), -float(np.min(S0))
        if hi - lo < 1e-300:
            eta[k] = max(eta_of_c(lo), 0.0)
        else:
            res = minimize_scalar(eta_of_c, bounds=(lo, hi), method="bounded",
                                  options={"xatol": 1e-10 * (hi - lo)})
            eta[k] = max(min(eta_of_c(res.x), eta_of_c(0.5 * (lo + hi))), 0.0)
        H_prev = Hk
    return eta, np.sqrt(2.0 / MU0 * np.cumsum(eta))


def bean_certified_loss_ben(model, Ha_seq, profiles, eps, i0):
    """Bracket BEN des pertes du cycle (indices >= i0 - 1) pour Bean.

    La boucle de la trajectoire est exacte (trapeze), l'erreur face a la
    solution continue est mu0 Sum |dHa_k| eps_{k+1} / sqrt(2a) (eps croissante).
    """
    Ha = np.asarray(Ha_seq)[i0 - 1:]
    mH = np.array([model.meanH(P) for P in profiles])[i0 - 1:]
    ep = np.asarray(eps)[i0 - 1:]
    f = Ha - mH
    dHa = np.diff(Ha)
    Q = MU0 * float(np.sum(0.5 * (f[:-1] + f[1:]) * dHa))
    err = MU0 * float(np.sum(np.abs(dHa) * ep[1:])) / np.sqrt(2.0 * model.a)
    return Q, err


# ----------------------------------------------------------------------
# 2) Loi de puissance : sinusoide exacte via le Lemme 2
# ----------------------------------------------------------------------

def sine_defect(Hm, freq, tau):
    """Majorant D de |sinus - interpolee affine| (sup en temps)."""
    w = 2.0 * np.pi * freq
    return Hm * (w * tau)**2 / 8.0


def cycle_slice(hist, cycle):
    """Indices de noeuds (debut inclus) du cycle `cycle` (1, 2, 3...)."""
    S = hist["steps_per_cycle"]
    # noeud j <-> t = (j + 1) tau ; le cycle c couvre t in [(c-1)T, cT]
    start = (cycle - 1) * S - 1          # -1 : etat initial (t = 0)
    return start, start + S


def _nodes(hist, arr, start, stop, init):
    """Valeurs aux noeuds start..stop, l'indice -1 designant t = 0."""
    a = np.asarray(arr)
    if start < 0:
        return np.concatenate([[init], a[:stop + 1]])
    return a[start:stop + 1]


def certified_loss_cycle(model, hist, eps_nodes, cycle, beta, freq=50.0):
    """Brackets certifies du cycle `cycle` : excitation affine ET sinusoide.

    Retourne dict(Q_p, err_p, Qhat_s, err_s, D).
    """
    start, stop = cycle_slice(hist, cycle)
    Ha = _nodes(hist, hist["Ha"], start, stop, 0.0)
    mH = _nodes(hist, hist["meanH"], start, stop, 0.0)
    ep = _nodes(hist, eps_nodes, start, stop, 0.0)
    t = _nodes(hist, hist["t"], start, stop, 0.0)
    L = 2.0 * model.a
    dHa = np.diff(Ha)
    f = Ha - mH
    Q_p = MU0 * float(np.sum(0.5 * (f[:-1] + f[1:]) * dHa))
    err_p = MU0 * float(np.sum(np.abs(dHa) * ep[1:])) / np.sqrt(L)
    # sinusoide exacte : Qhat_s = -mu0 Int <H_v>(t) Hm w cos(w t) dt, <H_v>
    # affine par slab (forme fermee, integration par parties)
    Hm = beta * model.Hp
    w = 2.0 * np.pi * freq
    tau = hist["tau"]
    t0, t1 = t[:-1], t[1:]
    m0, m1 = mH[:-1], mH[1:]
    I = Hm * (m1 * np.sin(w * t1) - m0 * np.sin(w * t0)) \
        + Hm * (m1 - m0) / (tau * w) * (np.cos(w * t1) - np.cos(w * t0))
    Qhat_s = -MU0 * float(np.sum(I))
    D = sine_defect(Hm, freq, tau)
    err_s = err_p + MU0 * D * float(np.sum(np.abs(dHa)))
    return dict(Q_p=Q_p, err_p=err_p, Qhat_s=Qhat_s, err_s=err_s, D=D)


# ----------------------------------------------------------------------
# 3) Periodicite certifiee
# ----------------------------------------------------------------------

def certified_periodicity(model, hist, eps_nodes):
    """Encadrement de d(cT) = ||u(cT) - u((c-1)T)||_L2, c = 2, 3, ..."""
    S = hist["steps_per_cycle"]
    K = len(hist["Ha"])
    out = []
    zero = np.zeros(model.N + 1)
    for c in range(2, K // S + 1):
        j1, j0 = c * S - 1, (c - 1) * S - 1
        H1 = hist["profiles"][j1]
        H0 = hist["profiles"][j0] if j0 >= 0 else zero
        e0 = eps_nodes[j0] if j0 >= 0 else 0.0
        dv = model.norm_L2(H1 - H0)
        slack = eps_nodes[j1] + e0
        out.append(dict(cycle=c, d_traj=dv, d_lo=max(dv - slack, 0.0),
                        d_up=dv + slack))
    return out


# ----------------------------------------------------------------------
# 4) Reference fine (excitation affine) et effectivite sur la perte
# ----------------------------------------------------------------------

def pwl_reference_states(n_exp, N_f, Ha_nodes, tau, nsub_t, Jc_seq=None,
                         Jc0=1.0e8):
    """Chaine prox fine sur le MEME probleme (excitation affine par morceaux,
    Jc constant par slab grossier si Jc_seq), etats aux noeuds grossiers."""
    m = PowerLawSlabCertified(n_exp=n_exp, N=N_f, Jc=Jc0)
    H = np.zeros(m.N + 1)
    tau_f = tau / nsub_t
    Ha_prev = 0.0
    states = []
    for k, hb in enumerate(Ha_nodes):
        if Jc_seq is not None:
            m.Jc = float(Jc_seq[k])
        for j in range(1, nsub_t + 1):
            hbj = Ha_prev + (hb - Ha_prev) * j / nsub_t
            H, _ = m.prox_step(H, hbj, tau_f, tol=1e-12)
        states.append(H.copy())
        Ha_prev = hb
    m.Jc = Jc0
    return m, states


def loss_from_states(model, Ha_nodes, states, S):
    """Boucle du dernier cycle (trapeze) a partir d'etats nodaux."""
    Ha = np.asarray(Ha_nodes)[-S - 1:]
    mH = np.array([model.meanH(P) for P in states])[-S - 1:]
    f = Ha - mH
    return MU0 * float(np.sum(0.5 * (f[:-1] + f[1:]) * np.diff(Ha)))


def loss_effectivity_factors(model, hist, eps_nodes, mref, ref_states):
    """Effectivite sur la perte et decomposition A x B x C (dernier cycle)."""
    S = hist["steps_per_cycle"]
    L = 2.0 * model.a
    Ha = np.asarray(hist["Ha"])[-S - 1:]
    dHa = np.diff(Ha)
    idx = range(len(hist["Ha"]) - S, len(hist["Ha"]))   # noeuds droits
    en, me = [], []
    for k in idx:
        Hc = np.interp(mref.x, model.x, hist["profiles"][k])
        e = Hc - ref_states[k]
        en.append(mref.norm_L2(e))
        me.append(mref.meanH(e))
    en, me = np.array(en), np.array(me)
    epsr = np.asarray(eps_nodes)[-S:]
    f = Ha - np.asarray(hist["meanH"])[-S - 1:]
    Q_h = MU0 * float(np.sum(0.5 * (f[:-1] + f[1:]) * dHa))
    Q_ref = loss_from_states(mref, hist["Ha"], ref_states, S)
    bound = MU0 * float(np.sum(np.abs(dHa) * epsr)) / np.sqrt(L)
    dQ = abs(Q_h - Q_ref)
    A = float(np.sum(np.abs(dHa) * epsr) / np.sum(np.abs(dHa) * en))
    B = float(np.sum(np.abs(dHa) * en) / (np.sqrt(L) * np.sum(np.abs(dHa) * np.abs(me))))
    C = float(MU0 * np.sum(np.abs(dHa) * np.abs(me)) / max(dQ, 1e-300))
    eff_field_T = float(epsr[-1] / en[-1])
    return dict(Q_h=Q_h, Q_ref=Q_ref, bound=bound, dQ=dQ,
                I_Q=bound / max(dQ, 1e-300), A=A, B=B, C=C,
                eff_field_T=eff_field_T, err_T=float(en[-1]),
                eps_T=float(epsr[-1]))
