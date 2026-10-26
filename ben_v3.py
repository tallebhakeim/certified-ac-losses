# -*- coding: utf-8 -*-
"""
PoC v3 : certificat espace-temps de Brezis-Ekeland-Nayroles (BEN).

THEOREME 1 (majorant BEN, forme Fenchel-Young pure). Soit u la solution de
    mu0 dH/dt = d/dx phi(dH/dx),  H(+-a,t) = H_a(t),  H(.,0) = H0,
avec H_a absolument continue et phi = Phi' (Phi convexe). Soit H_v toute
trajectoire avec CL et CI EXACTES, et s(x,t) tout champ verifiant
    ds/dx = mu0 dH_v/dt   (Faraday, a une constante c(t) libre pres).
Alors, pour tout t' dans [0,T] :

    mu0/2 || H_v(t') - u(t') ||_L2^2
        <=  Int_0^t' Int_Omega [ Phi(dH_v/dx) + Phi*(s) - s dH_v/dx ] dx dt.

Preuve (10 lignes, cf. README) : lifting w = H - H_a(t) (spatialement
constant, possible car les deux bords portent la meme donnee), principe de
Brezis-Ekeland-Nayroles avec forcage f = -mu0 dH_a/dt, representation en
flux de Psi*, et telescopage integral qui reduit tout au residu de
Fenchel-Young. L'integrande est >= 0 ponctuellement, d'ou la validite a
tout instant intermediaire.

CONSEQUENCES IMPLEMENTEES ICI :
- La trajectoire est l'interpolee LINEAIRE en temps de la chaine prox v2,
  et le probleme de reference est celui a excitation H_a affine par
  morceaux (les noeuds de la sinusoide) : CL de H_v exacte sans correction.
- s par slab de temps = c_k + (mu0/tau) Int (H^k - H^{k-1}) : le flux
  equilibre de la v2, constant en temps sur le slab.
- Quadratures a surestimation garantie : Phi(J(t)) trapeze en temps
  (J lineaire en t, Phi convexe), Phi*(s) corde + garde de courbure en
  espace (v2), terme croise s.J EXACT (Simpson en x, valeur moyenne en t).
- eps(t_k) = sqrt(2/mu0 * cumul) : borne d'erreur L2 CONTINUE en temps,
  remplace l'accumulation en sqrt(S) des gaps par pas de la v2 ET couvre
  l'erreur de schema en temps (la comparaison est face au continuum).
- QoI : l'excitation etant affine par morceaux, l'integrale de boucle de
  la trajectoire est EXACTE (trapeze = forme fermee), et l'erreur certifiee
  est mu0 sum |dHa_k| eps_k / sqrt(2a). Bracket entierement certifie face a
  la solution exacte du continuum (a excitation affine par morceaux).
"""

import numpy as np
from scipy.optimize import minimize_scalar

from powerlaw_slab_certified import MU0


def ben_certificate(model, hist, nsub=8, Jc_seq=None):
    """Certificat BEN de la chaine prox : renvoie (slabs, eps_nodes).

    Jc_seq (optionnel, revision R1.3) : Jc constant par slab de temps, Jc(t)
    prescrit ; le Theoreme 1 vaut pour Phi(t, .) convexe a chaque instant.

    slabs[k] = surestimation garantie de Int_slab_k Int FY dx dt >= 0 ;
    eps_nodes[k] = sqrt(2/mu0 * sum_{j<=k} slabs[j]) >= ||H_v(t_k) - u(t_k)||_L2.
    """
    h, N = model.h, model.N
    tau = hist["tau"]
    coef = MU0 / tau
    xi = np.linspace(0.0, 1.0, nsub + 1)[None, :]
    hsub = h / nsub
    K = len(hist["Ha"])
    H_prev = np.zeros(N + 1)
    slabs = np.empty(K)
    Jc_save = model.Jc
    for k in range(K):
        if Jc_seq is not None:
            model.Jc = float(Jc_seq[k])
        Hk = hist["profiles"][k]
        e = Hk - H_prev
        inc = 0.5 * h * (e[:-1] + e[1:])
        S0 = np.concatenate([[0.0], np.cumsum(inc)])
        J0 = np.diff(H_prev) / h                      # J au debut du slab
        J1 = np.diff(Hk) / h                          # J a la fin du slab
        Jmid = 0.5 * (J0 + J1)
        # Int Int Phi EXACTE : J lineaire en t, primitive analytique de Phi.
        # A(s) = Ec Jc^2 sign(s) |s/Jc|^(n+2) / ((n+1)(n+2)),  A' = Phi.
        nn = model.n
        CA = model.Ec * model.Jc**2 / ((nn + 1.0) * (nn + 2.0))
        A0 = CA * np.sign(J0) * (np.abs(J0) / model.Jc)**(nn + 2.0)
        A1 = CA * np.sign(J1) * (np.abs(J1) / model.Jc)**(nn + 2.0)
        dJ = J1 - J0
        small = np.abs(dJ) < 1e-12 * (np.abs(J0) + np.abs(J1) + 1e-30)
        mean_Phi = np.where(small, 0.5 * (model.Phi(J0) + model.Phi(J1)),
                            (A1 - A0) / np.where(small, 1.0, dJ))
        int_Phi = tau * h * float(np.sum(mean_Phi))
        # s = c + sigma0(x), constant en temps sur le slab (P2 en x)
        eL, eR = e[:-1][:, None], e[1:][:, None]
        sig0 = coef * (S0[:-1][:, None] + h * (eL * xi + 0.5 * (eR - eL) * xi**2))
        sig0_mid = coef * (S0[:-1] + h * (0.5 * e[:-1] + 0.125 * (e[1:] - e[:-1])))
        guard = (0.125 * coef * np.abs(e[1:] - e[:-1]) / h * hsub**2)[:, None]

        def slab_of_c(c):
            sig = sig0 + c
            int_sig = h * (sig[:, 0] + 4.0 * (sig0_mid + c) + sig[:, -1]) / 6.0
            av = np.abs(sig) + guard
            int_phis = hsub * (0.5 * model.Phi_star(av[:, 0])
                               + 0.5 * model.Phi_star(av[:, -1])
                               + np.sum(model.Phi_star(av[:, 1:-1]), axis=1))
            cross = float(np.sum(Jmid * int_sig))
            return int_Phi + tau * float(np.sum(int_phis)) - tau * cross

        phi_el = model.phi(Jmid)
        m_sig0 = coef * (S0[:-1] + 0.5 * inc)
        c0 = float(np.mean(phi_el - m_sig0))
        span = 10.0 * model.Ec + 0.5 * abs(c0) + 1e3 * float(np.std(phi_el))
        res = minimize_scalar(slab_of_c, bounds=(c0 - span, c0 + span),
                              method="bounded",
                              options={"xatol": 1e-5 * model.Ec, "maxiter": 60})
        c_opt = float(res.x) if np.isfinite(res.x) else c0
        slabs[k] = max(min(slab_of_c(c_opt), slab_of_c(c0)), 0.0)
        H_prev = Hk
    model.Jc = Jc_save
    eps_nodes = np.sqrt(2.0 / MU0 * np.cumsum(slabs))
    return slabs, eps_nodes


def certified_loss_v3(model, hist, eps_nodes):
    """Bracket ENTIEREMENT certifie des pertes du dernier cycle.

    L'excitation etant affine par morceaux, la boucle de la trajectoire
    Q = mu0 OINT (H_a - <H_v>) dH_a est EXACTE par la regle du trapeze
    (integrande lineaire par slab). L'erreur face a la solution exacte du
    continuum est bornee par mu0 sum |dHa| eps(t_k) / sqrt(2a), avec eps
    la borne BEN, valable a tout instant (integrande FY >= 0).
    """
    S = hist["steps_per_cycle"]
    Ha = hist["Ha"][-S - 1:]
    mH = np.asarray(hist["meanH"])[-S - 1:]
    eps = np.concatenate([[eps_nodes[-S - 1]], eps_nodes[-S:]])
    L = 2.0 * model.a
    f = Ha - mH
    dHa = np.diff(Ha)
    Q = MU0 * float(np.sum(0.5 * (f[:-1] + f[1:]) * dHa))
    # eps est croissante : eps(t) <= eps(t_k) sur le slab k
    err = MU0 * float(np.sum(np.abs(dHa) * eps[1:])) / np.sqrt(L)
    return Q, err
