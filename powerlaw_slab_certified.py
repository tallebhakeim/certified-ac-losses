# -*- coding: utf-8 -*-
"""
PoC v2 : pertes AC certifiees, slab supraconducteur en LOI DE PUISSANCE.

Loi E-J : E = Ec |J/Jc|^(n-1) (J/Jc), n ~ 5..100. Formulation H en 1D :

    mu0 dH/dt = d/dx [ phi(dH/dx) ],   phi(s) = Ec |s/Jc|^(n-1) (s/Jc),
    H(+-a, t) = H_a(t).

Pas implicite = probleme de PROX strictement convexe :

    H^k = argmin  F(H) = mu0/(2 tau) ||H - H^{k-1}||_L2^2 + Int Phi(H') dx,
    Phi(s) = Ec Jc |s/Jc|^(n+1) / (n+1),

sous CL de Dirichlet. Certificat par ECART PRIMAL-DUAL (dualite convexe,
identite exacte, aucune constante) : pour tout champ dual sigma dans H^1,

    F(H) - D(sigma) = Int mu0/(2 tau) [ (H - g) - (tau/mu0) sigma' ]^2 dx
                    + Int [ Phi(H') + Phi*(sigma) - sigma H' ] dx  >= 0,

et par forte convexite du terme de masse (module mu0/tau, INDEPENDANT DE n) :

    ||H - H*_continu||_L2^2  <=  (2 tau / mu0) * gap.

FLUX EQUILIBRE 1D : on choisit sigma(x) = c + (mu0/tau) Int_{-a}^x (H - g),
qui ANNULE IDENTIQUEMENT le terme quadratique. Le gap se reduit au residu
de Fenchel-Young, integre avec une quadrature a surestimation garantie.
La constante libre c est optimisee (probleme scalaire convexe).

Accumulation : le pas prox est non expansif (resolvante d'un operateur
maximal monotone), donc E_k = E_{k-1} + sqrt(2 tau gap_k / mu0), sans
amplification, et la comparaison se fait directement avec le minimiseur
CONTINU en espace (pas de terme de consistance separee, contrairement au
mode B de la v1 : c'est l'avantage du majorant fonctionnel).

Limite n -> infini : Phi -> indicatrice de {|s| <= Jc}, Phi* -> Jc |sigma|,
et l'on doit retrouver le modele de Bean de la v1 (verifie sur Q).

Reste non certifie dans ce PoC (crux article, cf. README) : erreur de
schema en temps (Euler implicite vs EDP, demontree par Richardson ; outil
vise = principe de Brezis-Ekeland-Nayroles), periodicite, quadrature de la
QoI en temps.
"""

import numpy as np
from scipy.linalg import solve_banded, cholesky_banded, cho_solve_banded
from scipy.optimize import minimize_scalar

MU0 = 4.0e-7 * np.pi


class PowerLawSlabCertified:
    """Slab 1D en loi de puissance, pas prox certifie par ecart primal-dual."""

    def __init__(self, a=1.0e-3, Jc=1.0e8, Ec=1.0e-4, n_exp=20, N=800):
        self.a, self.Jc, self.Ec = float(a), float(Jc), float(Ec)
        self.n = float(n_exp)
        self.N = int(N)
        self.h = 2.0 * a / N
        self.x = np.linspace(-a, a, N + 1)
        self.Hp = Jc * a

        h, N = self.h, self.N
        self.M_diag = np.full(N + 1, 2.0 * h / 3.0)
        self.M_diag[0] = self.M_diag[-1] = h / 3.0
        self.M_off = np.full(N, h / 6.0)
        ab = np.zeros((2, N - 1))
        ab[0, 1:] = self.M_off[1:N - 1]
        ab[1, :] = self.M_diag[1:N]
        self.Mff_chol = cholesky_banded(ab)
        self.w_int = np.full(N + 1, h)
        self.w_int[0] = self.w_int[-1] = h / 2.0

    # ---------- loi de puissance et paire conjuguee ----------

    def Phi(self, s):
        """Densite d'energie Phi(s) = Ec Jc |s/Jc|^(n+1) / (n+1)."""
        r = np.abs(np.asarray(s, dtype=float)) / self.Jc
        return self.Ec * self.Jc * r**(self.n + 1.0) / (self.n + 1.0)

    def Phi_star(self, sig):
        """Conjuguee Phi*(sig) = Ec Jc (n/(n+1)) |sig/Ec|^((n+1)/n)."""
        r = np.abs(np.asarray(sig, dtype=float)) / self.Ec
        return self.Ec * self.Jc * (self.n / (self.n + 1.0)) \
            * r**((self.n + 1.0) / self.n)

    def phi(self, s):
        """Flux phi(s) = Phi'(s) = Ec |s/Jc|^(n-1) (s/Jc)."""
        s = np.asarray(s, dtype=float)
        r = np.abs(s) / self.Jc
        return self.Ec * np.sign(s) * r**self.n

    def dphi(self, s):
        r = np.abs(np.asarray(s, dtype=float)) / self.Jc
        return (self.Ec * self.n / self.Jc) * r**(self.n - 1.0)

    # ---------- algebre ----------

    def M_dot(self, v):
        r = self.M_diag * v
        r[:-1] += self.M_off * v[1:]
        r[1:] += self.M_off * v[:-1]
        return r

    def norm_L2(self, v):
        return float(np.sqrt(max(v @ self.M_dot(v), 0.0)))

    def meanH(self, H):
        return float(self.w_int @ H) / (2.0 * self.a)

    def F_primal(self, H, g, tau):
        e = H - g
        quad = 0.5 * MU0 / tau * float(e @ self.M_dot(e))
        s = np.diff(H) / self.h
        return quad + self.h * float(np.sum(self.Phi(s)))

    # ---------- pas prox : Newton amorti globalement convergent ----------

    def _grad_free(self, H, g, tau):
        s = np.diff(H) / self.h
        grad_full = MU0 / tau * self.M_dot(H - g)
        f = self.phi(s)
        grad_full[:-1] -= f
        grad_full[1:] += f
        return grad_full[1:self.N]

    def prox_step(self, g, hb, tau, tol=1.0e-9, maxit=80):
        """Resout le pas implicite (probleme convexe), CL H(+-a) = hb.

        Newton amorti (Hessienne plafonnee, Armijo) + polish L-BFGS-B.
        Initialisation par ecretage conique (pente 2 Jc) : evite le pic de
        gradient du au saut de CL, qui ferait exploser |H'|^n a grand n.
        """
        n, N, h = self.n, self.N, self.h
        dbd = np.minimum(self.x + self.a, self.a - self.x)
        H = np.clip(g, hb - 2.0 * self.Jc * dbd, hb + 2.0 * self.Jc * dbd)
        gscale = MU0 / tau * h * max(self.Hp, abs(hb), 1.0)
        gtarget = tol * gscale
        r_cap = 10.0 ** (6.0 / max(self.n - 1.0, 1.0)) * self.Jc  # plafond Hessienne
        for it in range(maxit):
            gf = self._grad_free(H, g, tau)
            gnorm = float(np.max(np.abs(gf))) if N > 1 else 0.0
            if gnorm < gtarget:
                break
            s = np.diff(H) / h
            w = self.dphi(np.clip(s, -r_cap, r_cap)) / h
            band = np.zeros((3, N - 1))
            band[1, :] = MU0 / tau * self.M_diag[1:N] + w[:-1] + w[1:]
            band[0, 1:] = MU0 / tau * self.M_off[1:N - 1] - w[1:-1]
            band[2, :-1] = band[0, 1:]
            try:
                d = solve_banded((1, 1), band, -gf)
            except Exception:
                d = -gf
            slope = float(gf @ d)
            if not np.isfinite(slope) or slope >= 0.0:
                d = -gf
                slope = -float(gf @ gf)
            F0 = self.F_primal(H, g, tau)
            t = 1.0
            for _ in range(60):
                Ht = H.copy()
                Ht[1:N] = H[1:N] + t * d
                Ft = self.F_primal(Ht, g, tau)
                if np.isfinite(Ft) and Ft <= F0 + 1e-4 * t * slope:
                    H = Ht
                    break
                t *= 0.5
            else:
                break
        # polish si le Newton a stagne avant la cible
        gf = self._grad_free(H, g, tau)
        if N > 1 and float(np.max(np.abs(gf))) > gtarget:
            from scipy.optimize import minimize as _minimize

            def fun(Hf):
                Hfull = np.empty(N + 1)
                Hfull[0] = Hfull[-1] = hb
                Hfull[1:N] = Hf
                F = self.F_primal(Hfull, g, tau)
                gr = self._grad_free(Hfull, g, tau)
                if not np.isfinite(F):
                    return 1e300, np.zeros_like(Hf)
                return F, gr

            res = _minimize(fun, H[1:N], jac=True, method="L-BFGS-B",
                            options={"maxiter": 400, "ftol": 1e-18,
                                     "gtol": 0.1 * gtarget, "maxcor": 30})
            Hp_ = np.empty(N + 1)
            Hp_[0] = Hp_[-1] = hb
            Hp_[1:N] = res.x
            if self.F_primal(Hp_, g, tau) <= self.F_primal(H, g, tau):
                H = Hp_
        return H, it

    # ---------- certificat : flux equilibre + residu de Fenchel-Young ----------

    def certified_gap(self, H, g, hb, tau, nsub=8):
        """Ecart primal-dual GARANTI du pas prox.

        sigma(x) = c + (mu0/tau) Int (H - g) annule le terme quadratique ;
        gap = Int FY(H', sigma) surestime par sous-quadrature avec majoration
        de |sigma| par element (sigma est P2 par morceaux, sigma' est P1).
        Retourne (gap_sup, err = sqrt(2 tau gap / mu0), c_opt).
        """
        h, N = self.h, self.N
        e = H - g
        # primitives exactes de e (P1) : increments par element (trapeze exact)
        inc = 0.5 * h * (e[:-1] + e[1:])
        S0 = np.concatenate([[0.0], np.cumsum(inc)])   # Int_{-a}^{x_i} e
        s_el = np.diff(H) / h                           # H' par element (P0)
        Phi_el = self.h * self.Phi(s_el)                # Int Phi exact

        coef = MU0 / tau
        xi = np.linspace(0.0, 1.0, nsub + 1)[None, :]
        eL, eR = e[:-1][:, None], e[1:][:, None]
        # sigma0 (P2 par element) aux sous-points, exact
        sig0 = coef * (S0[:-1][:, None] + h * (eL * xi + 0.5 * (eR - eL) * xi**2))
        sig0_mid = coef * (S0[:-1] + h * (0.5 * e[:-1] + 0.125 * (e[1:] - e[:-1])))
        # garde de courbure : |sigma - corde| <= |sigma''| h_sub^2 / 8 par sous-pas
        hsub = h / nsub
        guard = (0.125 * coef * np.abs(e[1:] - e[:-1]) / h * hsub**2)[:, None]

        def gap_of_c(c):
            sig = sig0 + c
            int_sig = h * (sig[:, 0] + 4.0 * (sig0_mid + c) + sig[:, -1]) / 6.0
            # Int Phi*(sigma) <= trapeze de Phi*(|corde| + garde) par sous-pas :
            # x -> |corde(x)| + garde est convexe, Phi* convexe croissante,
            # donc la composee est convexe et le trapeze SURESTIME l'integrale.
            av = np.abs(sig) + guard
            int_phis = hsub * (0.5 * self.Phi_star(av[:, 0]) + 0.5 * self.Phi_star(av[:, -1])
                               + np.sum(self.Phi_star(av[:, 1:-1]), axis=1))
            return float(np.sum(Phi_el + int_phis - s_el * int_sig))

        # constante c initiale : moyenne de phi(H') - moyenne(sigma0)
        phi_el = self.phi(s_el)
        m_sig0 = coef * (S0[:-1] + 0.5 * inc)
        c0 = float(np.mean(phi_el - m_sig0))
        span = 10.0 * self.Ec + 0.5 * abs(c0) + 1e3 * np.std(phi_el)
        res = minimize_scalar(gap_of_c, bounds=(c0 - span, c0 + span),
                              method="bounded",
                              options={"xatol": 1e-5 * self.Ec, "maxiter": 100})
        c_opt = float(res.x) if np.isfinite(res.x) else c0
        gap = min(gap_of_c(c_opt), gap_of_c(c0))
        gap = max(gap, 0.0) + 1e-16 * max(1.0, abs(gap))
        err = float(np.sqrt(2.0 * tau * gap / MU0))
        return gap, err, c_opt

    def dual_value(self, sigma_nodes_c, H, g, hb, tau):
        """Valeur duale D(sigma) pour verification d'identite (quadrature fine).

        Ici sigma est le flux equilibre (P2 par morceaux) : D(sigma) =
        hb [sigma]_bords - Int [ tau/(2 mu0) sigma'^2 + g sigma' + Phi*(sigma) ].
        """
        h, N = self.h, self.N
        c = sigma_nodes_c
        e = H - g
        coef = MU0 / tau
        inc = 0.5 * h * (e[:-1] + e[1:])
        S0 = np.concatenate([[0.0], np.cumsum(inc)])
        sig_a = c + coef * S0[-1]
        sig_ma = c
        # sigma' = coef * e (P1) : integrales exactes
        int_sig2 = (tau / (2.0 * MU0)) * coef**2 * float(e @ self.M_dot(e))
        int_gsig = coef * float(g @ self.M_dot(e))
        # Int Phi*(sigma) par Gauss-Legendre 5 points par element (quasi exact)
        gl_x, gl_w = np.polynomial.legendre.leggauss(5)
        xi = (0.5 * (gl_x + 1.0))[None, :]
        eL, eR = e[:-1][:, None], e[1:][:, None]
        sig = c + coef * (S0[:-1][:, None] + h * (eL * xi + 0.5 * (eR - eL) * xi**2))
        tot_phis = 0.5 * h * float(np.sum(gl_w[None, :] * self.Phi_star(sig)))
        return hb * (sig_a - sig_ma) - int_sig2 - int_gsig - tot_phis


# ----------------------------------------------------------------------
# Pilote : cycles sinusoidaux, QoI certifiee (partie spatiale)
# ----------------------------------------------------------------------

def run_powerlaw_cycles(model, beta, freq=50.0, steps_per_cycle=200,
                        n_cycles=3, with_certificates=True):
    """Deroule n_cycles de H_a = Hm sin(2 pi f t) et certifie la chaine."""
    Hm = beta * model.Hp
    tau = 1.0 / (freq * steps_per_cycle)
    tgrid = tau * np.arange(1, n_cycles * steps_per_cycle + 1)
    Ha_seq = Hm * np.sin(2.0 * np.pi * freq * tgrid)
    H = np.zeros(model.N + 1)
    E = 0.0
    hist = {"t": tgrid, "Ha": Ha_seq, "meanH": [], "E": [], "gap": [],
            "profiles": {}, "newton_it": []}
    for k, hb in enumerate(Ha_seq):
        g = H
        H, it = model.prox_step(g, hb, tau)
        if with_certificates:
            gap, err, _ = model.certified_gap(H, g, hb, tau)
        else:
            gap, err = 0.0, 0.0
        E += err
        hist["meanH"].append(model.meanH(H))
        hist["E"].append(E)
        hist["gap"].append(gap)
        hist["newton_it"].append(it)
        hist["profiles"][k] = H.copy()
    for key in ("meanH", "E", "gap", "newton_it"):
        hist[key] = np.array(hist[key])
    hist["tau"] = tau
    hist["steps_per_cycle"] = steps_per_cycle
    return hist


def loss_last_cycle(model, hist):
    """QoI : pertes sur le dernier cycle + partie spatiale certifiee.

    Q_v = mu0 OINT (H_a - <H>) dH_a  [J/m^3/cycle], trapezes sur le dernier
    cycle ; err_space = propagation Cauchy-Schwarz des bornes L2 cumulees.
    """
    S = hist["steps_per_cycle"]
    Ha = hist["Ha"][-S - 1:]
    mH = hist["meanH"][-S - 1:]
    E = hist["E"][-S - 1:]
    L = 2.0 * model.a
    f = Ha - mH
    dHa = np.diff(Ha)
    Q = MU0 * float(np.sum(0.5 * (f[:-1] + f[1:]) * dHa))
    eps = E / np.sqrt(L)
    err_space = MU0 * float(np.sum(np.abs(dHa) * 0.5 * (eps[:-1] + eps[1:])))
    # derive de periodicite mesuree (etat debut vs fin de dernier cycle)
    S2 = 2 * S
    drift = np.nan
    if len(hist["Ha"]) >= S2:
        k_end = len(hist["Ha"]) - 1
        drift = model.norm_L2(hist["profiles"][k_end] - hist["profiles"][k_end - S])
    return Q, err_space, drift
