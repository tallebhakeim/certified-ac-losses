# -*- coding: utf-8 -*-
"""
Pertes AC certifiees pour un supraconducteur en modele de Bean, PoC v1.

Slab infini x in [-a, a], champ applique H_a(t) parallele (axe z).
Modele de Bean (etat critique) : |J| <= Jc, formulation H de Prigozhin,
processus de rafle (sweeping process) :

    H(., t) in K(t) = { v : v(+-a) = H_a(t), |v'| <= Jc },
    -mu0 dH/dt in N_{K(t)}(H).

Schema de rattrapage de Moreau (catch-up) : H^k = Proj_{L2}( H^{k-1} -> K(t_k) ).

Trois ingredients certifies :
1) EXACTITUDE EN TEMPS (prouvee, cf. README) : sur une branche monotone de
   H_a, la projection L2 admet la forme fermee max/min contre le cone de
   bord, et le rattrapage reproduit EXACTEMENT la solution continue aux
   noeuds de temps, quel que soit le pas. La dynamique est rate-independent.
2) CERTIFICAT SPATIAL PAR DUALITE : chaque projection discrete (P1) est un
   QP resolu par montee duale ; tout multiplicateur admissible lambda >= 0
   fournit une borne inferieure duale q(lambda), d'ou l'ecart de dualite
   gap = F(H_feasible) - q(lambda) >= 0 et, par forte convexite (module 1
   en norme L2 exacte, matrice de masse P1 consistante),
       || H_feasible - H_exact ||_{L2}^2 <= 2 * gap.
   Les erreurs par pas s'ADDITIONNENT sans amplification car la projection
   sur un convexe est non expansive.
3) QoI = pertes par cycle. Identite d'energie (cycle ferme) :
       Q_v = mu0 * OINT (H_a - <H>) dH_a   [J/m^3 par cycle],
   <H> = moyenne de H sur le slab, fonction LINEAIRE de H, donc l'erreur
   certifiee L2 se propage par Cauchy-Schwarz : |<e>| <= ||e||_L2 / sqrt(2a).
   Quadrature en temps : <H>(H_a) est C0 et par branches quadratique avec
   |d2<H>/dH_a^2| <= 1/H_p (demontre pour le slab), d'ou une borne de
   trapeze explicite sum |dH_a|^3 / (12 H_p).

Verdict : l'encadrement garanti [Q_lo, Q_up] doit contenir l'analytique de
Bean : Q_v = (2/3) mu0 Hp^2 beta^3 (beta <= 1), mu0 Hp^2 (2 beta - 4/3)
(beta >= 1), avec Hp = Jc a et beta = H_m / H_p.

Modes :
- mode A "aligne"  : pas dH_a accordes au maillage (les fronts tombent sur
  les noeuds), la solution exacte est P1 sur la grille -> consistance
  spatiale NULLE (preuve : P_K(z) in K_h => P_{K_h}(z) = P_K(z)).
- mode B "generique" : H_a(t) sinusoidal echantillonne librement, terme de
  consistance spatiale certifie ajoute a chaque pas (borne d'interpolation
  des fonctions Jc-lipschitziennes, delta = Jc h sqrt(2a) / 2).

Dependances : numpy, scipy. Auteur : PoC Talleb / GeePs, 2026.
"""

import numpy as np
from scipy.linalg import cholesky_banded, cho_solve_banded
from scipy.optimize import minimize

MU0 = 4.0e-7 * np.pi


# ----------------------------------------------------------------------
# Analytique de Bean (slab)
# ----------------------------------------------------------------------

def bean_Q_per_cycle(beta, Hp):
    """Pertes par cycle et par unite de volume [J/m^3], analytique de Bean."""
    beta = np.asarray(beta, dtype=float)
    q_low = (2.0 / 3.0) * MU0 * Hp**2 * beta**3
    q_high = MU0 * Hp**2 * (2.0 * beta - 4.0 / 3.0)
    return np.where(beta <= 1.0, q_low, q_high)


def bean_virgin_meanH(Ha, Hp):
    """<H> sur la courbe de premiere aimantation (Ha >= 0)."""
    Ha = np.asarray(Ha, dtype=float)
    return np.where(Ha <= Hp, Ha**2 / (2.0 * Hp), Ha - Hp / 2.0)


def bean_loop_meanH_down(Ha, Hm, Hp):
    """<H> sur la branche descendante Hm -> -Hm du cycle etabli."""
    Ha = np.asarray(Ha, dtype=float)
    dH = Hm - Ha
    top = bean_virgin_meanH(np.array(Hm), Hp)
    part = top - dH**2 / (4.0 * Hp)          # valable tant que dH <= 2 Hp
    full = Ha + Hp / 2.0                      # renversement complet (dH >= 2 Hp)
    return np.where(dH <= 2.0 * Hp, part, full)


def bean_profile_virgin(x, Ha, a, Jc):
    """Profil H(x) en premiere aimantation (etat vierge, Ha >= 0)."""
    d = np.minimum(x + a, a - x)              # distance au bord
    return np.maximum(Ha - Jc * d, 0.0)


def bean_profile_after_reversal(x, Hm, Ha, a, Jc):
    """Profil sur la branche descendante partant du sommet Hm (Ha <= Hm)."""
    zplus = bean_profile_virgin(x, Hm, a, Jc)     # etat au sommet
    d = np.minimum(x + a, a - x)
    return np.minimum(zplus, Ha + Jc * d)


# ----------------------------------------------------------------------
# Solveur certifie
# ----------------------------------------------------------------------

class BeanSlabCertified:
    """Slab de Bean 1D, P1 sur grille uniforme, projections certifiees."""

    def __init__(self, a=1.0e-3, Jc=1.0e8, N=200):
        self.a, self.Jc, self.N = float(a), float(Jc), int(N)
        self.h = 2.0 * a / N
        self.x = np.linspace(-a, a, N + 1)
        self.Hp = Jc * a
        self.c = Jc * self.h                  # borne sur |H_{i+1} - H_i|

        # matrice de masse P1 consistante (bande symetrique), norme L2 exacte
        h = self.h
        self.M_diag = np.full(N + 1, 2.0 * h / 3.0)
        self.M_diag[0] = self.M_diag[-1] = h / 3.0
        self.M_off = np.full(N, h / 6.0)

        # bloc libre (noeuds 1..N-1), Cholesky bande pour M_ff
        ab = np.zeros((2, N - 1))
        ab[0, 1:] = self.M_off[1:N - 1]       # sur-diagonale
        ab[1, :] = self.M_diag[1:N]
        self.Mff_chol = cholesky_banded(ab)

        # poids d'integration exacts pour P1 : integrale = regle des trapezes
        self.w_int = np.full(N + 1, h)
        self.w_int[0] = self.w_int[-1] = h / 2.0

    # ---------- algebre de base ----------

    def M_dot(self, v):
        r = self.M_diag * v
        r[:-1] += self.M_off * v[1:]
        r[1:] += self.M_off * v[:-1]
        return r

    def norm_M(self, v):
        return float(np.sqrt(max(v @ self.M_dot(v), 0.0)))

    def F_primal(self, H, g):
        e = H - g
        return 0.5 * float(e @ self.M_dot(e))

    def meanH(self, H):
        return float(self.w_int @ H) / (2.0 * self.a)

    # ---------- oracle exact (forme fermee du rattrapage, cf. README) ----------

    def project_exact_monotone(self, z, hb_new, hb_old):
        """Projection exacte quand la CL bouge de facon monotone.

        Montee du bord : P(z) = max(z, cone bas) ; descente : min(z, cone haut).
        Trois lignes de preuve dans le README (inequation variationnelle).
        """
        d = np.minimum(self.x + self.a, self.a - self.x)
        if hb_new >= hb_old:
            return np.maximum(z, hb_new - self.Jc * d)
        return np.minimum(z, hb_new + self.Jc * d)

    # ---------- projection QP avec certificat dual ----------

    def _H_of_nu(self, nu, g, hb):
        """Minimiseur libre du lagrangien pour multiplicateur signe nu."""
        r = np.zeros(self.N + 1)              # r = D^T nu
        r[:-1] -= nu
        r[1:] += nu
        rhs = self.M_dot(g)[1:self.N] - r[1:self.N]
        # retirer la contribution des noeuds de bord imposes a hb
        rhs[0] -= self.M_off[0] * hb
        rhs[-1] -= self.M_off[-1] * hb
        Hf = cho_solve_banded((self.Mff_chol, False), rhs)
        H = np.empty(self.N + 1)
        H[0] = H[-1] = hb
        H[1:self.N] = Hf
        return H

    def _dual_value_grad(self, lam, g, hb):
        """Valeur duale q(lambda) et gradient, lambda = (lam+, lam-) >= 0."""
        n = self.N
        nu = lam[:n] - lam[n:]
        H = self._H_of_nu(nu, g, hb)
        dH = np.diff(H)
        q = self.F_primal(H, g) + float(nu @ dH) - self.c * float(np.sum(lam))
        gplus = dH - self.c
        gminus = -dH - self.c
        return q, np.concatenate([gplus, gminus]), H

    def make_feasible(self, cand, hb):
        """Repare un candidat en un point STRICTEMENT admissible de K_h."""
        cm = self.c * (1.0 - 1.0e-12)
        v = cand.copy()
        v[0] = v[-1] = hb
        lo = v.copy()
        for i in range(1, self.N + 1):
            lo[i] = min(lo[i], lo[i - 1] + cm)
        for i in range(self.N - 1, -1, -1):
            lo[i] = min(lo[i], lo[i + 1] + cm)
        hi = v.copy()
        for i in range(1, self.N + 1):
            hi[i] = max(hi[i], hi[i - 1] - cm)
        for i in range(self.N - 1, -1, -1):
            hi[i] = max(hi[i], hi[i + 1] - cm)
        w = 0.5 * (lo + hi)
        idx = np.arange(self.N + 1)
        cone = cm * np.minimum(idx, self.N - idx)
        w = np.clip(w, hb - cone, hb + cone)
        return w

    def project_certified(self, g, hb, lam0=None, maxiter=800):
        """Projection de g sur K_h(hb) avec certificat.

        Retourne (H admissible, err = borne GARANTIE ||H - H*_h||_L2, lam).
        """
        n = self.N
        if lam0 is None:
            lam0 = np.zeros(2 * n)

        def fun(lam):
            q, grad, _ = self._dual_value_grad(lam, g, hb)
            return -q, -grad

        res = minimize(fun, lam0, jac=True, method="L-BFGS-B",
                       bounds=[(0.0, None)] * (2 * n),
                       options={"maxiter": maxiter, "ftol": 1e-18,
                                "gtol": 1e-14, "maxcor": 20})
        lam = np.maximum(res.x, 0.0)          # admissibilite duale exacte
        q, _, Hcand = self._dual_value_grad(lam, g, hb)
        H = self.make_feasible(Hcand, hb)
        gap = self.F_primal(H, g) - q
        scale = max(1.0, abs(self.F_primal(H, g)))
        gap_cert = max(gap, 0.0) + 1.0e-15 * scale   # garde d'arrondi
        err = float(np.sqrt(2.0 * gap_cert))
        return H, err, lam

    # ---------- borne de consistance spatiale (mode B) ----------

    def consistency_bound(self, dHa, n_kinks=6):
        """Borne certifiee de || P_{K_h}(z_ex) - P_K(z_ex) ||_L2 par pas.

        Argument (detaille dans le README, specifique au slab de Bean) :
        u = P_K(z_ex) est lineaire par morceaux avec au plus n_kinks ruptures
        de pente (profils = min/max de cones), donc l'erreur d'interpolation
        P1 est LOCALISEE :  ||u - I_h u||_L2^2 <= n_kinks (Jc h/2)^2 (4h/3),
        ||u - I_h u||_L1 <= n_kinks Jc h^2 / 2, et |u - z_ex|_inf <= |dHa|
        (l'ecretage conique coupe au plus la variation de la CL). La forte
        convexite avec point test I_h u donne alors
            ||P_{K_h} z - P_K z||^2 <= ||u - I_h u||^2 + 2 |dHa| ||u - I_h u||_L1.
        """
        d2 = n_kinks * (0.5 * self.Jc * self.h)**2 * (4.0 * self.h / 3.0)
        cross = abs(dHa) * n_kinks * self.Jc * self.h**2
        return float(np.sqrt(d2 + 2.0 * cross))


# ----------------------------------------------------------------------
# Pilotes de cycle
# ----------------------------------------------------------------------

def _run_schedule(model, Ha_seq, mode, check_oracle=True):
    """Deroule une sequence de valeurs de H_a et accumule les certificats.

    Retourne l'historique : Ha, <H>, borne d'erreur cumulee E_k sur ||e||_L2,
    profils, et ecart max au rattrapage exact (oracle) si demande.
    """
    H = np.zeros(model.N + 1)                 # etat vierge
    Hex = np.zeros(model.N + 1)
    hb_old = 0.0
    E = 0.0
    lam = None
    hist = {"Ha": [], "meanH": [], "E": [], "gap_err": [], "cons": [],
            "profiles": {}, "oracle_dev": 0.0, "min_margin": np.inf}
    for k, hb in enumerate(Ha_seq):
        cons = 0.0
        if mode == "B":
            cons = model.consistency_bound(hb - hb_old)
        H, gerr, lam = model.project_certified(H, hb, lam0=lam)
        E = E + gerr + cons
        margin = model.c - float(np.max(np.abs(np.diff(H))))
        hist["min_margin"] = min(hist["min_margin"], margin)
        if check_oracle:
            Hex = model.project_exact_monotone(Hex, hb, hb_old)
            dev = model.norm_M(H - Hex)
            hist["oracle_dev"] = max(hist["oracle_dev"], dev)
        hb_old = hb
        hist["Ha"].append(hb)
        hist["meanH"].append(model.meanH(H))
        hist["E"].append(E)
        hist["gap_err"].append(gerr)
        hist["cons"].append(cons)
        hist["profiles"][k] = H.copy()
    for key in ("Ha", "meanH", "E", "gap_err", "cons"):
        hist[key] = np.array(hist[key])
    return hist


def aligned_schedule(model, beta):
    """Mode A : pas accordes au maillage. Renvoie (sequence, i_debut_cycle)."""
    nv = beta * model.N / 2.0
    if abs(nv - round(nv)) > 1e-9:
        raise ValueError("mode A : beta*N/2 doit etre entier")
    nv = int(round(nv))
    Hm = beta * model.Hp
    up0 = np.linspace(model.c, Hm, nv)                     # 0 -> Hm (vierge)
    down = Hm - 2.0 * model.c * np.arange(1, nv + 1)       # Hm -> -Hm
    up = -Hm + 2.0 * model.c * np.arange(1, nv + 1)        # -Hm -> Hm
    seq = np.concatenate([up0, down, up])
    return seq, len(up0)


def sine_schedule(model, beta, nsteps_per_branch=160):
    """Mode B : H_a sinusoidal echantillonne uniformement en temps."""
    Hm = beta * model.Hp
    th_up0 = np.linspace(0, np.pi / 2, nsteps_per_branch // 2 + 1)[1:]
    th_down = np.linspace(np.pi / 2, 3 * np.pi / 2, nsteps_per_branch + 1)[1:]
    th_up = np.linspace(3 * np.pi / 2, 5 * np.pi / 2, nsteps_per_branch + 1)[1:]
    seq = Hm * np.sin(np.concatenate([th_up0, th_down, th_up]))
    return seq, len(th_up0)


def certified_cycle_loss(model, hist, i0):
    """Encadrement garanti des pertes par cycle et par volume [J/m^3].

    Cycle = tout ce qui suit l'indice i0 (descente + remontee).
    Q_v = mu0 * sum trapezes de (H_a - <H>) dH_a
    Erreur = (spatiale certifiee via E_k) + (quadrature |f''| <= 1/Hp).
    """
    Ha = hist["Ha"][i0 - 1:]
    mH = hist["meanH"][i0 - 1:]
    E = hist["E"][i0 - 1:]
    L = 2.0 * model.a
    f = Ha - mH
    dHa = np.diff(Ha)
    Q = MU0 * float(np.sum(0.5 * (f[:-1] + f[1:]) * dHa))
    eps = E / np.sqrt(L)                      # borne sur l'erreur de <H>
    err_space = MU0 * float(np.sum(np.abs(dHa) * 0.5 * (eps[:-1] + eps[1:])))
    err_quad = MU0 * float(np.sum(np.abs(dHa)**3)) / (12.0 * model.Hp)
    err = err_space + err_quad
    return Q, err, err_space, err_quad
