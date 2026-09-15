# Pertes AC certifiées pour supraconducteurs, PoC v1 : slab de Bean

**Objectif.** Premier encadrement à bornes GARANTIES (constantes entièrement
calculables, sans constante générique) des pertes AC par cycle d'un
supraconducteur. Cas test : slab infini en modèle de Bean (état critique,
|J| ≤ Jc), champ appliqué parallèle, verdict contre l'analytique de Bean.

**Positionnement.** La fouille bibliographique (juillet 2026, cf. mémoire
`hts-ac-loss-certified`) confirme que la case est vide, doublement :
personne n'encadre une perte (fonctionnelle quadratique) de façon garantie,
même en électrotechnique linéaire, et personne n'a de bornes garanties pour
les modèles supraconducteurs (loi de puissance ou état critique). Le
concurrent le plus proche est Wan et Laforest, SIAM J. Numer. Anal. 58(1),
2020 (a posteriori p-curl avec QoI pertes AC, mais résiduel à constantes
inconnues). L'accroche communautaire : Dular, Geuzaine et Vanderheyden
(IEEE TAS 2020) observent empiriquement que les formulations duales
h-conforme et b-conforme encadrent les pertes par côtés opposés, sans
preuve. Ce PoC est le premier pas vers la version rigoureuse.

## Modèle et méthode

Formulation H de Prigozhin : processus de rafle (sweeping process)

    H(., t) ∈ K(t) = { v : v(±a) = H_a(t), |v'| ≤ Jc },
    −μ0 ∂H/∂t ∈ N_{K(t)}(H),

résolu par le schéma de rattrapage de Moreau : H^k = Proj_L2(H^{k−1} → K(t_k)).

Trois ingrédients, chacun avec sa preuve courte :

1. **Exactitude en temps.** Sur une branche monotone de H_a, la projection
   admet la forme fermée « écrêtage conique » : montée du bord,
   P(z) = max(z, hb − Jc d∂(x)) ; descente, P(z) = min(z, hb + Jc d∂(x)).
   Preuve (3 lignes) : u = max(z, ψ) donne u − z ≥ 0 supporté là où u = ψ,
   et tout v ∈ K vérifie v ≥ ψ (lipschitzianité depuis le bord), donc
   (u − z, v − u) ≥ 0, l'inéquation variationnelle de la projection.
   Conséquence : le rattrapage est EXACT quel que soit le pas (la dynamique
   est rate-independent), la discrétisation en temps ne coûte qu'une erreur
   de quadrature sur la QoI. Ce point donne aussi un oracle indépendant qui
   contre-vérifie le solveur QP (test 3).

2. **Certificat spatial par dualité.** Chaque projection discrète (P1,
   masse consistante donc norme L2 exacte) est un QP ; tout multiplicateur
   λ = (λ+, λ−) ≥ 0 fournit une valeur duale q(λ) ≤ min, d'où
   ||H_adm − H*||²_L2 ≤ 2 (F(H_adm) − q(λ)) par forte convexité (module 1).
   Le point primal est rendu STRICTEMENT admissible par enveloppes
   lipschitziennes + écrêtage conique, la marge est vérifiée à chaque pas.
   Les erreurs par pas s'additionnent sans amplification : la projection sur
   un convexe est non expansive.

3. **QoI et propagation.** Identité d'énergie sur le cycle fermé :
   Q_v = μ0 ∮ (H_a − ⟨H⟩) dH_a en J/m³ par cycle. ⟨H⟩ est linéaire en H
   donc |⟨e⟩| ≤ ||e||_L2 / sqrt(2a) (Cauchy-Schwarz). Quadrature en temps :
   ⟨H⟩(H_a) est C¹ par branches, quadratique par morceaux, avec
   |d²⟨H⟩/dH_a²| ≤ 1/H_p (démontré branche par branche pour le slab), d'où
   la borne de trapèze Σ|ΔH_a|³/(12 H_p).

Deux modes :

- **Mode A, aligné.** Pas ΔH_a accordés au maillage (les fronts tombent sur
  les nœuds), la solution exacte est P1 sur la grille, donc P_K(z) ∈ K_h
  entraîne P_{K_h}(z) = P_K(z) : consistance spatiale NULLE. Bracket fin.
- **Mode B, générique.** H_a sinusoïdal échantillonné librement. Terme de
  consistance certifié par pas : l'exact est linéaire par morceaux avec au
  plus 6 ruptures de pente, l'erreur d'interpolation P1 est localisée, et
  |P_K z − z|_∞ ≤ |ΔH_a|, d'où
  cons² ≤ n_k (Jc h/2)² (4h/3) + 2|ΔH_a| n_k Jc h²/2.

## Résultats (verify.py : 14/14 PASS)

| β = H_m/H_p | Q numérique [J/m³] | ± err garanti | Q analytique | inclus | largeur |
|---|---|---|---|---|---|
| 0.4 | 535.83 | 2.89 | 536.17 | OUI | 1.08 % |
| 0.7 | 2872.92 | 11.36 | 2873.51 | OUI | 0.79 % |
| 1.0 | 8376.74 | 40.34 | 8377.58 | OUI | 0.96 % |
| 1.4 | 18429.84 | 69.97 | 18430.68 | OUI | 0.76 % |

Mode B (générique, β = 0.7) : le bracket garanti contient aussi
l'analytique, largeur ~±300 % à N = 400, convergence propre en O(h)
(74358 → 34804 → 16826 → 9100 J/m³ pour N = 100/200/400/800), fig. 4.

Figures : `fig1_profils.png` (profils vs analytique), `fig2_cycle.png`
(cycle d'hystérésis), `fig3_bracket.png` (LA figure : brackets garantis sur
la courbe de Bean), `fig4_convergence.png` (convergence du mode B).

## Fichiers

- `bean_slab_certified.py` : solveur (projections QP + certificats duaux,
  oracle exact, bornes de consistance, pertes certifiées).
- `verify.py` : 14 tests, verdict contre l'analytique de Bean.
- `run_poc.py` : génération des 4 figures.

Dépendances : numpy, scipy (L-BFGS-B pour la montée duale, Cholesky bande).
Paramètres physiques : a = 1 mm, Jc = 1e8 A/m², H_p = Jc a = 1e5 A/m.

## PoC v2 : loi de puissance E ~ J^n (verify_v2.py : 12/12 PASS)

Extension au modele realiste E = Ec |J/Jc|^(n-1) (J/Jc), n = 5..80,
H_a sinusoidal 50 Hz, beta = 0.7, Ec = 1e-4 V/m.

**Methode.** Chaque pas implicite est un probleme de PROX strictement
convexe. Certificat par ecart primal-dual (identite exacte, aucune
constante) :

    F(H) - D(sigma) = Int mu0/(2 tau) [(H - g) - (tau/mu0) sigma']^2
                    + Int [Phi(H') + Phi*(sigma) - sigma H']  >= 0,
    ||H - H*_continu||_L2^2 <= (2 tau/mu0) gap.

Deux choix structurants :

1. **Flux equilibre 1D** : sigma(x) = c + (mu0/tau) Int (H - g) annule
   IDENTIQUEMENT le terme quadratique ; le gap se reduit au residu de
   Fenchel-Young, integre par une quadrature a surestimation garantie
   (corde + garde de courbure, en exploitant la convexite de la composee
   x -> Phi*(|corde(x)| + garde)). La constante libre c est optimisee
   (probleme scalaire convexe). Contrairement au mode B de la v1, la borne
   compare directement au minimiseur CONTINU en espace : pas de terme de
   consistance separee. C'est le majorant fonctionnel en action.
2. **Robustesse en n STRUCTURELLE** : le module de forte convexite est
   mu0/tau, porte par le terme de masse, donc INDEPENDANT de n. La
   degenerescence p -> 1 qui tue les certificats elliptiques stationnaires
   est evitee par la structure prox du probleme d'evolution. Mesure :
   effectivite (borne/erreur reelle, reference maillage 8x) entre 4.3 et
   8.9 sur TOUT n = 5..80, sans tendance de degradation (figv2_robustesse).

**Resultats** (Q_Bean(0.7) = 2873.5 J/m^3) :

| n | Q [J/m^3] | effectivite du certificat par pas |
|---|---|---|
| 5 | 2248.8 | 8.9 |
| 10 | 2718.5 | 4.9 |
| 20 | 2911.2 | 4.3 |
| 40 | 2954.7 | 5.9 |
| 80 | 2944.6 | 5.1 |

- **Limite de Bean** : Q(n) est une suite de Cauchy (ecarts successifs
  470 > 193 > 44 > 10 J/m^3), les cycles convergent vers le cycle de Bean
  (figv2_cycles), et Q(80) est a 2.5 % de Q_Bean. Ce +2.5 % est PHYSIQUE
  (fluage a n fini) : il est stable a 0.2 % pres sous raffinement N=400/800
  et S=200/400. Le solveur certifie resout la physique que Bean ecrase.
- **Physique du fluage** : depassement J > Jc pres des bords, decroissant
  avec n (J_max/Jc = 1.35 a n=10, 1.03 a n=80), figv2_profils_J.
- **Solveur** : Newton amorti (Hessienne plafonnee, initialisation par
  ecretage conique qui evite l'explosion |H'|^n au saut de CL) + polish
  L-BFGS-B ; residus ~ 1e-15, converge jusqu'a n = 80 sans reglage.
- Bracket spatial certifie du run complet : ~+/-50 % (N=800), domine par
  l'accumulation en sqrt(S) des gaps par pas ; erreur en temps demontree
  O(tau) par Richardson ; quasi-periodicite mesuree (derive 2e-3 a n=80).

Fichiers v2 : `powerlaw_slab_certified.py`, `verify_v2.py`, `run_v2.py`,
figures `figv2_*.png`.

## PoC v3 : Théorème BEN et certificat espace-temps (verify_v3.py : 8/8 PASS)

La v3 ferme le verrou principal des v1/v2 : l'erreur de schéma en temps
n'était pas certifiée, et l'accumulation par pas croissait comme le nombre
de pas. Le remède est un majorant espace-temps global issu du principe de
Brezis-Ekeland-Nayroles (BEN).

**Théorème 1 (majorant BEN, forme Fenchel-Young pure).**
Soit u la solution de mu0 dH/dt = d/dx phi(dH/dx) sur Omega = (-a, a),
avec H(±a, t) = H_a(t) (H_a absolument continue), H(., 0) = H0, et
phi = Phi' pour Phi convexe. Soit H_v TOUTE trajectoire avec conditions aux
limites et initiale exactes, et s(x, t) tout champ vérifiant la loi de
Faraday ds/dx = mu0 dH_v/dt (à une constante c(t) libre près). Alors, pour
tout t' dans [0, T] :

    mu0/2 || H_v(t') - u(t') ||²_L2
       <=  Int_0^t' Int_Omega [ Phi(dH_v/dx) + Phi*(s) - s dH_v/dx ] dx dt.

*Preuve.* (i) Lifting spatialement constant w = H - H_a(t), licite car les
deux bords portent la même donnée : mu0 w' + dPsi(w) contient f = -mu0 H_a'
avec Psi(w) = Int Phi(w') et w à traces nulles. (ii) Pour v = H_v - H_a et
e = v - w : les inégalités de Fenchel-Young
Psi(v) >= Psi(w) + <f - mu0 w', v - w> (car f - mu0 w' est dans dPsi(w)) et
Psi*(f - mu0 v') >= <f - mu0 v', w> - Psi(w), sommées puis intégrées en
temps, donnent, après télescopage des termes de bord,
I(v) := Int [Psi(v) + Psi*(f - mu0 v') - <f, v>] dt
      + mu0/2 (||v(t')||² - ||v(0)||²)  >=  mu0/2 ||e(t')||².
(iii) Représentation en flux de Psi* en 1D : pour zeta = f - mu0 v', tout
s avec ds/dx = -zeta = mu0 dH_v/dt vérifie Psi*(zeta) <= Int Phi*(s) dx
(intégration par parties, traces nulles, constante c(t) libre). (iv) En
réinjectant, le terme <zeta, v> = Int s dv/dx dx fait apparaître exactement
le résidu de Fenchel-Young et le terme -mu0 <v', v> compense les termes de
bord en temps : I(v) <= Int Int FY dx dt. L'intégrande FY est >= 0
ponctuellement, d'où la validité à tout instant intermédiaire. QED.

Lecture physique : s est le champ électrique reconstruit par la loi de
Faraday (exacte), et le majorant mesure la violation de la loi de
comportement E-J le long de la trajectoire. Prager-Synge parabolique au
sens le plus propre : Faraday exacte d'un côté, constitutive relâchée de
l'autre, l'écart borne l'erreur.

**Implémentation** (`ben_v3.py`). La trajectoire est l'interpolée linéaire
en temps de la chaîne prox v2 ; le problème certifié est celui à excitation
AFFINE PAR MORCEAUX (les nœuds de la sinusoïde), ce qui rend les CL de la
trajectoire exactes sans correction ; pour le verdict Bean c'est sans perte
(Q de Bean ne dépend que de H_m, pas de la forme d'onde). Par slab de
temps : s = c_k + (mu0/tau) Int (H^k - H^{k-1}) (le flux équilibré v2,
constant en temps), Int Int Phi en FORME FERMÉE (J linéaire en t, primitive
analytique), Phi* par corde + garde de courbure (v2), terme croisé exact.
Et comme l'excitation est affine par morceaux, l'intégrale de boucle de la
QoI est EXACTE (le trapèze est la forme fermée) : le bracket
Q ± mu0 Sum |dHa_k| eps_k / sqrt(2a) est certifié FACE AU CONTINUUM,
espace + temps + QoI compris.

**Résultats** (n = 20, N = 400, S = 800, 3 cycles, beta = 0.7) :

- Théorème validé : eps(T) = 33.3 >= erreur réelle 0.78 (référence
  sur-résolue 2x en espace, 4x en temps), effectivité 43.
- **Gain x40 sur l'accumulation v2** (1324 -> 33), figure figv3_epsilon :
  la borne BEN reste quasi plate là où l'accumulation par pas croît
  linéairement, et elle couvre EN PLUS l'erreur de schéma en temps.
- Bracket TOTAL : Q = 2926.8 ± 239.0 J/m³ (16.3 %), contient la référence
  sur-résolue ; largeurs 16-20 % sur tout n = 5..80 (robustesse en n
  intacte : eps(T) = 66-76 pour n = 5..80) ; verdict Bean à n = 80
  reconfirmé avec bracket certifié total (Q/Q_Bean = 1.028).
- Convergence du bracket en O(sqrt(tau)) mesurée (ratios 1.72 et 1.31 par
  doublement de S) : le résidu dominant est le désaccord constitutif réel
  de la trajectoire affine par morceaux aux fronts mobiles, d'ordre
  Ec Jc x (volume de front) x tau par slab.

Fichiers v3 : `ben_v3.py`, `verify_v3.py`, `run_v3.py`, figures
`figv3_epsilon.png`, `figv3_bracket.png`.

## Article (n°34 du portfolio)

Jet complet v1 + cover letter pour **Superconductor Science and Technology**
(IOP), générés par `build_manuscript.js` et `build_cover.js` (lancer avec
`NODE_PATH` sur le node_modules de `Articles regroupés/`). Sortie directe
dans `Articles regroupés/Superconductor_Science_and_Technology/`
(Article_HTS_ACloss_Certified_SuST_v1.docx, abstract + 6 sections +
Théorème 1 avec preuve + 2 tables + 6 figures + équations OMML + 34 réf).
Ligne n°34 ajoutée au RECAPITULATIF maître.

## Crux pour l'article (ce que le PoC ne fait pas encore)

1. ~~Loi de puissance via écart primal-dual~~ : FAIT en v2, avec la
   robustesse en n démontrée numériquement (structurelle : module de forte
   convexité mu0/tau porté par la masse). Reste à en faire un théorème.
2. ~~Erreur de schéma en temps~~ : FAIT en v3 (Théorème 1 BEN, énoncé et
   prouvé ci-dessus, implémenté, validé, gain x40). Restes du chantier
   temps : (a) l'excitation certifiée est l'affine par morceaux ; passer à
   la sinusoïde exacte demande une estimée de stabilité par rapport à la
   donnée de bord (ou un relèvement localisé près des bords, dont le coût
   FY a été chiffré : la version naïve spatialement constante coûte
   Jc mu0 |delta'| L, rédhibitoire, la version localisée ~ (mu0 delta' w)²/phi'
   est jouable) ; (b) battre le O(sqrt(tau)) : le résidu vient des fronts,
   une trajectoire enrichie aux fronts (interpolation adaptée au transport
   du front dans le slab) devrait donner O(tau).
3. **Périodicité certifiée.** La dérive est mesurée (2e-3 à n=80) mais la
   distance à l'orbite périodique exacte n'est pas encadrée (le prox est
   non expansif, pas contractant). Piste naturelle : appliquer le Théorème
   1 sur UN cycle avec conditions périodiques (la structure BEN s'y prête)
   et jonction avec la ligne multiharmonique garantie de Linz
   (Langer-Repin-Wolfmayr, Wolfmayr 2024).
4. **Consistance spatiale du mode B (v1, Bean).** Remplaçable par le même
   majorant fonctionnel que la v2/v3 (Phi* = Jc |sigma|, fonction
   support) : unifier v1, v2 et v3 sous le seul certificat BEN (le
   Théorème 1 couvre le cas Bean avec Phi indicatrice telle quelle).
5. ~~Constante de quadrature de la QoI en temps~~ : FAIT en v3 (excitation
   affine par morceaux, boucle en forme fermée, plus aucune quadrature).
6. **2D (bande mince, benchmark n°1 du HTS Modelling Workgroup, verdict
   Norris), multi-conducteurs, formulation espace-temps** : la contrainte
   intégrale de courant sur maillage espace-temps (le « method 3 » ouvert
   de Grilli-Sirois-Laforest-Ashworth 2008, formalisé par la thèse Wan
   2014) reste un second front. Le Théorème 1 s'étend au p-curl 2D/3D
   (la représentation en flux devient une reconstruction H(div), c'est là
   que le flux équilibré 2D devient un vrai chantier).

## Références clés

- L. Prigozhin, Eur. J. Appl. Math. 7 (1996) 237, et J. Comput. Phys. 129
  (1996) 190 (formulation variationnelle du modèle de Bean).
- J. W. Barrett, L. Prigozhin, Interfaces Free Bound. 8 (2006) 349
  (formulations duales du critical state).
- A. T. S. Wan, M. Laforest, SIAM J. Numer. Anal. 58 (2020) 460
  (a posteriori p-curl, QoI pertes AC, non garanti : la référence à dépasser).
- J. Dular, C. Geuzaine, B. Vanderheyden, IEEE TAS 30 (2020)
  (encadrement empirique h-conforme / b-conforme).
- S. Repin, A Posteriori Estimates for PDE, de Gruyter 2008 ;
  D. Apushkinskaya, S. Repin, CMAM 22 (2022) (obstacle parabolique garanti).
- S. Bartels, A. Kaltenbach, arXiv:2402.06429 ; F. Gazca-Orozco,
  arXiv:2606.05088 (écart primal-dual, divergences de Bregman).
- F. Grilli, F. Sirois, M. Laforest, S. P. Ashworth, IEEE TAS 19 (2009)
  (formulation Periodic Space-Time, point de départ de ce chantier).
