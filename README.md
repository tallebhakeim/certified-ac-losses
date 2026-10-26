# certified-ac-losses

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22766982.svg)](https://doi.org/10.5281/zenodo.22766982)

Guaranteed (mathematically certified) two-sided error bounds for AC loss
calculations in type-II superconductors: from the Bean critical-state model
to the E-J power law, via a space-time convex-duality estimate of
Brezis-Ekeland-Nayroles type.

Companion code for the manuscript:

> H. Talleb, "Guaranteed error bounds for AC loss calculations in type-II
> superconductors: a certified framework from the Bean model to the E-J
> power law via a space-time convex-duality estimate", *Superconductor
> Science and Technology* (2026), manuscript SUST-107877, under revision.

Every number and every figure of the paper is reproduced by the scripts in
this repository. A French research-notes version of this document, with the
full derivations and the proof sketches, is in `README_FR.md`.

## Contents

| File | Role |
|---|---|
| `bean_slab_certified.py` | Bean model (Prigozhin sweeping process): exact-in-time Moreau catch-up, closed-form conical projection oracle, dual QP certificates, certified loss brackets |
| `powerlaw_slab_certified.py` | E-J power law: proximal implicit steps (damped Newton), exact primal-dual gap identity, 1D equilibrated flux, guaranteed Fenchel-Young quadratures |
| `ben_v3.py` | Theorem 1: space-time Brezis-Ekeland-Nayroles majorant (pure Fenchel-Young form), fully certified loss brackets (space + time + quantity of interest) |
| `verify.py` | Verification suite, Bean stage (14 tests) |
| `verify_v2.py` | Verification suite, power-law stage (12 tests) |
| `verify_v3.py` | Verification suite, space-time certificate (8 tests) |
| `run_poc.py`, `run_v2.py`, `run_v3.py` | Figure generation (research versions, French labels) |
| `make_figures_en.py` | Manuscript figures 1 to 7 (`fig_en_*.png`, English) |
| `revision_certificates.py` | Revision (v1.1): Theorem 1 for the Bean model on non-aligned meshes (residual (16)), Lemma 2 and the bracket for the exact sinusoid (15), certified periodicity indicators, exact factorization of the loss effectivity |
| `run_revision.py` | Revision: every new number of the revised paper (Tables 2 and 3, sections 3.2, 4.2-4.4, 5.3), written to `results_revision.json`, and figures 8-9 (`fig_en_loss_effectivity.png`, `fig_en_bean_ben.png`); about 6 min on 8 cores |
| `verify_revision.py` | Verification suite of the revision (12 tests) |
| `results_revision.json` | Output of `run_revision.py`, read by the manuscript generator |

## Requirements and usage

Python 3 with `numpy`, `scipy`, `matplotlib` (`pip install -r requirements.txt`). No other dependency.

```bash
python3 verify.py       # Bean stage        -> expected: 14/14 PASS
python3 verify_v2.py    # power-law stage   -> expected: 12/12 PASS
python3 verify_v3.py    # Theorem 1 stage   -> expected:  8/8 PASS
python3 make_figures_en.py   # regenerates figures 1 to 7
python3 run_revision.py      # revision numbers + figures 8 and 9
python3 verify_revision.py   # revision          -> expected: 12/12 PASS
```

Total: 34/34 + 12/12 tests, each suite printing the certified brackets and the
verdicts against the analytical references (Bean loss formulas, closed-form
profiles, over-resolved numerical references).

## Headline results reproduced by the suites

- Bean model: guaranteed loss brackets of relative width below 1.1% that
  provably contain Bean's analytical values for amplitude ratios 0.4 to 1.4.
- Power law: per-step certificate effectivities between 4.3 and 8.9
  uniformly over n = 5 to 80 (the strong-convexity modulus is carried by
  the mass term, hence independent of n). The flux-creep excess of the loss
  over the Bean value (about +3% at n = 40-80) is resolved by converged
  computations, not certified by the present brackets.
- Theorem 1 (space-time majorant): forty-fold tightening of the
  step-accumulated bounds, and fully certified brackets of the third-cycle
  loss of the slab driven by the exact 50 Hz sinusoid (with Lemma 2), of
  relative width 16-20% for n = 5 to 80 (N = 400, S = 800), covering
  spatial, temporal and quantity-of-interest errors; the loss effectivity
  (bound / true loss error) is 51 to 315, factorized in `run_revision.py`.
- Bean model on meshes not aligned with the sinusoidal drive (Theorem 1):
  brackets containing the analytical loss at every resolution, from 46% at
  N = 100 to 1.4% at N = 3200.

Scope: one-dimensional slab in a parallel field (verification benchmark),
constitutive laws convex in J with Jc independent of the solution,
isothermal. See section 5 of the paper for the path to thin strips and
coated conductors.

## Author

Hakeim Talleb, Sorbonne Universite, CNRS, Laboratoire de Genie Electrique
et Electronique de Paris, 75252, Paris, France; Universite Paris-Saclay,
CentraleSupelec, CNRS, Laboratoire de Genie Electrique et Electronique de
Paris, 91192, Gif-sur-Yvette, France. Contact: hakeim.talleb@sorbonne-universite.fr

## How to cite

Please cite the manuscript above and the archived code (concept DOI, always
resolving to the latest version): https://doi.org/10.5281/zenodo.22766982

## Licence

MIT, see `LICENSE`.
