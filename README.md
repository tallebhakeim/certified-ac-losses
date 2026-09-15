# certified-ac-losses

[![DOI](https://zenodo.org/badge/DOI/ZENODO_DOI_PLACEHOLDER.svg)](https://doi.org/ZENODO_DOI_PLACEHOLDER)

Guaranteed (mathematically certified) two-sided error bounds for AC loss
calculations in type-II superconductors: from the Bean critical-state model
to the E-J power law, via a space-time convex-duality estimate of
Brezis-Ekeland-Nayroles type.

Companion code for the manuscript:

> H. Talleb, "Guaranteed error bounds for AC loss calculations in type-II
> superconductors: a certified framework from the Bean model to the E-J
> power law via a space-time convex-duality estimate", submitted to
> *Superconductor Science and Technology* (2026).

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
| `make_figures_en.py` | All manuscript figures (`fig_en_*.png`, English) |

## Requirements and usage

Python 3 with `numpy`, `scipy`, `matplotlib` (`pip install -r requirements.txt`). No other dependency.

```bash
python3 verify.py       # Bean stage        -> expected: 14/14 PASS
python3 verify_v2.py    # power-law stage   -> expected: 12/12 PASS
python3 verify_v3.py    # Theorem 1 stage   -> expected:  8/8 PASS
python3 make_figures_en.py   # regenerates the manuscript figures
```

Total: 34/34 tests, each suite printing the certified brackets and the
verdicts against the analytical references (Bean loss formulas, closed-form
profiles, over-resolved numerical references).

## Headline results reproduced by the suites

- Bean model: guaranteed loss brackets of relative width below 1.1% that
  provably contain Bean's analytical values for amplitude ratios 0.4 to 1.4.
- Power law: certificate effectivities between 4.3 and 8.9 uniformly over
  n = 5 to 80 (the strong-convexity modulus is carried by the mass term,
  hence independent of n); certified resolution of the flux-creep excess,
  +2.5% above the Bean loss at n = 80.
- Theorem 1 (space-time majorant): forty-fold tightening of the
  step-accumulated bounds, and fully certified loss brackets of relative
  width about 16% with respect to the exact continuum solution, covering
  spatial, temporal and quantity-of-interest errors simultaneously.

## Author

Hakeim Talleb, Sorbonne Universite, Universite Paris-Saclay, CNRS,
CentraleSupelec, Laboratoire de Genie Electrique et Electronique de Paris
(GeePs), Gif-sur-Yvette, France. Contact: hakeim.talleb@sorbonne-universite.fr

## How to cite

Please cite the manuscript above and the archived code (concept DOI, always
resolving to the latest version): https://doi.org/ZENODO_DOI_PLACEHOLDER

## Licence

MIT, see `LICENSE`.
