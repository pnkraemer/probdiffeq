# probdiffeq

[![CI](https://github.com/pnkraemer/probdiffeq/workflows/ci/badge.svg?branch=main)](https://github.com/pnkraemer/probdiffeq/actions)
[![PyPI version](https://img.shields.io/pypi/v/probdiffeq.svg)](https://pypi.python.org/pypi/probdiffeq)
[![License](https://img.shields.io/pypi/l/probdiffeq.svg)](https://pypi.python.org/pypi/probdiffeq)
[![Python versions](https://img.shields.io/pypi/pyversions/probdiffeq.svg)](https://pypi.python.org/pypi/probdiffeq)

## Probabilistic ODE solvers in JAX

**Probdiffeq** implements adaptive probabilistic numerical solvers for ordinary differential equations (ODEs). It builds on [JAX](https://jax.readthedocs.io/en/latest/), thus inheriting **automatic differentiation**, **vectorisation**, and **GPU acceleration**.

## Features

- ⚡ Calibration and step-size adaptation  
- ⚡ Stable implementations of filtering, smoothing, and other estimation strategies  
- ⚡ Custom information operators, dense output, and posterior sampling  
- ⚡ State-space model factorisations  
- ⚡ Parameter estimation
- ⚡ Taylor-series estimation with and without Jets  
- ⚡ Seamless interoperability with [Optax](https://optax.readthedocs.io/en/latest/index.html), [BlackJAX](https://blackjax-devs.github.io/blackjax/), and other JAX-based libraries  
- ⚡ Numerous tutorials (basic and advanced) -- see the [documentation](https://pnkraemer.github.io/probdiffeq/)  

---

## Installation

Install the latest release from PyPI:

```bash
pip install probdiffeq
```

> This assumes [JAX](https://jax.readthedocs.io/en/latest/) is already installed.  

To install with JAX (CPU backend):  

```bash
pip install probdiffeq[cpu]
```

⚠️ **Note:** This is an active research project. Expect rough edges and breaking API changes.

---

## Benchmarks

We maintain benchmarks comparing **Probdiffeq** against other solvers and libraries, including [SciPy](https://scipy.org/), [JAX](https://jax.readthedocs.io/en/latest/), and [Diffrax](https://docs.kidger.site/diffrax/).

Run benchmarks locally:

```bash
pip install .[example,test]
make benchmarks-run
```


---

## Contributing

Contributions are very welcome!  
- Browse open issues (look for “good first issue”).  
- Check the developer documentation.  
- Open an issue for feature requests or ideas.  

---

## Citing

If you use **Probdiffeq** in your research, please cite:

```bibtex
@phdthesis{kramer2024implementing,
  title={Implementing probabilistic numerical solvers for differential equations},
  author={Kr{\"a}mer, Peter Nicholas},
  year={2024},
  school={Universit{"a}t T{"u}bingen}
}
```
The [PDF](https://tobias-lib.ub.uni-tuebingen.de/xmlui/handle/10900/152754) explains the mathematics and algorithms behind this library.  

For the *solve-and-save-at* functionality, cite:

```bibtex
@InProceedings{kramer2024adaptive,
  title     = {Adaptive Probabilistic ODE Solvers Without Adaptive Memory Requirements},
  author    = {Kr{\"a}mer, Nicholas},
  booktitle = {Proceedings of the First International Conference on Probabilistic Numerics},
  pages     = {12--24},
  year      = {2025},
  editor    = {Kanagawa, Motonobu and Cockayne, Jon and Gessner, Alexandra and Hennig, Philipp},
  volume    = {271},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v271/kramer25a.html}
}
```
Link to the paper: [PDF](https://arxiv.org/abs/2410.10530).

Link to the experiments: 
[Code for experiments](https://github.com/pnkraemer/code-adaptive-prob-ode-solvers).  


Algorithms in **Probdiffeq** are based on multiple research papers. If you’re unsure which to cite, feel free to reach out. 

A (subjective, probdiffeq-centric) list of relevant work includes:


- Numerically robust implementations of probabilistic solvers:

  > Nicholas Krämer & Philipp Hennig (2024). Stable implementation of probabilistic ODE solvers. Journal of Machine Learning Research, 25(111), 1–29.  
  
  All suggestions made in this work are critical to Probdiffeq (and other libraries). They are rarely discussed though, and almost taken for granted by now.


- State-space model factorisations:

  > Nicholas Krämer, Nathanael Bosch, Jonathan Schmidt & Philipp Hennig (2022). Probabilistic ODE solutions in millions of dimensions.  In ICML 2022, 11634–11649. PMLR.  

  Every time Probdiffeq uses state-space model factorisations, it follows the recommendations in this work. 

- Adaptive step-size selection:
  
  > Michael Schober, Simo Särkkä & Philipp Hennig (2019). A probabilistic model for the numerical solution of initial value problems. Statistics and Computing, 29(1), 99–122.  
  
  > Nathanael Bosch, Philipp Hennig & Filip Tronarp (2021). Calibrated adaptive probabilistic ODE solvers. In AISTATS 2021, 3466–3474. PMLR.  
  
  > Nicholas Krämer (2025). Adaptive Probabilistic ODE Solvers Without Adaptive Memory Requirements. In Kanagawa, M., Cockayne, J., Gessner, A., & Hennig, P. (Eds.), Proceedings of the First International Conference on Probabilistic Numerics, 12–24. PMLR.
  
- Constraints, linearisation, and information operators:
  
  > Bosch, Nathanael, Filip Tronarp, and Philipp Hennig. "Pick-and-mix information operators for probabilistic ODE solvers." International Conference on Artificial Intelligence and Statistics. PMLR, 2022.

  >Tronarp, Filip, et al. "Probabilistic solutions to ordinary differential equations as nonlinear Bayesian filtering: a new perspective." Statistics and Computing 29.6 (2019): 1297-1315.

  See also the Linearisation-chapter in:
  
  > Krämer, Nicholas. Implementing probabilistic numerical solvers for differential equations. Diss. Dissertation, Tübingen, Universität Tübingen, 2024.

  which describes some methods not mentioned anywhere else.

- Parameter estimation:

  > Kersting, H., Krämer, N., Schiegg, M., Daniel, C., Tiemann, M., & Hennig, P. (2020, November). Differentiable likelihoods for fast inversion of’likelihood-free’dynamical systems. In International Conference on Machine Learning (pp. 5198-5208). PMLR.

  > Tronarp, Filip, Nathanael Bosch, and Philipp Hennig. "Fenrir: Physics-enhanced regression for initial value problems." International Conference on Machine Learning. PMLR, 2022.

  > Beck, J., Bosch, N., Deistler, M., Kadhim, K. L., Macke, J. H., Hennig, P., & Berens, P. (2024, July). Diffusion Tempering Improves Parameter Estimation with Probabilistic Integrators for Ordinary Differential Equations. In International Conference on Machine Learning (pp. 3305-3326). PMLR.


Anything missing? Reach out! 

## Versioning

Probdiffeq follows **0.MINOR.PATCH** until its first stable release:  
- **PATCH** → bugfixes & new features  
- **MINOR** → breaking changes  

See [semantic versioning](https://semver.org/).
Notably, Probdiffeq's API is not guaranteed to be stable, but we do our best to follow the versioning scheme so that downstream projects remain reproducible.

---

## Related projects

- [Tornadox](https://github.com/pnkraemer/tornadox)  
- [ProbNumDiffEq.jl](https://nathanaelbosch.github.io/ProbNumDiffEq.jl/stable/)  
- [ProbNum](https://probnum.readthedocs.io/en/latest/)  

The docs include guidance on migrating from these packages. Missing something? Open an issue or pull request!

---

## You might also like

- [diffeqzoo](https://diffeqzoo.readthedocs.io/en/latest/) — reference implementations of differential equations in NumPy and JAX  
- [probfindiff](https://probfindiff.readthedocs.io/en/latest/) — probabilistic finite-difference methods in JAX  
