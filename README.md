# probdiffeq

[![CI](https://github.com/pnkraemer/probdiffeq/workflows/ci/badge.svg)](https://github.com/pnkraemer/probdiffeq/actions)
[![PyPI version](https://img.shields.io/pypi/v/probdiffeq.svg)](https://pypi.python.org/pypi/probdiffeq)
[![License](https://img.shields.io/pypi/l/probdiffeq.svg)](https://pypi.python.org/pypi/probdiffeq)
[![Python versions](https://img.shields.io/pypi/pyversions/probdiffeq.svg)](https://pypi.python.org/pypi/probdiffeq)

## Probabilistic ODE solvers in JAX

**Probdiffeq** implements adaptive probabilistic numerical solvers for initial value problems (IVPs).  
It is built on [JAX](https://jax.readthedocs.io/en/latest/), inheriting **automatic differentiation**, **vectorisation**, and **GPU acceleration**.

## Features

- ⚡ Calibration and step-size adaptation  
- ⚡ Stable implementations of filtering, smoothing, and other estimation strategies  
- ⚡ Custom information operators, dense output, and posterior sampling  
- ⚡ State-space model factorisations  
- ⚡ Parameter estimation
- ⚡ Taylor-series estimation with and without Jets  
- ⚡ Seamless interoperability with [Optax](https://optax.readthedocs.io/en/latest/index.html), [BlackJAX](https://blackjax-devs.github.io/blackjax/), and other JAX-based libraries  
- ⚡ Numerous tutorials (basic and advanced) — see the [documentation](https://pnkraemer.github.io/probdiffeq/)  

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
  author={Kr{"a}mer, Peter Nicholas},
  year={2024},
  school={Universit{"a}t T{"u}bingen}
}
```
The [PDF](https://tobias-lib.ub.uni-tuebingen.de/xmlui/handle/10900/152754) explains the mathematics and algorithms behind this library.  

For the *solve-and-save-at* functionality, cite:

```bibtex
@InProceedings{kramer2024adaptive,
  title     = {Adaptive Probabilistic ODE Solvers Without Adaptive Memory Requirements},
  author    = {Kr"{a}mer, Nicholas},
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

📌 Algorithms in Probdiffeq are based on multiple research papers. If you’re unsure which to cite, feel free to reach out.  

---

## Versioning

Probdiffeq follows **0.MINOR.PATCH** until its first stable release:  
- **PATCH** → bugfixes & new features  
- **MINOR** → breaking changes  

See [semantic versioning](https://semver.org/).

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
