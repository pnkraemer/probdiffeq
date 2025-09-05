# probdiffeq

[![Actions status](https://github.com/pnkraemer/probdiffeq/workflows/ci/badge.svg)](https://github.com/pnkraemer/probdiffeq/actions)
[![image](https://img.shields.io/pypi/v/probdiffeq.svg)](https://pypi.python.org/pypi/probdiffeq)
[![image](https://img.shields.io/pypi/l/probdiffeq.svg)](https://pypi.python.org/pypi/probdiffeq)
[![image](https://img.shields.io/pypi/pyversions/probdiffeq.svg)](https://pypi.python.org/pypi/probdiffeq)

## Probabilistic ODE solvers in JAX

Probdiffeq implements adaptive probabilistic numerical solvers for initial value problems.

It inherits automatic differentiation, vectorisation, and GPU capability from JAX.

**Features include:**

* ⚡ Calibration and step-size adaptation
* ⚡ Stable implementation of filtering, smoothing, and other estimation strategies
* ⚡ Custom information operators, dense output, and posterior sampling
* ⚡ State-space model factorisations
* ⚡ Physics-enhanced regression
* ⚡ Taylor-series estimation with and without Jets
* ⚡ Compatibility with other JAX-based libraries such as [Optax](https://optax.readthedocs.io/en/latest/index.html) or [BlackJAX](https://blackjax-devs.github.io/blackjax/).
* ⚡ Numerous tutorials (basic and advanced) -- check the documentation!



## Installation

Get the most recent version from PyPi:

```
pip install probdiffeq
```
This installation assumes that [JAX](https://jax.readthedocs.io/en/latest/) is already available.

To install Probdiffeq with `jax[cpu]`, run
```commandline
pip install probdiffeq[cpu]
```


**WARNING:**
_**This is a research project. Expect rough edges and sudden API changes.**_



## Citing this repository
If you find Probdiffeq helpful for your research, please consider citing:

```bibtex
@phdthesis{kramer2024implementing,
  title={Implementing probabilistic numerical solvers for differential equations},
  author={Kr{\"a}mer, Peter Nicholas},
  year={2024},
  school={Universit{\"a}t T{\"u}bingen}
}
```
This thesis contains detailed information about the maths and algorithms behind what is implemented here.
A PDF is available [at this link](https://tobias-lib.ub.uni-tuebingen.de/xmlui/handle/10900/152754).

If you use the solve-and-save-at functionality, please cite
```bibtex
@InProceedings{kramer2024adaptive,
  title     = {Adaptive Probabilistic ODE Solvers Without Adaptive Memory Requirements},
  author    = {Kr\"{a}mer, Nicholas},
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
This article introduced the algorithm we use.
The implementation is slightly different to what we would do for non-probabilistic solvers; see the paper.
A PDF is available [here](https://arxiv.org/abs/2410.10530) and the paper's experiments are [here](https://github.com/pnkraemer/code-adaptive-prob-ode-solvers).


_Probdiffeq's algorithms have been developed over many years and in multiple research papers.
Linking concrete citation information for specific algorithms is a work in progress.
Feel free to reach out if you need help determining which works to cite!_


**VERSIONING:**
As long as Probdiffeq is in its initial development phase (version 0.MINOR.PATCH), version numbers are increased as follows:

* Bugfixes and new features increase the PATCH version. 
* Breaking changes increase the MINOR version.

See also: [semantic versioning](https://semver.org/).

## Contributing
Contributions are welcome!
Check the existing issues for a "good first issue" and consult the developer documentation.

If you have a feature that you would like to see implemented, create an issue!

## Benchmarks

Probdiffeq curates a range of benchmarks that includes various library-internal configurations
but also other packages like [SciPy](https://scipy.org/), [JAX](https://jax.readthedocs.io/en/latest/), or [Diffrax](https://docs.kidger.site/diffrax/). 

To run the benchmark locally, install all dependencies via
```commandline
pip install .[example,test]
```
and then either open Jupyter and go to `docs/benchmarks`
or execute all benchmarks via
```commandline
make benchmarks-run
```
Be patient; it might take a while. 
Afterwards, open Jupyter to look at the result or build the documentation via
```
mkdocs serve
```
What do you find?

## Similar projects

* [Tornadox](https://github.com/pnkraemer/tornadox)
* [ProbNumDiffEq.jl](https://nathanaelbosch.github.io/ProbNumDiffEq.jl/stable/)
* [ProbNum](https://probnum.readthedocs.io/en/latest/)

The online docs explain how to transition from those packages.

Is anything missing from this list? Please open an issue or make a pull request.

## You might also like

* [diffeqzoo](https://diffeqzoo.readthedocs.io/en/latest/): 
  A library, for example, implementations of differential equations in NumPy and JAX
* [probfindiff](https://probfindiff.readthedocs.io/en/latest/): 
  Probabilistic numerical finite differences in JAX.
