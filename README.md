# probdiffeq

[![Actions status](https://github.com/pnkraemer/probdiffeq/workflows/ci/badge.svg)](https://github.com/pnkraemer/probdiffeq/actions)
[![image](https://img.shields.io/pypi/v/probdiffeq.svg)](https://pypi.python.org/pypi/probdiffeq)
[![image](https://img.shields.io/pypi/l/probdiffeq.svg)](https://pypi.python.org/pypi/probdiffeq)
[![image](https://img.shields.io/pypi/pyversions/probdiffeq.svg)](https://pypi.python.org/pypi/probdiffeq)

## Probabilistic solvers for differential equations in JAX

ProbDiffEq implements adaptive probabilistic numerical solvers for initial value problems.

It inherits automatic differentiation, vectorisation, and GPU capability from JAX.

**Features include:**

* ⚡ Calibration and step-size adaptation
* ⚡ Stable implementation of filtering, smoothing, and other estimation strategies
* ⚡ Custom information operators, dense output, and posterior sampling
* ⚡ State-space model factorisations
* ⚡ Physics-enhanced regression
* ⚡ Taylor-series estimation with and without Jets
* ⚡ Compatibility with other JAX-based libraries such as [Optax](https://optax.readthedocs.io/en/latest/index.html) or [BlackJAX](https://blackjax-devs.github.io/blackjax/).


**Tutorials:**

* **AN EASY EXAMPLE:** [LINK](https://pnkraemer.github.io/probdiffeq/examples_quickstart/easy_example/)
* **EXAMPLES:** [LINK](https://pnkraemer.github.io/probdiffeq/examples_solver_config/posterior_uncertainties/)
* **CHOOSING A SOLVER:** [LINK](https://pnkraemer.github.io/probdiffeq/getting_started/choosing_a_solver/)
* **API DOCUMENTATION:** [LINK](https://pnkraemer.github.io/probdiffeq/api_docs/ivpsolve/)
* **ISSUE TRACKER:** [LINK](https://github.com/pnkraemer/probdiffeq/issues)
* **BENCHMARKS:** [LINK](https://pnkraemer.github.io/probdiffeq/benchmarks/lotkavolterra/plot/)


## Installation

Get the most recent stable version from PyPi:

```
pip install probdiffeq
```
This installation assumes that [JAX](https://jax.readthedocs.io/en/latest/) is already available.

To install ProbDiffEq with `jax[cpu]`, run
```commandline
pip install probdiffeq[cpu]
```


**WARNING:**
_**This is a research project. Expect rough edges and sudden API changes.**_

**VERSIONING:**
As long as Probdiffeq is in its initial development phase (version 0.MINOR.PATCH), version numbers are increased as follows:

* Bugfixes and new features increase the PATCH version. 
* Breaking changes increase the MINOR version.

See also: [semantic versioning](https://semver.org/).


## What's next?

Start with the quickstart, continue with the `Solvers & Solutions` examples and only then move to the `Parameter estimation` examples and the API documentation.

The examples show how to interact with the API and explain some valuable facts about probabilistic numerical solvers.
They may be more instructive than the API docs.

The advanced examples show applications of probabilistic numerical solvers, often in conjunction with external libraries.
For example, [this notebook](https://pnkraemer.github.io/probdiffeq/advanced_examples/physics_enhanced_regression_1/) shows how to combine ProbDiffEq with [Optax](https://optax.readthedocs.io/en/latest/index.html), and [this notebook](https://pnkraemer.github.io/probdiffeq/advanced_examples/physics_enhanced_regression_2/) does the same with [BlackJAX](https://optax.readthedocs.io/en/latest/index.html).

## Citing this repository
If you find Probdiffeq helpful for your research project, please consider citing:

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

Probdiffeq's algorithms have been developed over many years and in multiple research papers.
Linking concrete citation information for specific algorithms is a work in progress.
Feel free to reach out if you need help determining which works to cite!

## Contributing
Contributions are welcome!
Check the existing issues for a "good first issue" and consult the developer documentation.

If you have a feature that you would like to see implemented, create an issue!

## Benchmarks

ProbDiffEq curates a range of benchmarks that includes various library-internal configurations
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
* [Diffrax](https://docs.kidger.site/diffrax/)

Here's how to transition from those packages: [link](https://pnkraemer.github.io/probdiffeq/quickstart/transitioning_from_other_packages/).

Is anything missing from this list? Please open an issue or make a pull request.

## You might also like

* [diffeqzoo](https://diffeqzoo.readthedocs.io/en/latest/): 
  A library, for example, implementations of differential equations in NumPy and JAX
* [probfindiff](https://probfindiff.readthedocs.io/en/latest/): 
  Probabilistic numerical finite differences in JAX.
