# probdiffeq


[![PyPi Version](https://img.shields.io/pypi/v/probdiffeq.svg?style=flat-square&color=darkgray)](https://pypi.org/project/probdiffeq/)
[![gh-actions](https://img.shields.io/github/actions/workflow/status/pnkraemer/probdiffeq/ci.yaml?branch=main&style=flat-square)](https://github.com/pnkraemer/probdiffeq/actions?query=workflow%3Aci)
<a href="https://github.com/pnkraemer/probdiffeq/blob/master/LICENSE"><img src="https://img.shields.io/github/license/pnkraemer/probdiffeq?style=flat-square&color=2b9348" alt="License Badge"/></a>
[![GitHub stars](https://img.shields.io/github/stars/pnkraemer/probdiffeq.svg?style=flat-square&logo=github&label=Stars&logoColor=white)](https://github.com/pnkraemer/probdiffeq)
![Python](https://img.shields.io/badge/python-3.9+-black.svg?style=flat-square)


## Probabilistic solvers for differential equations in JAX

ProbDiffEq implements adaptive probabilistic numerical solvers for initial value problems.

It inherits automatic differentiation, vectorisation, and GPU capability from JAX.
Features include:

* Stable implementation
* Calibration, step-size adaptation, and checkpointing
* State-space model factorisations
* Dense output and posterior sampling
* Filtering, smoothing, and many other backends
* Custom information operators
* Physics-enhanced regression
* Compatibility with other JAX-based libraries such as [Optax](https://optax.readthedocs.io/en/latest/index.html) or [BlackJAX](https://blackjax-devs.github.io/blackjax/).

and many more.



* **AN EASY EXAMPLE:** [LINK](https://pnkraemer.github.io/probdiffeq/getting_started/easy_example/)
* **EXAMPLES:** [LINK](https://pnkraemer.github.io/probdiffeq/examples_solver_config/posterior_uncertainties/)
* **CHOOSING A SOLVER:** [LINK](https://pnkraemer.github.io/probdiffeq/getting_started/choosing_a_solver/)
* **API DOCUMENTATION:** [LINK](https://pnkraemer.github.io/probdiffeq/api_docs/ivpsolve/)
* **ISSUE TRACKER:** [LINK](https://github.com/pnkraemer/probdiffeq/issues)
* **BENCHMARKS:** [LINK](https://pnkraemer.github.io/probdiffeq/benchmarks/lotkavolterra/plot/)
* **CHANGELOG:** [LINK](https://pnkraemer.github.io/probdiffeq/dev_docs/changelog/)


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

**WARNING:** This repository is experimental. Functionality may change frequently and without deprecation policies.

## What's next?

Start with the quickstart, continue with the `Solvers & Solutions` examples and only then move to the `Parameter estimation` examples and the API documentation.

The examples show how to interact with the API, and explain some useful facts about probabilistic numerical solvers.
While the API is not stable yet, the examples may be more instructive than the API docs.

The advanced examples show applications of probabilistic numerical solvers, often in conjunction with external libraries.
For example, [this notebook](https://pnkraemer.github.io/probdiffeq/advanced_examples/physics_enhanced_regression_1/) shows how to combine ProbDiffEq with [Optax](https://optax.readthedocs.io/en/latest/index.html), and [this notebook](https://pnkraemer.github.io/probdiffeq/advanced_examples/physics_enhanced_regression_2/) does the same with [BlackJAX](https://optax.readthedocs.io/en/latest/index.html).



## Contributing
Contributions are absolutely welcome!
Start with checking the existing issues for a "good first issue" and have a look at  the developer documentation.

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
make run-benchmarks
```
Be patient, it might take a while. 
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

Check out how to transition from those packages: [link](https://pnkraemer.github.io/probdiffeq/quickstart/transitioning_from_other_packages/).

Anything missing in this list? Please open an issue or make a pull request.

## You might also like

* [diffeqzoo](https://diffeqzoo.readthedocs.io/en/latest/): 
  A library for example implementations of differential equations in NumPy and JAX
* [probfindiff](https://probfindiff.readthedocs.io/en/latest/): 
  Probabilistic numerical finite differences, in JAX.
