# probdiffeq


[![PyPi Version](https://img.shields.io/pypi/v/probdiffeq.svg?style=flat-square)](https://pypi.org/project/probdiffeq/)
[![GitHub stars](https://img.shields.io/github/stars/pnkraemer/probdiffeq.svg?style=flat-square&logo=github&label=Stars&logoColor=white)](https://github.com/pnkraemer/probdiffeq)
[![gh-actions](https://img.shields.io/github/actions/workflow/status/pnkraemer/probdiffeq/ci.yaml?branch=main&style=flat-square)](https://github.com/pnkraemer/probdiffeq/actions?query=workflow%3Aci)
<a href="https://github.com/pnkraemer/probdiffeq/blob/master/LICENSE"><img src="https://img.shields.io/github/license/pnkraemer/probdiffeq?style=flat-square&color=2b9348" alt="License Badge"/></a>


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



* **QUICKSTART:** [LINK](https://pnkraemer.github.io/probdiffeq/quickstart/quickstart/)
* **EXAMPLES:** [LINK](https://pnkraemer.github.io/probdiffeq/examples/posterior_uncertainties/)
* **CHOOSING A SOLVER:** [LINK](https://pnkraemer.github.io/probdiffeq/quickstart/transitioning_from_other_packages/#choosing-a-solver)
* **API DOCUMENTATION:** [LINK](https://pnkraemer.github.io/probdiffeq/api_docs/solution_routines/)
* **ISSUE TRACKER:** [LINK](https://github.com/pnkraemer/probdiffeq/issues)
* **BENCHMARKS:** [LINK](https://pnkraemer.github.io/probdiffeq/api_docs/solution_routines/)


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


## What's next?

There are examples and there are advanced examples.

Consult the examples first. They show how to interact with the API, and explain some useful facts about probabilistic numerical solvers.

The advanced examples show applications of probabilistic numerical solvers, often in conjunction with external libraries.
For example, [this notebook](https://pnkraemer.github.io/probdiffeq/advanced_examples/physics_enhanced_regression_1/) shows how to combine ProbDiffEq with [Optax](https://optax.readthedocs.io/en/latest/index.html), and [this notebook](https://pnkraemer.github.io/probdiffeq/advanced_examples/physics_enhanced_regression_2/) does the same with [BlackJAX](https://optax.readthedocs.io/en/latest/index.html).

Proceed with the quickstart and the examples before moving to the advanced examples and the API documentation.


## Contributing
Contributions are absolutely welcome!
Start with checking the existing issues for a "good first issue" and have a look at  the developer documentation.

If you have a feature that you would like to see implemented, create an issue!

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
