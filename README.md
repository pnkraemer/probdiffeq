# probdiffeq

[![GitHub stars](https://img.shields.io/github/stars/pnkraemer/probdiffeq.svg?style=flat-square&logo=github&label=Stars&logoColor=white)](https://github.com/pnkraemer/probdiffeq)
[![gh-actions](https://img.shields.io/github/actions/workflow/status/pnkraemer/probdiffeq/ci.yaml?branch=main&style=flat-square)](https://github.com/pnkraemer/probdiffeq/actions?query=workflow%3Aci)
<a href="https://github.com/pnkraemer/probdiffeq/blob/master/LICENSE"><img src="https://img.shields.io/github/license/pnkraemer/probdiffeq?style=flat-square&color=2b9348" alt="License Badge"/></a>



* **DOCUMENTATION:** TBD
* **ISSUE TRACKER:** [LINK](https://github.com/pnkraemer/probdiffeq/issues)


## Installation

Get the most recent stable version from PyPi:

```
pip install probdiffeq
```
This installation assumes that JAX is already available.

To install `probdiffeq` with `jax[cpu]`, run
```commandline
pip install probdiffeq[cpu]
```


## Features include

### Initial value problem solvers
- [x] Stable implementation
- [x] Fixed-steps
- [x] Error estimation and step-size adaptation
- [x] I-control
- [x] PI-control
- [ ] PID-control
- [x] Zeroth-order Taylor series (Think: explicit solvers)
- [x] First-order Taylor series (Think: semi-implicit solvers w/ Jacobians)
- [x] Moment matching algorithms (think: semi-implicit solvers w/o Jacobians):
  - [x] UK
  - [x] GHK
  - [x] CK
  - [x] Arbitrary cubature rules
  - [ ] BQ
- [x] Global (automatic) calibration
- [x] Dynamic (automatic) calibration
- [x] No automatic calibration
- [ ] State-space model factorizations:
  - [x] Isotropic state-space models
  - [ ] Kronecker state-space models
  - [x] Diagonal state-space models
  - [x] Dense state-space models (no factorization)
- [x] First-order problems
- [x] Higher-order problems
- [ ] Mass-matrix problems 
- [ ] Implicit differential equations
- [ ] Manifold updates (think: energy conservation)
- [x] Custom information operators
- [x] Terminal-value simulation
- [x] Global simulation (the traditional ``solve()`` method)
- [x] Checkpointing
- [x] Discrete forward-mode differentiation
- [x] Discrete reverse-mode differentiation
- [ ] Continuous forward-mode differentiation
- [ ] Continuous reverse-mode differentiation
- [x] Autodiff initialisation
- [x] Non-autodiff initialisation
- [ ] Model fit evaluation
- [ ] Discrete event handling
- [ ] Continuous event handling
- [x] Physics-enhanced regression
- [ ] Dense output:
  - [x] Offgrid-marginalisation
  - [x] Posterior sampling
  - [ ] Joint distributions

and many more.

### Boundary value problem solvers
- [ ] Separable boundary conditions
- [ ] Non-separable boundary conditions
- [ ] Bridge priors
- [ ] Higher-order problems
- [ ] Backends:
  - [ ] Gauss-Newton
  - [ ] Levenberg-Marquardt
  - [ ] ADMM
- [ ] Error estimation
- [ ] Mesh refinement


### Tutorials
- [x] Getting started
- [x] Different solve() versions
- [x] Posterior uncertainties
- [x] Different ways of smoothing
- [x] Exploring the solution object
- [ ] Custom information operators
- [x] Physics-enhanced regression
- [ ] Probabilistic numerical method of lines
- [ ] TBC...

### Benchmarks
- [x] Lotka-Volterra
- [ ] Pleiades
- [ ] Stiff van-der-Pol
- [ ] 100-dimensional linear ODE
- [ ] 1000-dimensional linear ODE?
- [ ] HIRES


## Examples

There are examples and there are advanced examples.

Consult the examples first. They show how to interact with the API, and explain some useful facts about probabilistic numerical solvers.

The advanced examples show applications of probabilistic numerical solvers, often in conjunction with external libraries.


## Development

### Installation
To install all development-relevant dependencies, install either of
```
pip install probdiffeq[test]  # pytest, ...
pip install probdiffeq[format]  # black, isort, ...
pip install probdiffeq[lint]  # flake8, ...
pip install probdiffeq[example]  # tueplots, diffrax, blackjax, ...
pip install probdiffeq[doc]  # mkdocs, ...
pip install probdiffeq[full]  # all of the above
```

### Continuous integration
Run the checks with a makefile, use either of the below
```
make format
make lint
make test
make example
make doc
```
Remove auxiliary files with 
```
make clean
```
### Pre-commit hook
To ensure that all commits satisfy most of the linters, no big files are addedd accidentally, and so on, use a pre-commit hook
```
pip install pre-commit  # included in `pip install -e .[full]`
pre-commit install
```
You may verify the installation by running
```commandline
pre-commit run
```



## Similar projects

* ProbNum
* Tornadox
* ProbNumDiffEq.jl
* Diffrax

Check out how to transition from those packages (-> quickstart).

Anything missing in this list? Please open an issue or make a pull request.
