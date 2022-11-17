# probdiffeq

[![PyPi Version](https://img.shields.io/pypi/v/probdiffeq.svg?style=flat-square)](https://pypi.org/project/probdiffeq/)
[![Docs](https://readthedocs.org/projects/pip/badge/?version=latest&style=flat-square)](https://probdiffeq.readthedocs.io)
[![GitHub stars](https://img.shields.io/github/stars/pnkraemer/probdiffeq.svg?style=flat-square&logo=github&label=Stars&logoColor=white)](https://github.com/pnkraemer/probdiffeq)
[![gh-actions](https://img.shields.io/github/workflow/status/pnkraemer/probdiffeq/ci?style=flat-square)](https://github.com/pnkraemer/probdiffeq/actions?query=workflow%3Aci)
<a href="https://github.com/pnkraemer/probdiffeq/blob/master/LICENSE"><img src="https://img.shields.io/github/license/pnkraemer/probdiffeq?style=flat-square&color=2b9348" alt="License Badge"/></a>



* **DOCUMENTATION:** TBD
* **ISSUE TRACKER:** [LINK](https://github.com/pnkraemer/probdiffeq/issues)


## Installation

Get the most recent stable version from PyPi:

```
pip install probdiffeq
```
Or directly from GitHub:
```
pip install git+https://github.com/pnkraemer/probdiffeq.git
```

Read more about installing this package [here](https://probdiffeq.readthedocs.io/en/latest/getting_started/installation.html).


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


## Similar projects

* ProbNum
* Tornadox
* ProbNumDiffEq.jl
* Tornadox



Anything missing in this list? Please open an issue or make a pull request.
