# odefilter

[![PyPi Version](https://img.shields.io/pypi/v/odefilter.svg?style=flat-square)](https://pypi.org/project/odefilter/)
[![Docs](https://readthedocs.org/projects/pip/badge/?version=latest&style=flat-square)](https://odefilter.readthedocs.io)
[![GitHub stars](https://img.shields.io/github/stars/pnkraemer/odefilter.svg?style=flat-square&logo=github&label=Stars&logoColor=white)](https://github.com/pnkraemer/odefilter)
[![gh-actions](https://img.shields.io/github/workflow/status/pnkraemer/odefilter/ci?style=flat-square)](https://github.com/pnkraemer/odefilter/actions?query=workflow%3Aci)
<a href="https://github.com/pnkraemer/odefilter/blob/master/LICENSE"><img src="https://img.shields.io/github/license/pnkraemer/odefilter?style=flat-square&color=2b9348" alt="License Badge"/></a>



* **DOCUMENTATION:** TBD
* **ISSUE TRACKER:** [LINK](https://github.com/pnkraemer/odefilter/issues)


## Features include

### Initial value problem solvers
- [x] Stable implementation
- [x] Error estimation and step-size adaptation
- [x] EK0 (Think: explicit solvers)
- [x] EK1 (Think: semi-implicit solvers w/ Jacobians)
- [ ] UK1 (Think: semi-implicit solvers w/out Jacobians)
- [x] Global (automatic) calibration
- [x] Dynamic (automatic) calibration
- [ ] No automatic calibration
- [x] First-order problems
- [x] Higher-order problems
- [ ] Mass-matrix problems 
- [ ] Implicit differential equations
- [x] Custom information operators
- [x] Terminal-value simulation
- [x] Global simulation (the traditional ``solve()`` method)
- [x] Checkpointing
- [ ] Discrete forward-mode differentiation
- [ ] Discrete reverse-mode differentiation
- [ ] Continuous forward-mode differentiation
- [ ] Continuous reverse-mode differentiation
- [x] Autodiff initialisation
- [ ] Non-autodiff initialisation
- [ ] Model fit evaluation
- [ ] Discrete event handling
- [ ] Continuous event handling
- [ ] Physics-enhanced regression


### Boundary value problem solvers
- [ ] Separable boundary conditions
- [ ] Non-separable boundary conditions
- [ ] Bridge priors
- [ ] Higher-order problems
- [ ] Backends:
  - [ ] Gauss--Newton
  - [ ] Levenberg-Marquardt
  - [ ] ADMM
- [ ] Error estimation
- [ ] Mesh refinement


### State-space model machinery
- [ ] State-space model factorisations:
  - [x] Kronecker state-space models
  - [ ] Diagonal state-space models
  - [x] Dense state-space models (no factorisation)
- [ ] Dense output:
  - [x] Offgrid-marginalisation
  - [x] Posterior sampling
  - [ ] Joint distributions


## Installation

Get the most recent stable version from PyPi:

```
pip install odefilter
```
Or directly from GitHub:
```
pip install git+https://github.com/pnkraemer/odefilter.git
```

Read more about installing this package [here](https://odefilter.readthedocs.io/en/latest/getting_started/installation.html).


## Similar projects

* ProbNum
* Tornadox
* ProbNumDiffEq.jl
* Tornadox



Anything missing in this list? Please open an issue or make a pull request.
