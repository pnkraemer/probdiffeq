# odefilter

[![PyPi Version](https://img.shields.io/pypi/v/odefilter.svg?style=flat-square)](https://pypi.org/project/odefilter/)
[![Docs](https://readthedocs.org/projects/pip/badge/?version=latest&style=flat-square)](https://odefilter.readthedocs.io)
[![GitHub stars](https://img.shields.io/github/stars/pnkraemer/odefilter.svg?style=flat-square&logo=github&label=Stars&logoColor=white)](https://github.com/pnkraemer/odefilter)
[![gh-actions](https://img.shields.io/github/workflow/status/pnkraemer/odefilter/ci?style=flat-square)](https://github.com/pnkraemer/odefilter/actions?query=workflow%3Aci)
<a href="https://github.com/pnkraemer/odefilter/blob/master/LICENSE"><img src="https://img.shields.io/github/license/pnkraemer/odefilter?style=flat-square&color=2b9348" alt="License Badge"/></a>



* **DOCUMENTATION:** TBD
* **ISSUE TRACKER:** [LINK](https://github.com/pnkraemer/odefilter/issues)


## Features include


### Markov process machinery
- [ ] Dense output
- [ ] Sample from posterior
- [ ] Extrapolation
- [ ] Continuous-time process (for dense output)
- [ ] Discrete-time process (for no dense output)
- [ ] Simple Kalman filtering (for bridge priors, maybe for inverse problems)


### Initial value problem solvers
- [ ] EK0 (36 solvers!)
  - [ ] Kronecker vs. diagonal
  - [ ] Dense output vs terminal value vs checkpoint
  - [ ] first-order problem vs second-order problem vs general (Kronecker-structure-preserving) information operator
  - [ ] time-constant diffusion vs time-varying diffusion
- [ ] EK1
  - [ ] Full vs. diagonal
  - [ ] Dense output vs terminal value vs checkpoint
  - [ ] first-order problem vs second-order problem vs general (Kronecker-structure-preserving) information operator
  - [ ] time-constant diffusion vs time-varying diffusion
- [ ] UK1
  - [ ] Full vs. diagonal
  - [ ] Dense output vs terminal value vs checkpoint
  - [ ] first-order problem vs second-order problem vs general (Kronecker-structure-preserving) information operator
  - [ ] time-constant diffusion vs time-varying diffusion
- [ ] Error estimation, adaptive steps
- [ ] Autodiff initialisation
- [ ] Second-order problems
- [ ] Mass-matrix problems
- [ ] Build a solver from an information operator
- [ ] Build a Kronecker solver from an information operator
- [ ] Build a batch solver from an information operator
- [ ] Evaluate model-fit 
- [ ] General priors (and stacked state-space models!)
- [ ] Discrete event handling
- [ ] Continuous event handling
- [ ] Evaluate-extrapolate-correct solvers
- [ ] Extrapolate-evaluate-correct solvers
- [ ] TBC

### Boundary value problem solvers
- [ ] Separable boundary conditions
- [ ] Non-separable boundary conditions
- [ ] Bridge priors
- [ ] Higher-order problems
- [ ] Gauss--Newton
- [ ] Levenberg-Marquardt
- [ ] ADMM
- [ ] Error estimation
- [ ] Mesh refinement
- 



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
* ProbNumDiffEq.jl
* Tornadox



Anything missing in this list? Please open an issue or make a pull request.
