# Probdiffeq Migration Guide

This guide helps you get started with Probdiffeq for solving ordinary differential equations (ODEs), especially if you are familiar with other probabilistic or non-probabilistic ODE solvers in Python or Julia.

Probdiffeq is a JAX library that focuses on state-space-model-based formulations of probabilistic IVP solvers. For what this means, have a look at [this thesis](https://tobias-lib.ub.uni-tuebingen.de/xmlui/handle/10900/152754).



## Transitioning from ProbNumDiffEq.jl (Julia)

[ProbNumDiffEq.jl](https://nathanaelbosch.github.io/ProbNumDiffEq.jl/stable/) is a library for probabilistic IVP solvers in Julia, similar to Probdiffeq. However, both libraries are unrelated.

* Probdiffeq is Python/JAX-based; ProbNumDiffEq is Julia-based.
* Probdiffeq provides additional solvers, dense output, and posterior sampling.
* ProbNumDiffEq handles mass-matrix problems and callbacks, which Probdiffeq does not (yet).

To translate ProbNumDiffEq.jl code to Probdiffeq code:

| ProbNumDiffEq.jl           | ProbDiffEq Equivalent                                      |
|----------------------------|------------------------------------------------------------|
| `EK0` / `EK1`              | `ts0()` / `ts1()`                                         |
| `DynamicDiffusion` / `FixedDiffusion` | `ivpsolvers.solver_dynamic()` or `ivpsolvers.solver_mle()` |
| `IWP(diffusion=x^2)` |  `prior_wiener_integrated(output_scale=x)`                                       |
| Filtering and smoothing via `smooth=true/false`      | Solver strategy constructions, including one for fixed-point smoothing    |


Both libraries are evolving; consult the latest API documentation when in doubt.



## Transitioning from ProbNum (Python, Numpy)

[ProbNum](https://probnum.readthedocs.io/en/latest/) is a general probabilistic numerics library based on Numpy. Probdiffeq specializes in IVP solvers using pure JAX, offering:

* Greater efficiency for ODE problems because of JAX (e.g. jit)
* Probdiffeq implements more mature solvers. The algorithms are generally faster (eg state-space model factorisations, improved adaptive step-size selection)
* Probdiffeq offers more solvers and somewhat richer outputs (sampling, marginal likelihoods, etc.).



## Transitioning from Diffrax

[Diffrax](https://docs.kidger.site/diffrax/) is a JAX-based library for differential equations. Key differences:

* Diffrax solvers are non-probabilistic; Probdiffeq solvers are probabilistic.
* Vector fields: Diffrax uses `ODETerm()`; Probdiffeq uses plain functions `(*ys, t)`.
* Solver construction: Diffrax requires (`diffrax.Tsit5()`); Probdiffeq constructs probabilistic state-space models.

Approximate solver mapping:

| Diffrax                     | ProbDiffEq Equivalent                                     |
|-----------------------------|-----------------------------------------------------------|
| `Heun()`, `Midpoint()`      | `prior_ibm(num_derivatives=1)` or `ts0()`                |
| `Tsit5()`, `Dopri5()`       | Increase `num_derivatives=4`                               |
| `Dopri8()`                   | Increase `num_derivatives=5-7`; `ts1()` recommended but not required |
| `Kvaerno3()`–`Kvaerno5()`   | Use `num_derivatives=2-4` with `ts1()` correction         |
| Other methods                | Work in progress                                          |




## General differences from conventional ODE solvers (e.g., SciPy, jax.odeint)

* Solutions are posterior distributions instead of point estimates, enabling uncertainty quantification and more sophisticated models (eg easy switch to second-order problems).
* Solver modes are explicit: `simulate_terminal_values()`, `solve_adaptive_save_every_step()`, `solve_adaptive_save_at()` instead of a one-size-fits-all `solve()` method
