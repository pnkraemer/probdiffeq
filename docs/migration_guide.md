# Probdiffeq Migration Guide

This guide helps you get started with Probdiffeq for solving ordinary differential equations (ODEs), especially if you are familiar with other probabilistic or non-probabilistic ODE solvers in Python or Julia.

Probdiffeq is a JAX library that focuses on state-space-model-based formulations of probabilistic IVP solvers. For what this means, have a look at [this thesis](https://tobias-lib.ub.uni-tuebingen.de/xmlui/handle/10900/152754).



## Transitioning from ProbNumDiffEq.jl (Julia)

[ProbNumDiffEq.jl](https://nathanaelbosch.github.io/ProbNumDiffEq.jl/stable/) is a library for probabilistic IVP solvers in Julia, similar to Probdiffeq. However, while the feature offerings are similar, the libraries are unrelated.
To translate ProbNumDiffEq.jl code to Probdiffeq code:

| ProbNumDiffEq.jl           | ProbDiffEq Equivalent                                      |
|----------------------------|------------------------------------------------------------|
| `EK0` / `EK1`              | `constraint_ode_ts0()` / `constraint_ode_ts1()`                         |
| `DynamicDiffusion` / `FixedDiffusion` | `solver_dynamic()` or `solver_mle()` |
| `IWP(diffusion=x^2)` |  `prior_wiener_integrated(output_scale=x)`                                       |
| Filtering and smoothing via `smooth=true/false`      | `strategy_filter`, `strategy_smoother_fixedpoint`, `strategy_smoother_fixedinterval`    |


Both libraries are evolving, and these translation guides may not be up-to-date. 
Consult each libraries' latest API documentation when in doubt.



## Transitioning from ProbNum (Python, Numpy)

[ProbNum](https://probnum.readthedocs.io/en/latest/) is a general probabilistic numerics library based on Numpy. Probdiffeq specializes in IVP solvers using pure JAX, offering:

* Greater efficiency for ODE problems because of JAX (e.g. jit)
* Probdiffeq implements more mature solvers. The algorithms are generally faster (eg state-space model factorisations, improved adaptive step-size selection)
* Probdiffeq offers more solvers and somewhat richer outputs (sampling, marginal likelihoods, etc.).



## Transitioning from Diffrax

[Diffrax](https://docs.kidger.site/diffrax/) is a JAX-based library for differential equations. The key difference is that Diffrax's solvers are non-probabilistic; Probdiffeq solvers are probabilistic. Approximate solver mapping:

| Diffrax                     | ProbDiffEq Equivalent                                     |
|-----------------------------|-----------------------------------------------------------|
| `Heun()`, `Midpoint()`      | Track $n=2$ Taylor coefficients and use `constraint_ode_ts0()`.  |
| `Tsit5()`, `Dopri5()`       | Track $n=4$ Taylor coefficients instead.                               |
| `Dopri8()`                   | Track $n=5, 6, 7$ Taylor coefficients instead; `constraint_ode_ts1()` and `solver_dynamic()` recommended but not required |
| `Kvaerno3()`, `Kvaerno5()`   | Track $n=2,3,4$ Taylor coefficients and use `constraint_ode_ts1()`         |
| Other methods (e.g. SDE solvers)                | Work in progress                                          |




## General differences from other common ODE solvers (e.g., SciPy, jax.odeint)

* Probdiffeq's solutions are posterior distributions instead of point estimates, enabling uncertainty quantification and more sophisticated models (eg easy switch to second-order problems).
* Probdiffeq's solver modes are explicit: `simulate_terminal_values()`, and `solve_adaptive_save_at()` instead of a one-size-fits-all `solve()` method.
