---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# High Irradiance Response (HIRES)

The HIRES problem is a common stiff differential equation.

```python
import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy.integrate
from diffeqzoo import backend, ivps
from jax import config

from probdiffeq import ivpsolvers
from probdiffeq.doc_util import benchmark, info, notebook, workprecision
from probdiffeq.implementations import recipes
from probdiffeq.strategies import filters
```

```python
# x64 precision
config.update("jax_enable_x64", True)

# CPU
config.update("jax_platform_name", "cpu")

# IVP examples in JAX
if not backend.has_been_selected:
    backend.select("jax")

# Nice-looking plots
plt.rcParams.update(notebook.plot_config())

# Which version of the softwares are we using?
info.print_info()
```

```python
f, u0, (t0, t1), f_args = ivps.hires()


@jax.jit
def vf(x, *, t, p):
    return f(x, *f_args)


problem = benchmark.FirstOrderIVP(vector_field=vf, initial_values=(u0,), t0=t0, t1=t1)

problem_scipy = problem.to_scipy(t_eval=[t0, t1])
problem_diffrax = problem.to_diffrax()
problems = {
    "probdiffeq": problem,
    "diffrax": problem_diffrax,
    "scipy": problem_scipy,
}
```

```python
# Compute a reference solution
scipy_solution = scipy.integrate.solve_ivp(
    *problem_scipy.args,
    atol=1e-13,
    rtol=1e-13,
    method="BDF",
)

# Select all remaining problem parameters
rtols = 0.1 ** jnp.arange(1.0, 9.0, step=1.0)
atols = 1e-3 * rtols
num_repeats = 10
error_fn = benchmark.relative_rmse(solution=scipy_solution.y.T[-1, :])

solve_fn_2nd = benchmark.probdiffeq_terminal_values()
solve_fn = benchmark.probdiffeq_terminal_values()
solve_fn_diffrax = benchmark.diffrax_terminal_values()
solve_fn_scipy = benchmark.scipy_terminal_values()
solve_fns = {
    "probdiffeq": solve_fn,
    "diffrax": solve_fn_diffrax,
    "scipy": solve_fn_scipy,
}

# Bundle the problem setup
problem_config = workprecision.ProblemConfig(
    label="HIRES (terminal-value simulation)",
    problems=problems,
    error_fn=error_fn,
    solve_fns=solve_fns,
    atols=atols,
    rtols=rtols,
    repeat=num_repeats,
    error_unit="RMSE, relative",
)
```

```python
fig, ax = plt.subplots(figsize=(4, 3))
colors = ["C" + str(i) for i in range(7)]
y = scipy_solution.y.T
plt.plot(scipy_solution.t, scipy_solution.y.T, marker="None")
plt.show()
```

```python
# Some helper functions


def impl_to_method_config(impl, *, label):
    return strategy_to_method_config(filters.Filter(impl), label=label)


def strategy_to_method_config(strategy, *, label):
    solver = ivpsolvers.DynamicSolver(strategy)
    return workprecision.MethodConfig(
        method={"solver": solver},
        label="ProbDiffEq: " + label,
        key="probdiffeq",
        jit=True,
        plotting_kwargs={"color": "C0"},
    )


def diffrax_method_config(solver, label):
    return workprecision.MethodConfig(
        method={"solver": solver, "max_steps": 10_000},
        label="Diffrax: " + label,
        jit=True,
        key="diffrax",
        plotting_kwargs={"color": "C3"},
    )


def scipy_method_config(method):
    return workprecision.MethodConfig(
        method={"method": method},
        label="SciPy: " + method,
        jit=False,
        key="scipy",
        plotting_kwargs={"color": "C2"},
    )
```

```python
num_derivatives_low = 3
num_derivatives = 5

ts1_low = recipes.DenseTS1.from_params(
    ode_shape=u0.shape, num_derivatives=num_derivatives_low
)
ts1 = recipes.DenseTS1.from_params(ode_shape=u0.shape, num_derivatives=num_derivatives)

# Methods
methods = [
    diffrax_method_config(solver=diffrax.Kvaerno3(), label="Kvaerno3()"),
    diffrax_method_config(solver=diffrax.Kvaerno5(), label="Kvaerno5()"),
    scipy_method_config(method="LSODA"),
    scipy_method_config(method="Radau"),
    impl_to_method_config(ts1_low, label=f"DenseTS1({num_derivatives_low})"),
    impl_to_method_config(ts1, label=f"DenseTS1({num_derivatives})"),
]
```

```python
results = workprecision.create(problem=problem_config, methods=methods)
```

```python
fig, ax = plt.subplots(figsize=(6, 4))
fig, ax = workprecision.plot(
    results=results,
    fig=fig,
    ax=ax,
    title=problem_config.label,
    xlabel_unit=problem_config.error_unit,
)
plt.show()
```
