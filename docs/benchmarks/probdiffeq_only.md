---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Only probdiffeq-solvers

Let's find the fastest solver of the Lotka--Volterra problem, a standard benchmark problem. It is low-dimensional, not stiff, and generally poses no major problems for any numerical solver.

```python
import functools

import jax
import jax.experimental.ode
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffeqzoo import backend, ivps
from jax import config

from probdiffeq import controls, cubature, solution_routines, solvers
from probdiffeq.doc_util import benchmark, info, notebook, workprecision
from probdiffeq.implementations import recipes
from probdiffeq.strategies import filters, smoothers
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

This is the ODE problem:

```python
# Make a problem
f, u0, (t0, t1), f_args = ivps.lotka_volterra(time_span=(0.0, 50.0))


@jax.jit
def vf(x, *, t, p):
    return f(x, *p)


ts = jnp.linspace(t0, t1, num=250)
odeint_solution = jax.experimental.ode.odeint(
    lambda u, t, *p: vf(u, t=t, p=p), u0, ts, *f_args, atol=1e-12, rtol=1e-12
)


problem = (vf, (u0,), t0, t1)
problem_config = workprecision.ProblemConfig(
    label="Lotka-Volterra (terminal-value simulation)",
    problem=problem,
    error_fn=benchmark.relative_rmse(solution=odeint_solution[-1, :]),
    solve_fn=benchmark.probdiffeq_terminal_values(),
)
```

```python
print(problem_config)
```

```python
fig, ax = plt.subplots(figsize=(5, 1))
ax.plot(ts, odeint_solution, marker="None")
plt.show()
```

## Internal solvers
Let's start with finding the fastest probabilistic IVP solver.



### Which mode of linearization?

Should we linearize with a Taylor-approximation or by moment matching?

```python
def impl_to_method(impl, *, label):
    solver = solvers.MLESolver(filters.Filter(impl))
    return workprecision.MethodConfig(method=solver_to_method(solver), label=label)


def solver_to_method(solver):
    return {
        "parameters": f_args,
        "solver": solver,
        "control": controls.ProportionalIntegral(),
    }


def cubature_to_slr1(cubature, *, ode_shape):
    return recipes.DenseSLR1.from_params(
        ode_shape=ode_shape,
        cubature=cubature,
    )


# Different linearisation styles
ode_shape = u0.shape
ts1 = recipes.DenseTS1.from_params(ode_shape=ode_shape)
sci = cubature.ThirdOrderSpherical.from_params(input_shape=ode_shape)
ut = cubature.UnscentedTransform.from_params(input_shape=ode_shape, r=1.0)
gh = cubature.GaussHermite.from_params(input_shape=ode_shape, degree=3)
slr1_sci = cubature_to_slr1(sci, ode_shape=ode_shape)
slr1_ut = cubature_to_slr1(ut, ode_shape=ode_shape)
slr1_gh = cubature_to_slr1(gh, ode_shape=ode_shape)


# Methods
methods = [
    impl_to_method(impl=ts1, label="TS1()"),
    impl_to_method(impl=slr1_sci, label="SLR1(SCI)"),
    impl_to_method(impl=slr1_ut, label="SLR1(UT)"),
    impl_to_method(impl=slr1_gh, label="SLR1(GH)"),
]
```

```python
tolerances = 0.1 ** jnp.arange(1.0, 9.0, step=2.0)
results = workprecision.create(
    problem=problem_config, methods=methods, tolerances=tolerances, repeat=5
)
```

```python
fig, ax = plt.subplots(figsize=(5, 3))
fig, ax = workprecision.plot(
    results=results, fig=fig, ax=ax, title=problem_config.label
)
plt.show()

assert 0
```

The Taylor-series based method is more efficient. The cubature rule has little effect on the efficiency of the moment-matching solver.


### Which factorisation?

What is the performance difference between an `IsoTaylorZerothOrder`, a `BlockDiagTaylorZerothOrder`, and a `TaylorZerothOrder`?

```python
def solver(implementation):
    return solver_to_solve(
        solvers.MLESolver(strategy=filters.Filter(implementation=implementation))
    )


num_derivatives = 4
iso_solver = solver(recipes.IsoTS0.from_params(num_derivatives=num_derivatives))
batch_solver = solver(
    recipes.BlockDiagTS0.from_params(
        ode_shape=ode_shape, num_derivatives=num_derivatives
    )
)
dense_solver = solver(
    recipes.DenseTS0.from_params(ode_shape=ode_shape, num_derivatives=num_derivatives)
)


solve_fns = [
    (iso_solver, f"IsoTS0({num_derivatives})"),
    (batch_solver, f"BlockDiagTS0({num_derivatives})"),
    (dense_solver, f"DenseTS0({num_derivatives})"),
]
```

```python
# %#%time

results = workprecision_diagram(solve_fns=solve_fns)
```

```python
fig, ax = plt.subplots(figsize=(5, 3))
fig, ax = benchmark.workprecision_plot(
    results=results, fig=fig, ax=ax, ode_name=ODE_NAME
)
plt.show()
```

Unsurprisingly, the stronger the factorisation, the faster. The isotropic solver is the fastest, the batch-solver is second place, and the dense solver is the slowest (by far).
The isotropic solver and the dense solver actually compute the same posterior (the errors are vertically aligned), but at different speeds.


### Dynamic solvers or MLE solvers?

Are there differences between dynamic solvers and MLE solvers in terms of performance?
Let's try a comparison for high-order and low-order methods.

```python
def strategy_to_dynamic_solver(strategy):
    return solver_to_solve(solvers.DynamicSolver(strategy=strategy))


def strategy_to_mle_solver(strategy):
    return solver_to_solve(solvers.MLESolver(strategy=strategy))


filter_ts0_iso_low = filters.Filter(recipes.IsoTS0.from_params(num_derivatives=2))
filter_ts0_iso_medium = filters.Filter(recipes.IsoTS0.from_params(num_derivatives=3))
filter_ts0_iso_high = filters.Filter(recipes.IsoTS0.from_params(num_derivatives=5))

filter_ts1_low = filters.Filter(
    recipes.DenseTS1.from_params(ode_shape=ode_shape, num_derivatives=3)
)
filter_ts1_medium = filters.Filter(
    recipes.DenseTS1.from_params(ode_shape=ode_shape, num_derivatives=5)
)
filter_ts1_high = filters.Filter(
    recipes.DenseTS1.from_params(ode_shape=ode_shape, num_derivatives=8)
)


solve_fns = []
for strat, label in [
    (filter_ts0_iso_low, "IsoTS0(2)"),
    (filter_ts0_iso_medium, "IsoTS0(3)"),
    (filter_ts0_iso_high, "IsoTS0(5)"),
    (filter_ts1_low, "DenseTS1(3)"),
    (filter_ts1_medium, "DenseTS1(5)"),
    (filter_ts1_high, "DenseTS1(8)"),
]:
    dynamic_solver = strategy_to_dynamic_solver(strat)
    mle_solver = strategy_to_mle_solver(strat)
    solve_fns.append(
        [
            (dynamic_solver, f"DynamicSolver({label})"),
            (mle_solver, f"MLESolver({label})"),
        ]
    )
```

```python
# %#%time

results_all = [workprecision_diagram(solve_fns=fns) for fns in solve_fns]
```

```python
fig, axes = plt.subplots(figsize=(8, 5), nrows=2, ncols=3, sharex=True, sharey=True)

for ax, results in zip(axes.reshape((-1,)), results_all):
    fig, ax = benchmark.workprecision_plot(
        results=results, fig=fig, ax=ax, title=None, xlabel=None, ylabel=None
    )

for ax in axes:
    ax[0].set_ylabel("Work [wall time, s]")

for ax, string in zip(axes.T, ["Low order", "Medium order", "High order"]):
    ax[0].set_title(f"{string} solvers [{ODE_NAME}]")
    ax[1].set_xlabel("Precision [RMSE, absolute]")
plt.show()
```

For zeroth-order Taylor series, the choice between dynamic and non-dynamic solvers is fairly irrelevant (the non-dynamic solver seems to have a sliiiiiight edge over the dynamic solver, but that is not very apparent).
For first-order Taylor series, non-dynamic calibration is better, but the difference is minimal for low-order methods and low precision.


### Should I use a filter or a smoother?

```python
impl = recipes.IsoTS0.from_params()
filter_ = filters.Filter(impl)
smoother = smoothers.Smoother(impl)
fixpt_smoother = smoothers.FixedPointSmoother(impl)

solve_fns = [
    (strategy_to_mle_solver(filter_), "Filter()"),
    (strategy_to_mle_solver(smoother), "Smoother()"),
    (strategy_to_mle_solver(fixpt_smoother), "FixedPointSmoother()"),
]
```

```python
# %#%time

results = workprecision_diagram(solve_fns=solve_fns)
```

```python
fig, ax = plt.subplots(figsize=(5, 3))
fig, ax = benchmark.workprecision_plot(
    results=results, fig=fig, ax=ax, ode_name=ODE_NAME
)
plt.show()
```

The filter is the fastest, the smoother is second, and the fixed-point smoother is the slowest.
But this is unsurprising, because the smoother and the fixed-point smoother compute extra quantities that the filter ignores.
These extra quantities make the smoothing solutions a valid posterior distribution at all time-points, whereas the filtering distribution makes sense mostly at the terminal value.

But when the quantitity of interest is the terminal value, filters are the go-to strategy.




### Fastest methods

So what are the fastest solvers?
We know that filters are better than smoothers for terminal-value simulation.
We also know that isotropic, zeroth-order approximations are the fastest zeroth-order methods,
and that Taylor-series based linearisations outperform moment matching.
We know that the dynamic solver is superior over the non-dynamic solver for low order methods with first-order linearisation, and that the is preferrable in the other setups (sometimes only incrementally).

Let's compare the winning solvers to find the best one.

```python
def impl_to_dynamic_solver_filter(impl):
    return strategy_to_dynamic_solver(filters.Filter(impl))


def impl_to_mle_solver_filter(impl):
    return strategy_to_mle_solver(filters.Filter(impl))


num_low, num_medium, num_high = 3, 5, 8

ts0_iso_low = recipes.IsoTS0.from_params(num_derivatives=num_low)
ts0_iso_medium = recipes.IsoTS0.from_params(num_derivatives=num_medium)

ts1_low = recipes.DenseTS1.from_params(ode_shape=ode_shape, num_derivatives=num_low)
ts1_medium = recipes.DenseTS1.from_params(
    ode_shape=ode_shape, num_derivatives=num_medium
)
ts1_high = recipes.DenseTS1.from_params(ode_shape=ode_shape, num_derivatives=num_high)


slr1_high = recipes.DenseSLR1.from_params(ode_shape=ode_shape, num_derivatives=num_high)

solve_fns = [
    (impl_to_mle_solver_filter(ts0_iso_low), f"IsoTS0({num_low}), MLE"),
    (impl_to_mle_solver_filter(ts0_iso_medium), f"IsoTS0({num_medium}), MLE"),
    (impl_to_dynamic_solver_filter(ts1_low), f"DenseTS1({num_low}), Dynamic"),
    (impl_to_mle_solver_filter(ts1_medium), f"DenseTS1({num_medium}), MLE"),
    (impl_to_mle_solver_filter(ts1_high), f"DenseTS1({num_high}), MLE"),
    (impl_to_mle_solver_filter(slr1_high), f"DenseSLR1({num_high}), MLE"),
]
```

```python
# %#%time

results = workprecision_diagram(solve_fns=solve_fns)
```

```python
fig, ax = plt.subplots(figsize=(5, 3))
fig, ax = benchmark.workprecision_plot(
    results=results, fig=fig, ax=ax, ode_name=ODE_NAME
)

plt.show()
```

## Conclusion

If these results show one thing, then the fact that it is worth building a specialised ODE solver
for simulate_terminal_values(). The smoothing-based solvers compute extra factors that are just not needed for terminal-value simulation, and the extra factors turn out to be quite expensive. Every smoother was slower than its filtering-equivalent.

We can observe more:
* Dynamic calibration seems to perform at most as good as non-dynamic calibration. (Except for low order, low-precision EKF1, where the dynamic calibration seems to help. But even with the dynamic calibration is the low-order EKF1 one of the slowest solvers.)
* Low precision is best achieved with an isotropic EKF0(3). High precision is best achieved with an EKF1(8). The middle ground is better covered by an isotropic EKF0(5) than an EKF1(5).
* The cubature filters are more expensive than the extended filters (because cubature-linearisation costs more than Taylor-linearisation)
