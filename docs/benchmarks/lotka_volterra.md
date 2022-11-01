---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Lotka-Volterra

Let's find the fastest solver of the Lotka--Volterra problem, a standard benchmark problem. It is low-dimensional, not stiff, and generally poses no major problems for any numerical solver.

```python
from functools import partial

import jax
import jax.experimental.ode
import jax.numpy as jnp
import matplotlib.pyplot as plt
from _benchmark_utils import most_recent_commit, plot_config, time
from diffeqzoo import backend, ivps
from jax import config
from tqdm import tqdm

from odefilter import controls, cubature, ivpsolve, solvers
from odefilter.implementations import dense
from odefilter.strategies import filters

# Nice-looking plots
plt.rcParams.update(plot_config())

# x64 precision
config.update("jax_enable_x64", True)

# IVP examples in JAX, not in NumPy
backend.select("jax")
```

```python
from odefilter import __version__ as odefilter_version  # noqa: E402

commit = most_recent_commit(abbrev=6)


print(f"Most recent commit:\n\t{commit}")
print(f"odefilter version:\n\t{odefilter_version}")
print()
jax.print_environment_info()
```

This is the problem:

```python
# Make a problem
f, u0, (t0, t1), f_args = ivps.lotka_volterra(time_span=(0.0, 50.0))
ODE_NAME = "Lotka-Volterra"


@jax.jit
def vf(x, *, t, p):
    return f(x, *p)


# Compile
vf(u0, t=t0, p=f_args)

ts = jnp.linspace(t0, t1, num=200)
odeint_solution = jax.experimental.ode.odeint(
    lambda u, t, *p: vf(u, t=t, p=p), u0, ts, *f_args, atol=1e-12, rtol=1e-12
)
ys_reference = odeint_solution[-1, :]

fig, ax = plt.subplots(figsize=(5, 2))
ax.plot(ts, odeint_solution, marker="None")
plt.show()
```

## Internal solvers
Let's start with finding the fastest ODE filter.

### Terminal-value simulation

```python
def solver_to_solve(solver, **kwargs):
    return jax.jit(partial(_solve, solver=solver))


def _solve(*, solver, tol):
    solution = ivpsolve.simulate_terminal_values(
        vf,
        initial_values=(u0,),
        t0=t0,
        t1=t1,
        parameters=f_args,
        solver=solver,
        atol=1e-3 * tol,
        rtol=tol,
        control=controls.ClippedProportionalIntegral(),
    )
    diff = (solution.u - ys_reference) / (1e-5 + ys_reference)
    return jnp.linalg.norm(diff) / jnp.sqrt(diff.size)
```

```python
def workprecision(*, solve_fns, tols, **kwargs):
    results = {}
    for solve_fn, label in tqdm(solve_fns):

        times, errors = [], []

        for rtol in tols:

            def bench():
                return solve_fn(tol=rtol)

            t, error = time(bench, **kwargs)

            times.append(t)
            errors.append(error)

        results[label] = (times, errors)
    return results
```

```python
d = u0.shape[0]


def correction_to_solver(correction, num_derivatives):
    return solver_to_solve(
        solvers.MLESolver(
            strategy=filters.Filter(
                extrapolation=dense.IBM.from_params(
                    ode_dimension=d, num_derivatives=num_derivatives
                ),
                correction=correction,
            )
        )
    )


ekf1_correction = dense.TaylorLinear(ode_dimension=d)
ckf1_correction = dense.MomentMatch(
    cubature=cubature.SphericalCubatureIntegration.from_params(ode_dimension=d),
    ode_dimension=d,
)
ukf1_correction = dense.MomentMatch(
    cubature=cubature.UnscentedTransform.from_params(ode_dimension=d, r=1.0),
    ode_dimension=d,
)
ghkf1_correction = dense.MomentMatch(
    cubature=cubature.GaussHermite.from_params(ode_dimension=d, degree=3),
    ode_dimension=d,
)

num_derivatives = 5
ekf1 = correction_to_solver(ekf1_correction, num_derivatives)
ckf1 = correction_to_solver(ckf1_correction, num_derivatives)
ukf1 = correction_to_solver(ukf1_correction, num_derivatives)
ghkf1 = correction_to_solver(ghkf1_correction, num_derivatives)

solve_fns = [
    (ekf1, f"EKF1({num_derivatives})"),
    (ckf1, f"CKF1({num_derivatives})"),
    (ukf1, f"UKF1({num_derivatives})"),
    (ghkf1, f"GHKF1({num_derivatives})"),
]
```

```python
%%time

tolerances = 0.1 ** jnp.arange(1.0, 11.0, step=1.0)

results = workprecision(solve_fns=solve_fns, tols=tolerances, number=3, repeat=3)
```

```python
fig, ax = plt.subplots(figsize=(5, 5))

for solver in results:
    times, errors = results[solver]
    ax.loglog(errors, times, label=solver)

ax.grid("both")
ax.set_xticks(0.1 ** (jnp.arange(1.0, 12.0, step=1.0)))
ax.set_yticks(0.1 ** (jnp.arange(1.0, 4.5, step=0.5)))
ax.set_title(f"Internal solvers [{ODE_NAME}]")
ax.set_xlabel("Precision [RMSE, absolute]")
ax.set_ylabel("Work [wall time, s]")
ax.legend()
plt.show()
```

If these results show one thing, then the fact that it is worth building a specialised ODE solver
for simulate_terminal_values(). The smoothing-based solvers compute extra factors that are just not needed for terminal-value simulation, and the extra factors turn out to be quite expensive. Every smoother was slower than its filtering-equivalent.

We can observe more:
* Dynamic calibration seems to perform at most as good as non-dynamic calibration. (Except for low order, low-precision EKF1, where the dynamic calibration seems to help. But even with the dynamic calibration is the low-order EKF1 one of the slowest solvers.)
* Low precision is best achieved with an isotropic EKF0(3). High precision is best achieved with an EKF1(8). The middle ground is better covered by an isotropic EKF0(5) than an EKF1(5).
* The cubature filters are more expensive than the extended filters (because cubature-linearisation costs more than Taylor-linearisation)



## External solvers
TBD.
