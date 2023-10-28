---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Simulate second-order systems

```python
"""Demonstrate how to solve second-order IVPs without transforming them first."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffeqzoo import backend, ivps
from jax.config import config

from probdiffeq import adaptive, ivpsolve
from probdiffeq.impl import impl
from probdiffeq.solvers import calibrated
from probdiffeq.solvers.strategies import filters
from probdiffeq.solvers.strategies.components import corrections, priors
from probdiffeq.taylor import autodiff
from probdiffeq.util.doc_util import notebook
```

```python
plt.rcParams.update(notebook.plot_style())
plt.rcParams.update(notebook.plot_sizes())
```

```python
if not backend.has_been_selected:
    backend.select("jax")  # ivp examples in jax

config.update("jax_platform_name", "cpu")
```

Quick refresher: first-order ODEs

```python
impl.select("isotropic", ode_shape=(4,))
f, u0, (t0, t1), f_args = ivps.three_body_restricted_first_order()


@jax.jit
def vf_1(y, t):  # noqa: ARG001
    """Evaluate the three-body problem as a first-order IVP."""
    return f(y, *f_args)


ibm = priors.ibm_adaptive(num_derivatives=4)
ts0 = corrections.ts0()
solver_1st = calibrated.mle(filters.filter_adaptive(ibm, ts0))
adaptive_solver_1st = adaptive.adaptive(solver_1st, atol=1e-5, rtol=1e-5)


tcoeffs = autodiff.taylor_mode_scan(lambda y: vf_1(y, t=t0), (u0,), num=4)
init = solver_1st.initial_condition(tcoeffs, output_scale=1.0)
```

```python
solution = ivpsolve.solve_and_save_every_step(
    vf_1, init, t0=t0, t1=t1, dt0=0.1, adaptive_solver=adaptive_solver_1st
)
```

```python
norm = jnp.linalg.norm((solution.u[-1, ...] - u0) / jnp.abs(1.0 + u0))
plt.title((solution.u.shape, norm))
plt.plot(solution.u[:, 0], solution.u[:, 1], marker=".")
plt.show()
```

The default configuration assumes that the ODE to be solved is of first order.
Now, the same game with a second-order ODE

```python
impl.select("isotropic", ode_shape=(2,))
f, (u0, du0), (t0, t1), f_args = ivps.three_body_restricted()


@jax.jit
def vf_2(y, dy, t):  # noqa: ARG001
    """Evaluate the three-body problem as a second-order IVP."""
    return f(y, dy, *f_args)


# One derivative more than above because we don't transform to first order
ibm = priors.ibm_adaptive(num_derivatives=4)
ts0 = corrections.ts0(ode_order=2)
solver_2nd = calibrated.mle(filters.filter_adaptive(ibm, ts0))
adaptive_solver_2nd = adaptive.adaptive(solver_2nd, atol=1e-5, rtol=1e-5)


tcoeffs = autodiff.taylor_mode_scan(lambda *ys: vf_2(*ys, t=t0), (u0, du0), num=3)
init = solver_2nd.initial_condition(tcoeffs, output_scale=1.0)
```

```python
solution = ivpsolve.solve_and_save_every_step(
    vf_2, init, t0=t0, t1=t1, dt0=0.1, adaptive_solver=adaptive_solver_2nd
)
```

```python
norm = jnp.linalg.norm((solution.u[-1, ...] - u0) / jnp.abs(1.0 + u0))
plt.title((solution.u.shape, norm))
plt.plot(solution.u[:, 0], solution.u[:, 1], marker=".")
plt.show()
```

The results are indistinguishable from the plot. While the runtimes of both solvers are similar, the error of the second-order solver is much lower. 

See the benchmarks for more quantitative versions of this statement.
