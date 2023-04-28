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

# Simulate second-order systems

```python
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffeqzoo import backend, ivps
from jax.config import config

from probdiffeq import ivpsolve, ivpsolvers
from probdiffeq.doc_util import notebook
from probdiffeq.statespace import recipes
from probdiffeq.strategies import filters
```

```python
plt.rcParams.update(notebook.plot_config())

if not backend.has_been_selected:
    backend.select("jax")  # ivp examples in jax

config.update("jax_platform_name", "cpu")
```

Quick refresher: first-order ODEs

```python
f, u0, (t0, t1), f_args = ivps.three_body_restricted_first_order()


@jax.jit
def vf_1(y, t, p):
    return f(y, *p)


ts0_1 = ivpsolvers.solver_mle(*filters.filter(*recipes.ts0_iso()))
ts = jnp.linspace(t0, t1, endpoint=True, num=500)
```

```python
%%time

solution = ivpsolve.solve_and_save_at(
    vf_1,
    initial_values=(u0,),
    save_at=ts,
    solver=ts0_1,
    atol=1e-5,
    rtol=1e-5,
    parameters=f_args,
)
```

```python
plt.title((solution.u.shape, jnp.linalg.norm(solution.u[-1, ...] - u0)))
plt.plot(solution.u[:, 0], solution.u[:, 1], marker=".")
plt.show()
```

The default configuration assumes that the ODE to be solved is of first order.
Now, the same game with a second-order ODE

```python
f, (u0, du0), (t0, t1), f_args = ivps.three_body_restricted()


@jax.jit
def vf_2(y, dy, t, p):
    return f(y, dy, *p)


# One derivative more than above because we don't transform to first order
implementation = recipes.ts0_iso(ode_order=2, num_derivatives=5)
ts0_2 = ivpsolvers.solver_mle(*filters.filter(*implementation))
ts = jnp.linspace(t0, t1, endpoint=True, num=500)
```

```python
%%time

solution = ivpsolve.solve_and_save_at(
    vf_2,
    initial_values=(u0, du0),
    save_at=ts,
    solver=ts0_2,
    atol=1e-5,
    rtol=1e-5,
    parameters=f_args,
)
```

```python
plt.title((solution.u.shape, jnp.linalg.norm(solution.u[-1, ...] - u0)))
plt.plot(solution.u[:, 0], solution.u[:, 1], marker=".")
plt.show()
```

The results are indistinguishable from the plot. While the runtimes of both solvers are similar, the error of the second-order solver is much lower. 

See the benchmarks for more quantitative versions of this statement.
