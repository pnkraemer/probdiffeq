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

# Exploring the solution object

ODE filters are probabilistic numerical algorithms, which means they compute probability distributions over possible solutions instead of simple point estimates.
A probabilistic description is much richer than a non-probabilistic description, so the solution objects returned by the ODE filter are a lot of fun.

```python tags=[]
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffeqzoo import backend, ivps
from jax.config import config

from odefilter import ivpsolve, recipes

backend.select("jax")  # ivp examples in jax


config.update("jax_enable_x64", True)
```

```python tags=[]
# Make a problem
f, u0, (t0, t1), f_args = ivps.sir()


@jax.jit
def vector_field(y, *, t, p):
    return f(y, *p)


# Make a solver
solver, info_op = recipes.ekf1(ode_dimension=3, num_derivatives=2)
```

```python tags=[]
%%time
solution = ivpsolve.solve(
    vector_field,
    initial_values=(u0,),
    t0=t0,
    t1=t1,
    solver=solver,
    info_op=info_op,
    parameters=f_args,
    atol=1e-1,
    rtol=1e-1,
)
```

We can access elements of the solution.

```python tags=[]
print(len(solution))
print(solution[-1])
```

We can plot an estimate of the solution.

```python tags=[]
plt.plot(solution.t, solution.u, ".-")
plt.show()
```

But we can also look at the underlying distribution.
For starters, maybe we want to compute the marginal distribution over the solution away from
the grid points. This is similar to dense output, but waaaay cooler: there is not _one_ way of dense output with
probabilistic solvers, but there are

* marginals on off-grid points (most similar to traditional dense output)
* joint distributions on grid points and away from the grid points
* joint samples from the posterior

and many more options.
Look at this:

```python tags=[]
ts = jnp.linspace(t0+1e-4, t1-1e-3, num=400, endpoint=True)
dense = solver.offgrid_marginals_searchsorted(ts=ts, solution=solution)

fig, ax = plt.subplots(
    nrows=2, sharex=True, tight_layout=True
)

for i in [0, 1, 2]:  # ["S", "I", "R"]
    ms = dense.marginals.mean[:, i]
    ls = dense.marginals.cov_sqrtm_lower[:, i, :]
    stds = jnp.sqrt(jnp.einsum("jn,jn->j", ls, ls))

    ax[0].plot(dense.t, ms)
    ax[0].fill_between(dense.t, ms - 1.96 * stds, ms + 1.96 * stds, alpha=0.3)
    ax[0].set_ylabel("Posterior credible intervals")

    ax[1].semilogy(dense.t, stds)
    ax[1].set_ylabel("Standard deviation")

ax[1].set_xlabel("Time")
plt.show()
```

Stay tuned for more.
