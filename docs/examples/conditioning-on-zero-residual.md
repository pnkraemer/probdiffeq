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

```python
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffeqzoo import backend, ivps
from jax.config import config

from probdiffeq import dense_output, ivpsolve, solvers
from probdiffeq.implementations import recipes
from probdiffeq.strategies import smoothers

if not backend.has_been_selected:
    backend.select("jax")  # ivp examples in jax


config.update("jax_enable_x64", True)
```

```python
# Make a problem
f, u0, (t0, t1), f_args = ivps.fitzhugh_nagumo(time_span=(0.0, 1.0))


@jax.jit
def vector_field(y, *, t, p):
    return f(y, *p)


# Make a solver
solver = solvers.DynamicSolver(smoothers.Smoother(recipes.IsoTS0.from_params()))
```

```python
%%time
solution = ivpsolve.solve(
    vector_field,
    initial_values=(u0,),
    t0=t0,
    t1=t1,
    solver=solver,
    parameters=f_args,
)
```

```python
mesh = jnp.linspace(t0 + 1e-2, t1 - 1e-2, endpoint=True)
```

```python
u, marginals = dense_output.offgrid_marginals_searchsorted(
    ts=mesh, solution=solution, solver=solver
)
```

```python
m = marginals.hidden_state.mean
c_sqrtm = marginals.hidden_state.cov_sqrtm_lower
c = jnp.einsum("ijk,ikm->ijm", c_sqrtm, c_sqrtm)
```

```python
v = jnp.diagonal(c, axis1=1, axis2=2)
s = jnp.sqrt(v)[:, 0]
```

```python

```

```python
plt.plot(mesh, u)
plt.plot(mesh, u)
plt.semilogy(mesh, s)
plt.show()
```

```python

```

```python

```

```python

```
