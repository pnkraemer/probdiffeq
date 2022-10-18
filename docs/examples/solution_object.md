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

TBD.

```python
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffeqzoo import backend, ivps
from jax.config import config

from odefilter import ivpsolve, recipes

backend.select("jax")  # ivp examples in jax


config.update("jax_enable_x64", True)
# config.update("jax_log_compiles", 1)
```

```python
# Make a problem
f, u0, (t0, t1), f_args = ivps.van_der_pol_first_order()


@jax.jit
def vector_field(y, t, p):
    return f(y, *p)


# Make a solver
ekf0, info_op = recipes.dynamic_ekf1(ode_dimension=2)
```

```python
%%time
solution = ivpsolve.solve(
    vector_field,
    initial_values=(u0,),
    t0=t0,
    t1=t1,
    solver=ekf0,
    info_op=info_op,
    parameters=f_args,
)
```

```python
# Look at the solution
print(len(solution))
```

```python
plt.plot(solution.t, solution.u, ".-")
plt.ylim((-5, 5))
plt.show()
```

```python
plt.semilogy(solution.t[:-1], jnp.diff(solution.t))
plt.show()
```
