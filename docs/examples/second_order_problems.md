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

# Simulate second-order systems

```python
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffeqzoo import backend, ivps
from jax.config import config

from odefilter import ivpsolve, recipes

config.update("jax_enable_x64", True)
backend.select("jax")
```

Quick refresher: first-order ODEs

```python
f, u0, (t0, t1), f_args = ivps.three_body_restricted_first_order()


@jax.jit
def vf_1(y, t, p):
    return f(y, *p)


ek0_1, info_op_1 = recipes.dynamic_isotropic_ekf0(num_derivatives=5)
ts = jnp.linspace(t0, t1, endpoint=True, num=500)
```

```python
%%time

solution = ivpsolve.simulate_checkpoints(
    vf_1,
    initial_values=(u0,),
    ts=ts,
    solver=ek0_1,
    info_op=info_op_1,
    atol=1e-5,
    rtol=1e-5,
    parameters=f_args,
)
```

```python
plt.title((solution.u.shape, jnp.linalg.norm(solution.u[-1, ...] - u0)))
plt.plot(solution.u[:, 0], solution.u[:, 1])
plt.show()
```

```python
f, (u0, du0), (t0, t1), f_args = ivps.three_body_restricted()


@jax.jit
def vf_2(y, dy, t, p):
    return f(y, dy, *p)


# One derivative more than above because we don't transform to first order
ek0_2, info_op_2 = recipes.dynamic_isotropic_ekf0(num_derivatives=6, ode_order=2)
ts = jnp.linspace(t0, t1, endpoint=True, num=500)
```

```python
%%time

solution = ivpsolve.simulate_checkpoints(
    vf_2,
    initial_values=(u0, du0),
    ts=ts,
    solver=ek0_2,
    info_op=info_op_2,
    atol=1e-5,
    rtol=1e-5,
    parameters=f_args,
)
```

```python
plt.title((solution.u.shape, jnp.linalg.norm(solution.u[-1, ...] - u0)))
plt.plot(solution.u[:, 0], solution.u[:, 1])
plt.show()
```

The second-order solver is faster and way more accurate.
