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

# Physics-enhanced regression

```python
import jax
import jax.numpy as jnp
import jax.random
import matplotlib.pyplot as plt
from diffeqzoo import backend, ivps
from jax.config import config

from odefilter import dense_output, ivpsolve, solvers
from odefilter.implementations import dense
from odefilter.strategies import filters, smoothers

config.update("jax_enable_x64", True)

if not backend.has_been_selected:
    backend.select("jax")
```

```python
f, u0, (t0, t1), f_args = ivps.seir()
f_args = jnp.asarray(f_args)


@jax.jit
def vf(y, t, p):
    return f(y, *p)
```

```python
# make data

ts = jnp.linspace(t0, t1, endpoint=True, num=3)

ek1 = solvers.MLESolver(
    strategy=filters.Filter(
        extrapolation=dense.IBM.from_params(ode_dimension=4),
        correction=dense.TaylorFirstOrder(ode_dimension=4),
    )
)

solution_true = ivpsolve.simulate_terminal_values(
    vf, initial_values=(u0,), t0=t0, t1=t1, solver=ek1, parameters=f_args + 0.05
)
data = solution_true.u
print(data)
```

```python


```

```python
# Initial guess

solution_wrong = ivpsolve.simulate_terminal_values(
    vf, initial_values=(u0,), t0=t0, t1=t1, solver=ek1, parameters=f_args
)
print(solution_wrong.u)
```

```python

```

```python
@jax.jit
def param_to_nmll(p):
    observation_std = jnp.ones_like(ts) * 0.1
    solution_wrong = ivpsolve.simulate_terminal_values(
        vf, initial_values=(u0,), t0=t0, t1=t1, solver=ek1, parameters=p
    )

    m_obs = ek1.strategy.correction._select_derivative(solution_wrong.marginals.mean, 0)
    l_obs = ek1.strategy.correction._select_derivative_vect(
        solution_wrong.marginals.cov_sqrtm_lower, 0
    )

    return (data - solution_wrong.u) @ (data - solution_wrong.u)


#     return (solution_wrong.u[-1, ...] - 20.) @ (solution_wrong.u[-1, ...] - 20.)
#     return dense_output.negative_marginal_log_likelihood(
#         observation_std=observation_std, u=data, solution=solution_wrong, solver=ek1
#     )
```

```python
param_to_nmll(f_args)
```

```python
df = jax.jit(jax.jacfwd(param_to_nmll))
```

```python
f0 = f_args
f1 = f0 - 1e-1 * df(f0)
print(f1, f_args)
f1 = f1 - 1e-1 * df(f1)
print(f1, f_args)
f1 = f1 - 1e-1 * df(f1)
print(f1, f_args)
```

```python

```

```python

```

```python

```

```python

```
