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
import functools

from odefilter import dense_output, ivpsolve, solvers
from odefilter.implementations import dense, isotropic
from odefilter.strategies import filters, smoothers

config.update("jax_enable_x64", True)

if not backend.has_been_selected:
    backend.select("jax")
```

```python
f, u0, (t0, t1), f_args = ivps.lotka_volterra()
f_args = jnp.asarray(f_args)

parameter_true = f_args + 0.1
parameter_guess = f_args
@jax.jit
def vf(y, t, p):
    return f(y, *p)
```

```python
# make data

ts = jnp.linspace(t0, t1, endpoint=True, num=50)

strategy = smoothers.Smoother(
    extrapolation=isotropic.IsoIBM.from_params(num_derivatives=1),
)
solver = solvers.Solver(strategy=strategy, output_scale_sqrtm=1.)


solution_true = ivpsolve.solve_fixed_grid(
    vf, initial_values=(u0,), ts=ts, solver=solver, parameters=parameter_true
)
data = solution_true.u
print(data[::2])
```

```python

```

```python
# Initial guess

solution_wrong = ivpsolve.solve_fixed_grid(
    vf, initial_values=(u0,), ts=ts, solver=solver, parameters=parameter_guess
)
print(solution_wrong.u[::2])
```

```python

```

```python
def data_likelihood(parameters, u0, ts, solver, vf, data):
    solution_wrong = ivpsolve.solve_fixed_grid(
        vf, initial_values=(u0,), ts=ts, solver=solver, parameters=parameters
    )

    observation_std = jnp.ones_like(ts) * 1e-4
    return dense_output.negative_marginal_log_likelihood(
        observation_std=observation_std, u=data, solution=solution_wrong, solver=solver
    )


parameter_to_solution = functools.partial(
    data_likelihood, solver=solver, ts=ts, vf=vf, u0=u0, data=data
)
sensitivity = jax.jit(jax.grad(parameter_to_solution))

```

```python
parameter_to_solution(parameter_guess)
sensitivity(parameter_guess)
```

```python

```

```python
f1 = parameter_guess
lrate = 1e-7
for i in range(100):
    for _ in range(100):
        f1 = f1 - lrate * sensitivity(f1)
    print(f"{i+1}00 iterations:", f1, parameter_true)


print(f1, parameter_true)

```

```python

```

```python

```

```python

```

```python

```
