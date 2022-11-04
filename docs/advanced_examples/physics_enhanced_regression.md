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

Aka Phenrir (Fenrir).

```python
import functools

import jax
import jax.numpy as jnp
import jax.random
import matplotlib.pyplot as plt
from diffeqzoo import backend, ivps
from jax.config import config

from odefilter import dense_output, ivpsolve, solvers
from odefilter.implementations import isotropic
from odefilter.strategies import smoothers

config.update("jax_enable_x64", True)

if not backend.has_been_selected:
    backend.select("jax")
```

```python
f, u0, (t0, t1), f_args = ivps.lotka_volterra()
f_args = jnp.asarray(f_args)

parameter_true = f_args + 0.05
parameter_guess = f_args


@jax.jit
def vf(y, t, p):
    return f(y, *p)
```

```python
# make data

ts = jnp.linspace(t0, t1, endpoint=True, num=100)

strategy = smoothers.Smoother(
    extrapolation=isotropic.IsoIBM.from_params(num_derivatives=1),
)
solver = solvers.Solver(strategy=strategy, output_scale_sqrtm=10.0)


solution_true = ivpsolve.solve_fixed_grid(
    vf, initial_values=(u0,), ts=ts, solver=solver, parameters=parameter_true
)
data = solution_true.u
plt.plot(ts, data, "P-")
plt.show()
```

```python

```

```python
solution_wrong = ivpsolve.solve_fixed_grid(
    vf, initial_values=(u0,), ts=ts, solver=solver, parameters=parameter_guess
)
plt.plot(ts, data, color="k", linestyle="solid", linewidth=6, alpha=0.125)
plt.plot(ts, solution_wrong.u)
plt.show()
```

```python

```

```python
def data_likelihood(parameters_, u0_, ts_, solver_, vf_, data_):
    sol_ = ivpsolve.solve_fixed_grid(
        vf_, initial_values=(u0_,), ts=ts_, solver=solver_, parameters=parameters_
    )

    observation_std = jnp.ones_like(ts_) * 1e-1
    return dense_output.negative_marginal_log_likelihood(
        observation_std=observation_std, u=data_, solution=sol_, solver=solver_
    )


parameter_to_solution = jax.jit(
    functools.partial(
        data_likelihood, solver_=solver, ts_=ts, vf_=vf, u0_=u0, data_=data
    )
)
sensitivity = jax.jit(jax.grad(parameter_to_solution))
```

```python
%%time

parameter_to_solution(parameter_guess)
sensitivity(parameter_guess)
```

```python

```

```python
%%time

f1 = parameter_guess
lrate = 2e-6
block_size = 50
for i in range(block_size):
    for _ in range(block_size):
        f1 = f1 - lrate * sensitivity(f1)

    print(f"After {(i+1)*block_size} iterations:", f1, parameter_true)


print(f1, parameter_true)
```

```python
solution_wrong = ivpsolve.solve_fixed_grid(
    vf, initial_values=(u0,), ts=ts, solver=solver, parameters=f1
)
plt.plot(ts, data, color="k", linestyle="solid", linewidth=6, alpha=0.125)
plt.plot(ts, solution_wrong.u)
plt.show()
```
