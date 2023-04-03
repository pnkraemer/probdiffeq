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

# Train a Neural ODE with Optax

We can use the parameter estimation functionality to fit a neural ODE to a time series data set.

```python
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from diffeqzoo import backend, ivps
from jax.config import config

from probdiffeq import solution, solution_routines, solvers
from probdiffeq.doc_util import notebook
from probdiffeq.implementations import recipes
from probdiffeq.strategies import smoothers
```

```python
plt.rcParams.update(notebook.plot_config())

if not backend.has_been_selected:
    backend.select("jax")  # ivp examples in jax

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")
```

To keep the problem nice and small, assume that the data set is a trigonometric function (which solve differential equations).

```python
grid = jnp.linspace(0, 1, num=100)
data = jnp.sin(5 * jnp.pi * grid)

plt.plot(grid, data, ".-", label="Data")
plt.legend()
plt.show()
```

```python
def build_loss_fn(vf, initial_values, obs_stdev=1e-2):
    """Build a loss function from an ODE problem."""

    @jax.jit
    def loss_fn(parameters):
        sol = solution_routines.solve_fixed_grid(
            vf,
            initial_values=initial_values,
            grid=grid,
            solver=solver,
            parameters=parameters,
        )

        observation_std = jnp.ones_like(grid) * obs_stdev
        return solution.negative_marginal_log_likelihood(
            observation_std=observation_std, u=data[:, None], solution=sol
        )

    return loss_fn
```

```python
def build_update_fn(*, optimizer, loss_fn):
    """Build a function for executing a single step in the optimization."""

    @jax.jit
    def update(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    return update
```

## Construct an MLP with tanh activation

Let's start with the example given in the [implicit layers tutorial](http://implicit-layers-tutorial.org/neural_odes/). The vector field is provided by [DiffEqZoo](https://diffeqzoo.readthedocs.io/).

```python
f, u0, (t0, t1), f_args = ivps.neural_ode_mlp(layer_sizes=(2, 20, 1))


@jax.jit
def vf(y, *, t, p):
    return f(y, t, *p)


# Make a solver
strategy = smoothers.Smoother(
    recipes.IsoTS0.from_params(num_derivatives=1),
)
solver = solvers.CalibrationFreeSolver(strategy, output_scale_sqrtm=1.0)
```

```python
sol = solution_routines.solve_fixed_grid(
    vf, initial_values=(u0,), grid=grid, solver=solver, parameters=f_args
)

plt.plot(sol.t, sol.u, ".-", label="Initial estimate")
plt.plot(grid, data, ".-", label="Data")
plt.legend()
plt.show()
```

## Set up a loss function and an optimiser

Like in the other tutorials, we use [Optax](https://optax.readthedocs.io/en/latest/index.html).

```python
loss_fn = build_loss_fn(vf=vf, initial_values=(u0,))
optim = optax.adam(learning_rate=1e-2)
update_fn = build_update_fn(optimizer=optim, loss_fn=loss_fn)
```

```python
p = f_args
state = optim.init(p)

chunk_size = 30
for i in range(chunk_size):
    for _ in range(chunk_size**2):
        p, state = update_fn(p, state)
    print(f"After {(i+1)*chunk_size**2}/{chunk_size**3} steps:", loss_fn(p))
```

```python
plt.plot(sol.t, data, "-", linewidth=5, alpha=0.5, label="Data")


sol = solution_routines.solve_fixed_grid(
    vf, initial_values=(u0,), grid=grid, solver=solver, parameters=p
)
plt.plot(sol.t, sol.u, ".-", label="Final guess")

sol = solution_routines.solve_fixed_grid(
    vf, initial_values=(u0,), grid=grid, solver=solver, parameters=f_args
)
plt.plot(sol.t, sol.u, ".-", label="Initial guess")


plt.legend()
plt.show()
```

<!-- #region -->
## What's next


The same example can be constructed with deep learning libraries such as [Equinox](https://docs.kidger.site/equinox/), [Haiku](https://dm-haiku.readthedocs.io/en/latest/), or [Flax](https://flax.readthedocs.io/en/latest/getting_started.html).
To do so, define a corresponding vector field and a parameter set, build a new loss function and repeat.


<!-- #endregion -->
