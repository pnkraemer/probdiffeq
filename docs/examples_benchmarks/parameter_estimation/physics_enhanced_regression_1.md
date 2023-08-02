---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.7
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region -->
# Parameter estimation with Optax

**Time-series data and optimization with ``optax``**

We create some fake-observational data, compute the marginal likelihood of this fake data _under the ODE posterior_ (which is something you cannot do with non-probabilistic solvers!), and optimize the parameters with `optax`.


Tronarp, Bosch, and Hennig call this "physics-enhanced regression" ([link to paper](https://arxiv.org/abs/2202.01287)).
<!-- #endregion -->

```python
import functools

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from diffeqzoo import backend, ivps
from jax.config import config

from probdiffeq import ivpsolve, solution
from probdiffeq.doc_util import notebook
from probdiffeq.ivpsolvers import uncalibrated
from probdiffeq.statespace import recipes
from probdiffeq.strategies import smoothers
```

```python
plt.rcParams.update(notebook.plot_config())

if not backend.has_been_selected:
    backend.select("jax")  # ivp examples in jax

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")
```

Create a problem and some fake-data:

```python
f, u0, (t0, t1), f_args = ivps.lotka_volterra()
f_args = jnp.asarray(f_args)

parameter_true = f_args + 0.05
parameter_guess = f_args


@jax.jit
def vf(y, t, p):
    return f(y, *p)


ts = jnp.linspace(t0, t1, endpoint=True, num=100)

impl = recipes.ts0_iso(num_derivatives=1)
strategy = smoothers.smoother(*impl)
solver = uncalibrated.solver(*strategy)


solution_true = ivpsolve.solve_fixed_grid(
    vf,
    initial_values=(u0,),
    grid=ts,
    solver=solver,
    parameters=parameter_true,
    output_scale=10.0,
)
data = solution_true.u
plt.plot(ts, data, "P-")
plt.show()
```

We make an initial guess, but it does not lead to a good data fit:

```python
solution_guess = ivpsolve.solve_fixed_grid(
    vf,
    initial_values=(u0,),
    grid=ts,
    solver=solver,
    parameters=parameter_guess,
    output_scale=10.0,
)
plt.plot(ts, data, color="k", linestyle="solid", linewidth=6, alpha=0.125)
plt.plot(ts, solution_guess.u)
plt.show()
```

Use the probdiffeq functionality to compute a parameter-to-data fit function.

This incorporates the likelihood of the data under the distribution induced by the probabilistic ODE solution (which was generated with the current parameter guess).

```python
def param_to_log_likelihood(parameters_, u0_, ts_, solver_, vf_, data_, obs_stdev=1e-1):
    sol_ = ivpsolve.solve_fixed_grid(
        vf_,
        initial_values=(u0_,),
        grid=ts_,
        solver=solver_,
        parameters=parameters_,
        output_scale=10.0,
    )

    observation_std = jnp.ones_like(ts_) * obs_stdev
    return -1.0 * solution.log_marginal_likelihood(
        observation_std=observation_std,
        u=data_,
        posterior=sol_.posterior,
        strategy=solver_.strategy,
    )


parameter_to_data_fit = jax.jit(
    functools.partial(
        param_to_log_likelihood, solver_=solver, ts_=ts, vf_=vf, u0_=u0, data_=data
    )
)

sensitivities = jax.jit(jax.grad(parameter_to_data_fit))
```

We can differentiate the function forward- and reverse-mode (the latter is possible because we use fixed steps)

```python
parameter_to_data_fit(parameter_guess)
sensitivities(parameter_guess)
```

Now, enter optax: build an optimizer, and optimise the parameter-to-model-fit function. The following is more or less taken from the [optax-documentation](https://optax.readthedocs.io/en/latest/optax-101.html).

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


optim = optax.adam(learning_rate=1e-2)
update_fn = build_update_fn(optimizer=optim, loss_fn=parameter_to_data_fit)
```

```python
p = parameter_guess
state = optim.init(p)

chunk_size = 10
for i in range(chunk_size):
    for _ in range(chunk_size):
        p, state = update_fn(p, state)

    print(f"After {(i+1)*chunk_size} iterations:", p)
```

The solution looks much better:

```python
solution_wrong = ivpsolve.solve_fixed_grid(
    vf, initial_values=(u0,), grid=ts, solver=solver, parameters=p, output_scale=1.0
)
plt.plot(ts, data, color="k", linestyle="solid", linewidth=6, alpha=0.125)
plt.plot(ts, solution_wrong.u)
plt.show()
```
