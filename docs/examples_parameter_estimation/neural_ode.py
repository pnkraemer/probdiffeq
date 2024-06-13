# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Train a Neural ODE with Optax
#
# We can use the parameter estimation functionality
# to fit a neural ODE to a time series data set.

# +
"""Train a neural ODE with ProbDiffEq and Optax."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from diffeqzoo import backend, ivps

from probdiffeq import ivpsolve
from probdiffeq.impl import impl
from probdiffeq.solvers import components, solution, solvers, strategies
from probdiffeq.util.doc_util import notebook

# -

plt.rcParams.update(notebook.plot_style())
plt.rcParams.update(notebook.plot_sizes())

# +
if not backend.has_been_selected:
    backend.select("jax")  # ivp examples in jax

# Catch NaN gradients in CI
# Disable to improve speed
jax.config.update("jax_debug_nans", True)

jax.config.update("jax_platform_name", "cpu")
# -

impl.select("isotropic", ode_shape=(1,))

# To keep the problem nice and small, assume that the data set is a
# trigonometric function (which solve differential equations).

# +
grid = jnp.linspace(0, 1, num=100)
data = jnp.sin(5 * jnp.pi * grid)

plt.plot(grid, data, ".-", label="Data")
plt.legend()
plt.show()


# -


def build_loss_fn(vf, initial_values, solver, *, standard_deviation=1e-2):
    """Build a loss function from an ODE problem."""

    @jax.jit
    def loss_fn(parameters):
        """Loss function: log-marginal likelihood of the data."""
        tcoeffs = (*initial_values, vf(*initial_values, t=t0, p=parameters))
        init = solver.initial_condition(tcoeffs, output_scale=1.0)

        sol = ivpsolve.solve_fixed_grid(
            lambda *a, **kw: vf(*a, **kw, p=parameters), init, grid=grid, solver=solver
        )

        observation_std = jnp.ones_like(grid) * standard_deviation
        marginal_likelihood = solution.log_marginal_likelihood(
            data[:, None], standard_deviation=observation_std, posterior=sol.posterior
        )
        return -1 * marginal_likelihood

    return loss_fn


def build_update_fn(*, optimizer, loss_fn):
    """Build a function for executing a single step in the optimization."""

    @jax.jit
    def update(params, opt_state):
        """Update the optimiser state."""
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    return update


# ## Construct an MLP with tanh activation
#
# Let's start with the example given in the
# [implicit layers tutorial](http://implicit-layers-tutorial.org/neural_odes/).
# The vector field is provided by [DiffEqZoo](https://diffeqzoo.readthedocs.io/).

# +
f, u0, (t0, t1), f_args = ivps.neural_ode_mlp(layer_sizes=(2, 20, 1))


@jax.jit
def vf(y, *, t, p):
    """Evaluate the MLP."""
    return f(y, t, *p)


# Make a solver
ibm = components.ibm_adaptive(num_derivatives=1)
ts0 = components.ts0()
strategy = strategies.smoother_adaptive(ibm, ts0)
solver_ts0 = solvers.solver(strategy)

# +
tcoeffs = (u0, vf(u0, t=t0, p=f_args))
init = solver_ts0.initial_condition(tcoeffs, output_scale=1.0)

sol = ivpsolve.solve_fixed_grid(
    lambda *a, **kw: vf(*a, **kw, p=f_args), init, grid=grid, solver=solver_ts0
)

plt.plot(sol.t, sol.u, ".-", label="Initial estimate")
plt.plot(grid, data, ".-", label="Data")
plt.legend()
plt.show()
# -

# ## Set up a loss function and an optimiser
#
# Like in the other tutorials, we use [Optax](https://optax.readthedocs.io/en/latest/index.html).

loss_fn = build_loss_fn(vf=vf, initial_values=(u0,), solver=solver_ts0)
optim = optax.adam(learning_rate=2e-2)
update_fn = build_update_fn(optimizer=optim, loss_fn=loss_fn)

p = f_args
state = optim.init(p)
chunk_size = 25
for i in range(chunk_size):
    for _ in range(chunk_size**2):
        p, state = update_fn(p, state)
    print(
        "Negative log-marginal-likelihood after "
        f"{(i+1)*chunk_size**2}/{chunk_size**3} steps:",
        loss_fn(p),
    )

# +
plt.plot(sol.t, data, "-", linewidth=5, alpha=0.5, label="Data")
tcoeffs = (u0, vf(u0, t=t0, p=p))
init = solver_ts0.initial_condition(tcoeffs, output_scale=1.0)

sol = ivpsolve.solve_fixed_grid(
    lambda *a, **kw: vf(*a, **kw, p=p), init, grid=grid, solver=solver_ts0
)


plt.plot(sol.t, sol.u, ".-", label="Final guess")

tcoeffs = (u0, vf(u0, t=t0, p=f_args))
init = solver_ts0.initial_condition(tcoeffs, output_scale=1.0)

sol = ivpsolve.solve_fixed_grid(
    lambda *a, **kw: vf(*a, **kw, p=f_args), init, grid=grid, solver=solver_ts0
)
plt.plot(sol.t, sol.u, ".-", label="Initial guess")


plt.legend()
plt.show()
# -

# ## What's next
#
#
# The same example can be constructed with deep learning libraries
# such as [Equinox](https://docs.kidger.site/equinox/),
# [Haiku](https://dm-haiku.readthedocs.io/en/latest/), or
# [Flax](https://flax.readthedocs.io/en/latest/getting_started.html).
# To do so, define a corresponding vector field and a parameter set,
# build a new loss function and repeat.
#
#
