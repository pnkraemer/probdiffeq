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

# # Neural ODEs
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

from probdiffeq import ivpsolve, ivpsolvers, stats
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
        ibm, ssm = ivpsolvers.prior_ibm(tcoeffs, ssm_fact="isotropic")
        ts0 = ivpsolvers.correction_ts0(ssm=ssm)
        strategy = ivpsolvers.strategy_smoother(ssm=ssm)
        solver_ts0 = ivpsolvers.solver(strategy, prior=ibm, correction=ts0, ssm=ssm)
        init = solver_ts0.initial_condition()

        sol = ivpsolve.solve_fixed_grid(
            lambda *a, **kw: vf(*a, **kw, p=parameters),
            init,
            grid=grid,
            solver=solver,
            ssm=ssm,
        )

        observation_std = jnp.ones_like(grid) * standard_deviation
        marginal_likelihood = stats.log_marginal_likelihood(
            data[:, None],
            standard_deviation=observation_std,
            posterior=sol.posterior,
            ssm=sol.ssm,
        )
        return -1 * marginal_likelihood

    return loss_fn


def build_update_fn(*, optimizer, loss_fn):
    """Build a function for executing a single step in the optimization."""

    @jax.jit
    def update(params, opt_state):
        """Update the optimiser state."""
        _loss, grads = jax.value_and_grad(loss_fn)(params)
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
tcoeffs = (u0, vf(u0, t=t0, p=f_args))
ibm, ssm = ivpsolvers.prior_ibm(tcoeffs, output_scale=1.0, ssm_fact="isotropic")
ts0 = ivpsolvers.correction_ts0(ssm=ssm)
strategy = ivpsolvers.strategy_smoother(ssm=ssm)
solver_ts0 = ivpsolvers.solver(strategy, prior=ibm, correction=ts0, ssm=ssm)
init = solver_ts0.initial_condition()

# +
sol = ivpsolve.solve_fixed_grid(
    lambda *a, **kw: vf(*a, **kw, p=f_args), init, grid=grid, solver=solver_ts0, ssm=ssm
)

plt.plot(sol.t, sol.u[0], ".-", label="Initial estimate")
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
        f"{(i + 1) * chunk_size**2}/{chunk_size**3} steps:",
        loss_fn(p),
    )

# +
plt.plot(sol.t, data, "-", linewidth=5, alpha=0.5, label="Data")
tcoeffs = (u0, vf(u0, t=t0, p=f_args))
ibm, ssm = ivpsolvers.prior_ibm(tcoeffs, output_scale=1.0, ssm_fact="isotropic")
ts0 = ivpsolvers.correction_ts0(ssm=ssm)
strategy = ivpsolvers.strategy_smoother(ssm=ssm)
solver_ts0 = ivpsolvers.solver(strategy, prior=ibm, correction=ts0, ssm=ssm)
init = solver_ts0.initial_condition()

sol = ivpsolve.solve_fixed_grid(
    lambda *a, **kw: vf(*a, **kw, p=p), init, grid=grid, solver=solver_ts0, ssm=ssm
)


plt.plot(sol.t, sol.u[0], ".-", label="Final guess")

tcoeffs = (u0, vf(u0, t=t0, p=f_args))
ibm, ssm = ivpsolvers.prior_ibm(tcoeffs, output_scale=1.0, ssm_fact="isotropic")
ts0 = ivpsolvers.correction_ts0(ssm=ssm)
strategy = ivpsolvers.strategy_smoother(ssm=ssm)
solver_ts0 = ivpsolvers.solver(strategy, prior=ibm, correction=ts0, ssm=ssm)
init = solver_ts0.initial_condition()

sol = ivpsolve.solve_fixed_grid(
    lambda *a, **kw: vf(*a, **kw, p=f_args), init, grid=grid, solver=solver_ts0, ssm=ssm
)
plt.plot(sol.t, sol.u[0], ".-", label="Initial guess")


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
