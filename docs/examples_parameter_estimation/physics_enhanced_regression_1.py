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

# # Estimating ODE parameters with Optax
#
# **Time-series data and optimization with ``optax``**
#
# We create some fake-observational data,
# compute the marginal likelihood of this fake data _under the ODE posterior_
# (which is something you cannot do with non-probabilistic solvers!),
# and optimize the parameters with `optax`.
#
#
# Tronarp, Bosch, and Hennig call this "physics-enhanced regression"
# ([link to paper](https://arxiv.org/abs/2202.01287)).

# +
"""Estimate ODE parameters with ProbDiffEq and Optax."""

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

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
# -


# Create a problem and some fake-data:

# +
f, u0, (t0, t1), f_args = ivps.lotka_volterra()
f_args = jnp.asarray(f_args)


@jax.jit
def vf(y, t, *, p):  # noqa: ARG001
    """Evaluate the Lotka-Volterra vector field."""
    return f(y, *p)


def solve(p):
    """Evaluate the parameter-to-solution map."""
    tcoeffs = (u0, vf(u0, t0, p=p))
    output_scale = 10.0
    ibm, ssm = ivpsolvers.prior_ibm(
        tcoeffs, output_scale=output_scale, ssm_fact="isotropic"
    )
    ts0 = ivpsolvers.correction_ts0(ssm=ssm)
    strategy = ivpsolvers.strategy_smoother(ssm=ssm)
    solver = ivpsolvers.solver(strategy, prior=ibm, correction=ts0, ssm=ssm)

    init = solver.initial_condition()
    return ivpsolve.solve_fixed_grid(
        lambda y, t: vf(y, t, p=p), init, grid=ts, solver=solver, ssm=ssm
    )


parameter_true = f_args + 0.05
parameter_guess = f_args


ts = jnp.linspace(t0, t1, endpoint=True, num=100)
solution_true = solve(parameter_true)
data = solution_true.u[0]
plt.plot(ts, data, "P-")
plt.show()
# -

# We make an initial guess, but it does not lead to a good data fit:

solution_guess = solve(parameter_guess)
plt.plot(ts, data, color="k", linestyle="solid", linewidth=6, alpha=0.125)
plt.plot(ts, solution_guess.u[0])
plt.show()


# Use the probdiffeq functionality to compute a parameter-to-data fit function.
#
# This incorporates the likelihood of the data under the distribution induced
# by the probabilistic ODE solution
# (which was generated with the current parameter guess).


# +
@jax.jit
def parameter_to_data_fit(parameters_, /, standard_deviation=1e-1):
    """Evaluate the data fit as a function of the parameters."""
    sol_ = solve(parameters_)
    return -1.0 * stats.log_marginal_likelihood(
        data,
        standard_deviation=jnp.ones_like(sol_.t) * standard_deviation,
        posterior=sol_.posterior,
        ssm=sol_.ssm,
    )


sensitivities = jax.jit(jax.grad(parameter_to_data_fit))
# -

# We can differentiate the function forward- and reverse-mode
# (the latter is possible because we use fixed steps)

parameter_to_data_fit(parameter_guess)
sensitivities(parameter_guess)


# Now, enter optax: build an optimizer,
# and optimise the parameter-to-model-fit function.
# The following is more or less taken from the
# [optax-documentation](https://optax.readthedocs.io/en/latest/optax-101.html).


# +
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


optim = optax.adam(learning_rate=1e-2)
update_fn = build_update_fn(optimizer=optim, loss_fn=parameter_to_data_fit)

# +
p = parameter_guess
state = optim.init(p)

chunk_size = 10
for i in range(chunk_size):
    for _ in range(chunk_size):
        p, state = update_fn(p, state)

    print(f"After {(i + 1) * chunk_size} iterations:", p)
# -

# The solution looks much better:

solution_better = solve(p)
plt.plot(ts, data, color="k", linestyle="solid", linewidth=6, alpha=0.125)
plt.plot(ts, solution_better.u[0])
plt.show()
