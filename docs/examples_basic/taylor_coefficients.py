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

# # Taylor coefficients
#
# To build a probabilistic solver, we need to build a specific state-space model.
# To build this specific state-space model, we interact with Taylor coefficients.
# Here are some examples how Taylor coefficients
# play a role in Probdiffeq's solution routines.


# +
"""Demonstrate how central Taylor coefficient estimation is to Probdiffeq."""

import collections

import jax
import jax.numpy as jnp
from diffeqzoo import backend, ivps

from probdiffeq import ivpsolve, probdiffeq, taylor

if not backend.has_been_selected:
    backend.select("jax")  # ivp examples in jax

# -

# We start by defining an ODE.

# +
f, u0, (t0, t1), f_args = ivps.logistic()


def vf(*y, t):  # noqa: ARG001
    """Evaluate the vector field."""
    return f(*y, *f_args)


# -


# Here is a wrapper arounds Probdiffeq's solution routine.


# +
def solve(tc):
    """Solve the ODE."""
    init, prior, ssm = probdiffeq.prior_wiener_integrated(tc, ssm_fact="dense")
    ts0 = probdiffeq.constraint_ode_ts0(ssm=ssm)
    strategy = probdiffeq.strategy_smoother_fixedpoint(ssm=ssm)
    solver = probdiffeq.solver_mle(
        vf, strategy=strategy, prior=prior, constraint=ts0, ssm=ssm
    )
    ts = jnp.linspace(t0, t1, endpoint=True, num=10)
    errorest = probdiffeq.errorest_local_residual_cached(prior=prior, ssm=ssm)
    solve = ivpsolve.solve_adaptive_save_at(solver=solver, errorest=errorest)
    return solve(init, save_at=ts, atol=1e-2, rtol=1e-2)


# -

# It's time to solve some ODEs:

# +
tcoeffs = taylor.odejet_padded_scan(lambda *y: vf(*y, t=t0), [u0], num=2)
solution = solve(tcoeffs)
print(jax.tree.map(jnp.shape, solution))

# -

# The type of solution.u matches that of the initial condition.

# +

print(jax.tree.map(jnp.shape, tcoeffs))
print(jax.tree.map(jnp.shape, solution.u))


# -

# Anything that behaves like a list work.
# For example, we can use lists or tuples, but also named tuples.

# +

CustomTCoeffs = collections.namedtuple(
    "CustomTCoeffs", ["state", "velocity", "acceleration"]
)
tcoeffs = CustomTCoeffs(*tcoeffs)
solution = solve(tcoeffs)

print(jax.tree.map(jnp.shape, tcoeffs))
print(jax.tree.map(jnp.shape, solution))
print(jax.tree.map(jnp.shape, solution.u))


# -

# The same applies to statistical quantities that we can extract from the solution.
# For example, the standard deviation or samples from the solution object:

# +

key = jax.random.PRNGKey(seed=15)
_, _, ssm = probdiffeq.prior_wiener_integrated(tcoeffs, ssm_fact="dense")
strategy = probdiffeq.strategy_filter(ssm)
posterior = solution.posterior
sample_one = strategy.markov_sample(key, posterior, reverse=True)
sample_many = strategy.markov_sample(key, posterior, shape=(1, 2, 3), reverse=True)

print(jax.tree.map(jnp.shape, solution.u.mean))
print(jax.tree.map(jnp.shape, solution.u.std))
print(jax.tree.map(jnp.shape, sample_one))
print(jax.tree.map(jnp.shape, sample_many))

# -
