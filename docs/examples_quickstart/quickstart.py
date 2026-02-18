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

# # Quickstart
#
# Let's have a look at an easy example.

# +
"""Solve the logistic equation."""

import jax
import jax.numpy as jnp

from probdiffeq import ivpsolve, probdiffeq, taylor

# Define a differential equation


@jax.jit
def vf(y, /, *, t):  # noqa: ARG001
    """Evaluate the dynamics of the logistic ODE."""
    return 2 * y * (1 - y)


u0 = jnp.asarray(0.1)
t0, t1 = 0.0, 5.0


# Set up a state-space model
tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), (u0,), num=1)
init, iwp, ssm = probdiffeq.prior_wiener_integrated(tcoeffs, ssm_fact="dense")


# Build a solver
ts = probdiffeq.constraint_ode_ts1(vf, ssm=ssm)
strategy = probdiffeq.strategy_filter(ssm=ssm)
solver = probdiffeq.solver_mle(ssm=ssm, strategy=strategy, prior=iwp, constraint=ts)
error = probdiffeq.error_residual_std(constraint=ts, prior=iwp, ssm=ssm)


# Solve the ODE
# To all users: Try different solution routines.
save_at = jnp.linspace(t0, t1, num=100, endpoint=True)
solve = ivpsolve.solve_adaptive_save_at(solver=solver, error=error)
solution = jax.jit(solve)(init, save_at, atol=1e-3, rtol=1e-3)


# Look at the solution
print(f"\ninitial = {jax.tree.map(jnp.shape, init)}")
print(f"\nsolution = {jax.tree.map(jnp.shape, solution)}")
