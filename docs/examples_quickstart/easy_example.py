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

from probdiffeq import ivpsolve, ivpsolvers, taylor

jax.config.update("jax_platform_name", "cpu")


# -

# Create a problem:


# +
@jax.jit
def vf(y, *, t):  # noqa: ARG001
    """Evaluate the vector field."""
    return 2.0 * y * (1 - y)


u0 = jnp.asarray([0.1])
t0, t1 = 0.0, 5.0
# -

# Configuring a probabilistic IVP solver is a little more
# involved than configuring your favourite Runge-Kutta method:
# we must choose a prior distribution and a correction scheme,
# then we put them together as a filter or smoother,
# wrap everything into a solver, and (finally) make the solver adaptive.
#

# +

# Set up a state-space model
tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), (u0,), num=4)
ibm, ssm = ivpsolvers.prior_ibm(tcoeffs, ssm_fact="dense")
ts0 = ivpsolvers.correction_ts1(ode_order=1, ssm=ssm)
strategy = ivpsolvers.strategy_smoother(ssm=ssm)

# Build a solver
solver = ivpsolvers.solver_mle(strategy, prior=ibm, correction=ts0, ssm=ssm)
adaptive_solver = ivpsolvers.adaptive(solver, ssm=ssm)
# -


# Other software packages that implement
# probabilistic IVP solvers do a lot of this work
# implicitly; probdiffeq enforces that
# the user makes these decisions, not only because
# it simplifies the solver implementations
# (quite a lot, actually),
# but it also shows how easily we can
# build a custom solver for our favourite problem
# (consult the other tutorials for examples).

# From here on, the rest is standard ODE-solver machinery:

# +
# Solve the ODE
init = solver.initial_condition()
dt0 = 0.1
solution = ivpsolve.solve_adaptive_save_every_step(
    vf, init, t0=t0, t1=t1, dt0=dt0, adaptive_solver=adaptive_solver, ssm=ssm
)

# Look at the solution
print(f"u = {jax.tree.map(jnp.shape, solution.u)}")  # Taylor coefficients
print(f"solution = {jax.tree.map(jnp.shape, solution)}")  # IVP solution
# -
