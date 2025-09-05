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
import matplotlib.pyplot as plt

from probdiffeq import ivpsolve, ivpsolvers, taylor

# Define a differential equation


@jax.jit
def vf(y, *, t):  # noqa: ARG001
    """Evaluate the dynamics of the logistic ODE."""
    return 2 * y * (1 - y)


u0 = jnp.asarray([0.1])
t0, t1 = 0.0, 5.0


# Set up a state-space model
tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), (u0,), num=1)
ibm, ssm = ivpsolvers.prior_ibm(tcoeffs, ssm_fact="dense")


# Build a solver
ts = ivpsolvers.correction_ts1(ssm=ssm, ode_order=1)
strategy = ivpsolvers.strategy_filter(ssm=ssm)
solver = ivpsolvers.solver_mle(ssm=ssm, strategy=strategy, prior=ibm, correction=ts)
adaptive_solver = ivpsolvers.adaptive(solver, ssm=ssm)


# Solve the ODE
# To all users: Try different solution routines.
init = solver.initial_condition()
solution = ivpsolve.solve_adaptive_save_every_step(
    vf, init, t0=t0, t1=t1, dt0=0.1, adaptive_solver=adaptive_solver, ssm=ssm
)

# Look at the solution
print(f"\ninitial = {jax.tree.map(jnp.shape, init)}")
print(f"\nsolution = {jax.tree.map(jnp.shape, solution)}")


fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(solution.t, solution.u[0])
ax.set_xlabel("Time")
ax.set_ylabel("ODE solution")
ax.set_xlim((t0, t1))
plt.tight_layout()
plt.show()
