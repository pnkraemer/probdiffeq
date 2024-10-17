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

# # An easy example
#
# Let's have a look at an easy example.

# +
"""Solve the logistic equation."""

import jax
import jax.numpy as jnp

from probdiffeq import ivpsolve, ivpsolvers, taylor
from probdiffeq.impl import impl

jax.config.update("jax_platform_name", "cpu")


# -

# Create a problem:


# +
@jax.jit
def vf(y, *, t):  # noqa: ARG001
    """Evaluate the vector field."""
    return 0.5 * y * (1 - y)


u0 = jnp.asarray([0.1])
t0, t1 = 0.0, 1.0
# -

#
# ProbDiffEq contains three levels of implementations:
#
# **Low:** Implementations of random-variable-arithmetic
# (marginalisation, conditioning, etc.)
#
# **Medium:** Probabilistic IVP solver components (this is what you're here for.)
#
# **High:** ODE-solving routines.
#
#
# There are several random-variable implementations
# (read: state-space model factorisations)
# which model different correlations between variables.
# All factorisations can be used interchangeably,
# but they have different speed, stability,
# and uncertainty-quantification properties.
# Since the chosen implementation powers almost everything,
# we choose one (and only one) of them,
# assign it to a global variable, and call it the "impl(ementation)".
#

impl.select("dense", ode_shape=(1,))
# But don't worry, this configuration does not make the library any less light-weight.
# It merely affects the shapes of the arrays
# describing means and covariances of Gaussian
# random variables, and assigns functions that know how to manipulate those parameters.
#

# Configuring a probabilistic IVP solver is a little more
# involved than configuring your favourite Runge-Kutta method:
# we must choose a prior distribution and a correction scheme,
# then we put them together as a filter or smoother,
# wrap everything into a solver, and (finally) make the solver adaptive.
#

# +
tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), (u0,), num=4)
output_scale = 1.0  # or any other value with the same shape
ibm = ivpsolvers.prior_ibm(tcoeffs, output_scale)
ts0 = ivpsolvers.correction_ts1(ode_order=1)

strategy = ivpsolvers.strategy_smoother(ibm, ts0)
solver = ivpsolvers.solver(strategy)
adaptive_solver = ivpsolve.adaptive(solver)


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
init = solver.initial_condition()
dt0 = 0.1
solution = ivpsolve.solve_adaptive_save_every_step(
    vf, init, t0=t0, t1=t1, dt0=dt0, adaptive_solver=adaptive_solver
)


# Look at the solution
print("u =", solution.u, "\n")
print("solution =", solution)
