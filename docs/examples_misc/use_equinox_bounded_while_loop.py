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

# # Use Equinox's reverse-mode differentiable while-loops
#
# Use [Equinox's](https://docs.kidger.site/equinox/)
# bounded while loop to enable reverse-mode differentiation of adaptive IVP solvers.

# +
"""Use Equinox's while loops to compute gradients of `simulate_terminal_values`."""

import jax
import jax.config
import jax.numpy as jnp

from probdiffeq import adaptive, ivpsolve, timestep
from probdiffeq.impl import impl
from probdiffeq.solvers import uncalibrated
from probdiffeq.solvers.strategies import fixedpoint
from probdiffeq.solvers.strategies.components import corrections, priors
from probdiffeq.taylor import autodiff

jax.config.update("jax_platform_name", "cpu")
impl.select("dense", ode_shape=(1,))


# -

# Overwrite the while-loop:

# +
import equinox  # noqa: E402

from probdiffeq.backend import control_flow  # noqa: E402


def while_loop_func(*a, **kw):
    """Call a reverse-mode differentiable while loop."""
    return equinox.internal.while_loop(*a, **kw, kind="bounded", max_steps=100)


control_flow.overwrite_while_loop_func(while_loop_func)

# -

# The rest is the similar to the "easy example" in the quickstart,
# with the exception of simulating adaptively and
# computing the value and the gradient
# (which is impossible without the specialised while-loop implementation).


# +


@jax.jit
def vf(y, *, t):  # noqa: ARG001
    """Evaluate the vector field."""
    return 0.5 * y * (1 - y)


u0 = jnp.asarray([0.1])
t0, t1 = 0.0, 1.0

ibm = priors.ibm_adaptive(num_derivatives=1)
ts0 = corrections.ts0(ode_order=1)

strategy = fixedpoint.fixedpoint_adaptive(ibm, ts0)
solver = uncalibrated.solver(strategy)
adaptive_solver = adaptive.adaptive(solver)

tcoeffs = autodiff.taylor_mode_scan(lambda y: vf(y, t=t0), (u0,), num=1)
init = solver.initial_condition(tcoeffs, 1.0)
dt0 = timestep.initial(lambda y: vf(y, t=t0), (u0,))  # or use e.g. dt0=0.1


def simulate(init_val):
    """Simulate the norm of the terminal value of an IVP."""
    sol = ivpsolve.simulate_terminal_values(
        vf, init_val, t0=t0, t1=t1, dt0=dt0, adaptive_solver=adaptive_solver
    )
    return jnp.dot(sol.u, sol.u)


# Look at the solution
solution, gradient = jax.value_and_grad(simulate)(init)
print(solution)
print(gradient)
