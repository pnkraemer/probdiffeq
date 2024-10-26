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

# # Using Equinox's reverse-mode differentiable while-loops
#
# Use [Equinox's](https://docs.kidger.site/equinox/)
# bounded while loop to enable reverse-mode differentiation of adaptive IVP solvers.

# +
"""Use Equinox's while loop to compute gradients of `simulate_terminal_values`."""

import equinox
import jax
import jax.numpy as jnp

from probdiffeq import ivpsolve, ivpsolvers, taylor
from probdiffeq.backend import control_flow

jax.config.update("jax_platform_name", "cpu")


# -

# Overwrite the while-loop (via a context manager):


# +
def while_loop_func(*a, **kw):
    """Evaluate a bounded while loop."""
    return equinox.internal.while_loop(*a, **kw, kind="bounded", max_steps=100)


context_compute_gradient = control_flow.context_overwrite_while_loop(while_loop_func)
# -

# The rest is the similar to the "easy example" in the quickstart,
# except for simulating adaptively and
# computing the value and the gradient
# (which is impossible without the specialised while-loop implementation).


def solution_routine():
    """Construct a parameter-to-solution function and an initial value."""

    def vf(y, *, t):  # noqa: ARG001
        """Evaluate the vector field."""
        return 0.5 * y * (1 - y)

    t0, t1 = 0.0, 1.0
    u0 = jnp.asarray([0.1])

    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), (u0,), num=1)
    ibm, ssm = ivpsolvers.prior_ibm(tcoeffs, ssm_fact="isotropic")
    ts0 = ivpsolvers.correction_ts0(ode_order=1, ssm=ssm)

    strategy = ivpsolvers.strategy_fixedpoint(ssm=ssm)
    solver = ivpsolvers.solver(strategy, prior=ibm, correction=ts0, ssm=ssm)
    adaptive_solver = ivpsolvers.adaptive(solver, ssm=ssm)
    init = solver.initial_condition()

    def simulate(init_val):
        """Evaluate the parameter-to-solution function."""
        sol = ivpsolve.solve_adaptive_terminal_values(
            vf,
            init_val,
            t0=t0,
            t1=t1,
            dt0=0.1,
            adaptive_solver=adaptive_solver,
            ssm=ssm,
        )

        # Any scalar function of the IVP solution would do
        return jnp.dot(sol.u[0], sol.u[0])

    return simulate, init


try:
    solve, x = solution_routine()
    solution, gradient = jax.value_and_grad(solve)(x)
except ValueError as err:
    print(f"Caught error:\n\t {err}")

with context_compute_gradient:
    # Construct the solution routine inside the context
    solve, x = solution_routine()

    # Compute gradients
    solution, gradient = jax.value_and_grad(solve)(x)

    print(solution)
    print(gradient)
