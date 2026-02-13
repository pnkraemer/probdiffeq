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

# # Equinox's while-loops
#
# Use [Equinox's](https://docs.kidger.site/equinox/)
# bounded while loop to enable reverse-mode differentiation of adaptive IVP solvers.

# +
"""Use Equinox's while loop to compute gradients of `simulate_terminal_values`."""

import equinox
import jax
import jax.numpy as jnp

from probdiffeq import ivpsolve, probdiffeq, taylor

# -


def solution_routine(while_loop):
    """Construct a parameter-to-solution function and an initial value."""

    def vf(y, *, t):  # noqa: ARG001
        """Evaluate the vector field."""
        return 0.5 * y * (1 - y)

    t0, t1 = 0.0, 1.0
    u0 = jnp.asarray([0.1])

    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), (u0,), num=1)
    init, ibm, ssm = probdiffeq.prior_wiener_integrated(tcoeffs, ssm_fact="isotropic")
    ts0 = probdiffeq.correction_ts0(vf, ode_order=1, ssm=ssm)

    strategy = probdiffeq.strategy_smoother_fixedpoint(ssm=ssm)
    solver = probdiffeq.solver(strategy, prior=ibm, correction=ts0, ssm=ssm)
    errorest = probdiffeq.errorest_schober_bosch(prior=ibm, correction=ts0, ssm=ssm)
    solve_adaptive = ivpsolve.solve_adaptive_terminal_values(
        solver=solver, errorest=errorest, while_loop=while_loop
    )

    def simulate(init_val):
        """Evaluate the parameter-to-solution function."""
        sol = solve_adaptive(init_val, t0=t0, t1=t1, atol=1e-3, rtol=1e-3)

        # Any scalar function of the IVP solution would do
        # Try the log-marginal-likelihood losses (see the other tutorials).
        return jnp.dot(sol.u.mean[0], sol.u.mean[0])

    return simulate, init


# This is the default behaviour
solve, x = solution_routine(jax.lax.while_loop)

try:
    solution, gradient = jax.jit(jax.value_and_grad(solve))(x)
except ValueError as err:
    print(f"Caught error:\n\t {err}")

# This while-loop makes the solver differentiable


def while_loop_func(*a, **kw):
    """Evaluate a bounded while loop."""
    return equinox.internal.while_loop(*a, **kw, kind="bounded", max_steps=100)


solve, x = solution_routine(while_loop=while_loop_func)

# Compute gradients
solution, gradient = jax.jit(jax.value_and_grad(solve))(x)

print(solution)
print(gradient)
