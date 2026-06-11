"""Enable reverse-mode derivatives.

Use [Equinox's](https://docs.kidger.site/equinox/)
bounded while loop to enable reverse-mode differentiation of adaptive IVP solvers.
"""

import equinox
import jax
import jax.numpy as jnp

from probdiffeq import ivpsolve, probdiffeq

# Fail this notebook on NaN detection (to catch those in the CI)
jax.config.update("jax_debug_nans", True)


def main():
    """Solve an ODE with and without a bounded while loop."""
    # This is the default behaviour:
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


def solution_routine(while_loop):
    """Construct a parameter-to-solution function and an initial value."""

    @probdiffeq.ode
    def vf(y, /, *, t):  # noqa: ARG001
        """Evaluate the vector field."""
        return 0.5 * y * (1 - y)

    t0, t1 = 0.0, 1.0
    u0 = jnp.asarray([0.1])

    jetexpand = probdiffeq.jetexpand_ode_padded_scan(num=1)
    tcoeffs, _ = jetexpand(vf, (u0,), t=t0)
    ssm = probdiffeq.state_space_model_isotropic()
    init, iwp = ssm.prior_wiener_integrated(tcoeffs)
    ts0 = ssm.constraint_ode_ts0(vf)

    strategy = probdiffeq.strategy_smoother_fixedpoint()
    solver = probdiffeq.solver(strategy=strategy, prior=iwp, constraint=ts0)
    error = probdiffeq.error_residual_std(constraint=ts0, prior=iwp)
    solve_adaptive = ivpsolve.solve_adaptive_terminal_values(
        solver=solver, error=error, while_loop=while_loop
    )

    def simulate(init_val):
        """Evaluate the parameter-to-solution function."""
        sol = solve_adaptive(init_val, t0=t0, t1=t1, atol=1e-3, rtol=1e-3)

        # Any scalar function of the IVP solution would do
        # Try the log-marginal-likelihood losses (see the other tutorials).
        return jnp.dot(sol.u.mean[0], sol.u.mean[0])

    return simulate, init


if __name__ == "__main__":
    main()
