"""Enable reverse-mode derivatives.

Reverse-mode differentiation through a `jax.lax.while_loop` is not supported
by default, because JAX cannot trace through a loop with a data-dependent
iteration count.
Equinox provides a bounded while loop that unrolls up to a fixed number of steps,
making the gradient tractable.
This example shows how switching to `equinox.internal.while_loop` enables
reverse-mode differentiation of adaptive IVP solvers.
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
    iwp = ssm.prior_wiener_integrated(tcoeffs)
    ts0 = ssm.constraint_ode_ts0(vf)

    strategy = probdiffeq.strategy_smoother_fixedpoint()
    solver = probdiffeq.solver(strategy=strategy, constraint=ts0)
    error = probdiffeq.error_residual_std(constraint=ts0)
    solve_adaptive = ivpsolve.solve_adaptive_terminal_values(
        solver=solver, error=error, while_loop=while_loop
    )

    def simulate(prior):
        """Evaluate the parameter-to-solution function."""
        sol = solve_adaptive(prior, t0=t0, t1=t1, atol=1e-3, rtol=1e-3)

        # Any scalar function of the IVP solution would do
        # Try the log-marginal-likelihood losses (see the other tutorials).
        return jnp.dot(sol.u.mean[0], sol.u.mean[0])

    return simulate, iwp


if __name__ == "__main__":
    main()
