"""Solve second-order IVPs without transformation."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffeqzoo import backend, ivps

from probdiffeq import ivpsolve, probdiffeq

# Fail this notebook on NaN detection (to catch those in the CI)
jax.config.update("jax_debug_nans", True)


if not backend.has_been_selected:
    backend.select("jax")  # ivp examples in jax


def main():
    """Solve the three-body problem with and without transforming the problem."""
    # Solve the IVP as a first-order problem
    f, u0, (t0, t1), f_args = ivps.three_body_restricted_first_order()
    save_at = jnp.linspace(t0, t1, endpoint=True, num=250)
    atol, rtol = 1e-5, 1e-5

    @probdiffeq.ode
    def vf_1(y, /, *, t):
        """Evaluate the three-body problem as a first-order IVP."""
        del t
        return f(y, *f_args)

    jetexpand = probdiffeq.jetexpand_ode_padded_scan(num=4)
    tcoeffs, _ = jetexpand(vf_1, (u0,), t=t0)
    ssm = probdiffeq.state_space_model_isotropic()
    init, iwp = ssm.prior_wiener_integrated(tcoeffs, output_scale=1.0)
    strategy = probdiffeq.strategy_filter()
    ts0 = ssm.constraint_ode_ts0(vf_1)
    solver_1st = probdiffeq.solver_mle(strategy=strategy, prior=iwp, constraint=ts0)
    error_1st = probdiffeq.error_residual_std(constraint=ts0, prior=iwp)
    solve = ivpsolve.solve_adaptive_save_at(solver=solver_1st, error=error_1st)

    # Plot the result
    solution = jax.jit(solve)(init, save_at=save_at, atol=atol, rtol=rtol)
    plt.plot(solution.u.mean[0][:, 0], solution.u.mean[0][:, 1], marker=".")
    plt.show()

    # Solve the IVP as a second-order problem (its natural formulation)

    f, (u0, du0), (t0, t1), f_args = ivps.three_body_restricted()

    @probdiffeq.ode_second_order
    def vf_2(y, dy, /, *, t):
        """Evaluate the three-body problem as a second-order IVP."""
        del t
        return f(y, dy, *f_args)

    # Different derivative count because we don't transform to first order
    # The goal is to match the number of tracked taylor coefficients.
    jetexpand = probdiffeq.jetexpand_ode_padded_scan(num=3)
    tcoeffs, _ = jetexpand(vf_2, (u0, du0), t=t0)
    ssm = probdiffeq.state_space_model_isotropic()
    init, iwp = ssm.prior_wiener_integrated(tcoeffs, output_scale=1.0)
    ts0 = ssm.constraint_ode_ts0(vf_2)
    strategy = probdiffeq.strategy_filter()
    solver_2nd = probdiffeq.solver_mle(strategy=strategy, prior=iwp, constraint=ts0)
    error_2nd = probdiffeq.error_residual_std(constraint=ts0, prior=iwp)
    solve = ivpsolve.solve_adaptive_save_at(solver=solver_2nd, error=error_2nd)

    # Plot the result
    solution = jax.jit(solve)(init, save_at=save_at, atol=atol, rtol=rtol)
    plt.plot(solution.u.mean[0][:, 0], solution.u.mean[0][:, 1], marker=".")
    plt.show()


if __name__ == "__main__":
    main()
