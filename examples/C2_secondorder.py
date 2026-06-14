"""Solve second order IVPs without transformation.

Solve the restricted three-body problem first as a first-order IVP
(by concatenating position and velocity into a flat state vector),
then as its natural second-order IVP.
The second-order formulation tracks fewer Taylor coefficients.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from probdiffeq import ivpsolve, probdiffeq

# Fail this notebook on NaN detection (to catch those in the CI)
jax.config.update("jax_debug_nans", True)

# Restricted three-body problem (Arenstorf orbit).
# Source: Hairer, Norsett, Wanner (1993), p. 129.
mu = 0.012277471
mp = 1.0 - mu
u0 = jnp.asarray([0.994, 0.0])
du0 = jnp.asarray([0.0, -2.00158510637908252240537862224])
t0, t1 = 0.0, 17.0652165601579625588917206249


def main():
    """Solve the three-body problem with and without transforming the problem."""
    save_at = jnp.linspace(t0, t1, endpoint=True, num=250)
    atol, rtol = 1e-5, 1e-5

    # Solve the IVP as a first-order problem
    u0_1st = jnp.concatenate([u0, du0])

    @probdiffeq.ode
    def vf_1(y, /, *, t):
        """Evaluate the three-body problem as a first-order IVP."""
        del t
        pos, vel = y[:2], y[2:]
        D1 = jnp.linalg.norm(jnp.asarray([pos[0] + mu, pos[1]])) ** 3.0
        D2 = jnp.linalg.norm(jnp.asarray([pos[0] - mp, pos[1]])) ** 3.0
        acc0 = pos[0] + 2 * vel[1] - mp * (pos[0] + mu) / D1 - mu * (pos[0] - mp) / D2
        acc1 = pos[1] - 2 * vel[0] - mp * pos[1] / D1 - mu * pos[1] / D2
        return jnp.concatenate([vel, jnp.asarray([acc0, acc1])])

    jetexpand = probdiffeq.jetexpand_ode_padded_scan(num=4)
    tcoeffs, _ = jetexpand(vf_1, (u0_1st,), t=t0)
    ssm = probdiffeq.state_space_model_isotropic()
    iwp = ssm.prior_wiener_integrated(tcoeffs, output_scale=1.0)
    strategy = probdiffeq.strategy_filter()
    ts0 = ssm.constraint_ode_ts0(vf_1)
    solver_1st = probdiffeq.solver_mle(strategy=strategy, constraint=ts0)
    error_1st = probdiffeq.error_residual_std(constraint=ts0)
    solve = ivpsolve.solve_adaptive_save_at(solver=solver_1st, error=error_1st)

    # Plot the result
    solution = jax.jit(solve)(iwp, save_at=save_at, atol=atol, rtol=rtol)
    plt.plot(solution.u.mean[0][:, 0], solution.u.mean[0][:, 1], marker=".")
    plt.show()

    # Solve the IVP as a second-order problem (its natural formulation)

    @probdiffeq.ode_order_two
    def vf_2(y, dy, /, *, t):
        """Evaluate the three-body problem as a second-order IVP."""
        del t
        D1 = jnp.linalg.norm(jnp.asarray([y[0] + mu, y[1]])) ** 3.0
        D2 = jnp.linalg.norm(jnp.asarray([y[0] - mp, y[1]])) ** 3.0
        ddy0 = y[0] + 2 * dy[1] - mp * (y[0] + mu) / D1 - mu * (y[0] - mp) / D2
        ddy1 = y[1] - 2 * dy[0] - mp * y[1] / D1 - mu * y[1] / D2
        return jnp.asarray([ddy0, ddy1])

    # Different derivative count because we don't transform to first order:
    # the goal is to match the number of tracked Taylor coefficients.
    jetexpand = probdiffeq.jetexpand_ode_padded_scan(num=3)
    tcoeffs, _ = jetexpand(vf_2, (u0, du0), t=t0)
    ssm = probdiffeq.state_space_model_isotropic()
    iwp = ssm.prior_wiener_integrated(tcoeffs, output_scale=1.0)
    ts0 = ssm.constraint_ode_ts0(vf_2)
    strategy = probdiffeq.strategy_filter()
    solver_2nd = probdiffeq.solver_mle(strategy=strategy, constraint=ts0)
    error_2nd = probdiffeq.error_residual_std(constraint=ts0)
    solve = ivpsolve.solve_adaptive_save_at(solver=solver_2nd, error=error_2nd)

    # Plot the result
    solution = jax.jit(solve)(iwp, save_at=save_at, atol=atol, rtol=rtol)
    plt.plot(solution.u.mean[0][:, 0], solution.u.mean[0][:, 1], marker=".")
    plt.show()


if __name__ == "__main__":
    main()
