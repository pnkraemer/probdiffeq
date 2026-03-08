"""Some solvers are more stable than others.

This example shows how to solve a semilinear ODE with different solvers and priors.
"""

import functools

import jax
import jax.experimental.ode
import jax.numpy as jnp
import matplotlib.pyplot as plt

from probdiffeq import ivpsolve, probdiffeq, taylor

# Fail this notebook on NaN detection (to catch those in the CI)
jax.config.update("jax_debug_nans", True)

# Enable 64-bit precision for better stability in this example.
jax.config.update("jax_enable_x64", True)


def main():
    """Plot the solution of a semilinear ODE with different solvers and priors."""
    A = jnp.asarray([[-2.5, -1], [2.5, -1]])

    def vf(u, *, t):
        """Solve a linear ODE in 2d with different scales."""
        del t  # unused argument

        # Increase or decrease the nonlinearity to see how it affects the solvers!
        return A @ u + 0.01 * jnp.flip(u) ** 3

    u0 = jnp.asarray([1.0, -1.0])
    t0, t1 = 0.0, 4.0

    # Set up a state-space model over Taylor coefficients
    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), (u0,), num=2)
    init, ssm = probdiffeq.ssm_taylor(tcoeffs)

    # Build a solver
    strategy = probdiffeq.strategy_smoother_fixedinterval(ssm=ssm)
    iwp = probdiffeq.prior_wiener_integrated(ssm=ssm)
    ioup = probdiffeq.prior_ornstein_uhlenbeck_integrated(lambda s: A @ s, ssm=ssm)
    ts0 = probdiffeq.constraint_ode_ts0(vf, ssm=ssm)
    ts1 = probdiffeq.constraint_ode_ts1(vf, ssm=ssm)

    # Prepare the plot
    _fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(5, 3),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    solvers = [
        ("IWP+TS0", iwp, ts0),
        ("IOUP+TS0", ioup, ts0),
        ("IWP+TS1", iwp, ts1),
        ("IOUP+TS1", ioup, ts1),
    ]
    for (label, prior, constraint), ax in zip(solvers, axes.flatten()):
        # Set up the solver and solve the ODE
        solver = probdiffeq.solver(
            strategy=strategy, prior=prior, constraint=constraint, ssm=ssm
        )
        solve = ivpsolve.solve_fixed_grid(solver=solver)
        grid = jnp.linspace(t0, t1, num=15, endpoint=True)
        solution = jax.jit(solve)(init, grid=grid)

        # Calculate the solution at a finer grid for plotting
        ts = jnp.linspace(t0 + 1e-4, t1 - 1e-4, num=100, endpoint=True)
        dense = functools.partial(solver.offgrid_marginals, solution=solution)
        u = jax.jit(jax.vmap(dense))(ts)

        # Exaggerate the uncertainty for better visibility
        m, s = u.mean[0][:, 0], u.std[0][:, 0]
        ax.plot(ts, m, alpha=0.5, label=label)
        ax.fill_between(ts, m - 50 * s, m + 50 * s, alpha=0.25)
        ax.legend(fontsize="medium", frameon=False)

    # Label the plot
    axes[1][0].set_xlabel("Time", fontsize="medium")
    axes[1][1].set_xlabel("Time", fontsize="medium")
    axes[0][0].set_ylabel("State", fontsize="medium")
    axes[1][0].set_ylabel("State", fontsize="medium")
    plt.show()


if __name__ == "__main__":
    main()
