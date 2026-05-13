"""Some solvers are more stable than others.

This tutorial reproduces Figure 1 from:

> Bosch, N., Hennig, P., & Tronarp, F. (2023).
  Probabilistic exponential integrators.
  Advances in Neural Information Processing Systems, 36, 40450-40467.

It shows the stability of first and zeroth-order methods with different priors.
"""

import functools

import jax
import jax.experimental.ode
import jax.numpy as jnp
import matplotlib.pyplot as plt

from probdiffeq import diffeqjet, ivpsolve, probdiffeq

# Fail this notebook on NaN detection (to catch those in the CI)
jax.config.update("jax_debug_nans", True)


def main():
    """Plot the solution of a semilinear ODE with different solvers and priors."""

    def vf(u, *, t):
        """Evaluate a linear vector field."""
        del t
        du1 = -0.5 * u[0] + 20 * u[1]
        du2 = -20 * u[1]
        return jnp.asarray([du1, du2])

    u0 = jnp.asarray([0.0, 1.0])
    t0, t1 = 0.0, 3.0

    A = jnp.asarray([[-0.5, 20], [0, -20]])

    # Set up a state-space model over Taylor coefficients
    ssm = probdiffeq.state_space_model()

    # Build a solver
    tcoeffs = diffeqjet.odejet_padded_scan(lambda y: vf(y, t=t0), (u0,), num=3)
    init, iwp = probdiffeq.prior_wiener_integrated(tcoeffs, ssm=ssm)
    strategy = probdiffeq.strategy_smoother_fixedinterval(ssm=ssm)
    _init, ioup = probdiffeq.prior_ornstein_uhlenbeck_integrated(
        lambda s: A @ s, tcoeffs, ssm=ssm
    )
    ts0 = probdiffeq.constraint_ode_ts0(vf, ssm=ssm)
    ts1 = probdiffeq.constraint_ode_ts1(vf, ssm=ssm)

    # Prepare the plot
    _fig, axes = plt.subplots(ncols=4, figsize=(13, 3), constrained_layout=True)
    solvers = [
        ("IWP + TS0 (300 steps)", iwp, ts0, 300),
        ("IOUP + TS0 (275 steps)", ioup, ts0, 275),
        ("IWP + TS1 (15 steps)", iwp, ts1, 15),
        ("IOUP + TS1 (6 steps)", ioup, ts1, 6),
    ]
    for i, ((label, prior, constraint, num), ax) in enumerate(
        zip(solvers, axes.flatten())
    ):
        # Set up the solver and solve the ODE
        solver = probdiffeq.solver_mle(
            strategy=strategy, prior=prior, constraint=constraint, ssm=ssm
        )
        solve = ivpsolve.solve_fixed_grid(solver=solver)
        grid = jnp.linspace(t0, t1, num=num, endpoint=True)
        solution = jax.jit(solve)(init, grid=grid)

        # Calculate the solution at a finer grid for plotting
        ts = jnp.linspace(t0 + 1e-4, t1 - 1e-4, num=200, endpoint=True)
        dense = functools.partial(solver.offgrid_marginals, solution=solution)
        u = jax.jit(jax.vmap(dense))(ts)

        # Plot the solution
        ax.set_title(label, fontsize="medium")
        for d in (0, 1):
            ax.plot(
                solution.t,
                solution.u.mean[0][:, d],
                ".",
                alpha=0.75,
                markerfacecolor=f"C{i}",
                markeredgecolor="black",
            )
            m, s = u.mean[0][:, d], u.std[0][:, d]
            ax.plot(ts, m, alpha=0.5, color=f"C{i}")
            ax.fill_between(ts, m - s, m + s, alpha=0.25, color=f"C{i}")

        # Set axis limits and labels
        ax.set_xlim((t0 - 0.05, t1 + 0.05))
        ax.set_ylim((-0.125, 1.125))
        ax.set_xlabel("Time", fontsize="medium")

    axes[0].set_ylabel("State", fontsize="medium")
    plt.show()


if __name__ == "__main__":
    main()
