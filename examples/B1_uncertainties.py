"""Visualise solution uncertainty.

Probabilistic ODE solvers return a posterior distribution over trajectories,
not just a single mean trajectory.
This example plots the posterior mean together with uncertainty bands
for the state and its first few derivatives.
"""

# Set up the ODE

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from probdiffeq import ivpsolve, probdiffeq

# Fail this notebook on NaN detection (to catch those in the CI)
jax.config.update("jax_debug_nans", True)


def main():
    """Plot means and standard deviations of solvers."""

    @probdiffeq.ode
    def vf(y, /, *, t):  # noqa: ARG001
        """Evaluate the Lotka-Volterra vector field."""
        y0, y1 = y[0], y[1]

        y0_new = 0.5 * y0 - 0.05 * y0 * y1
        y1_new = -0.5 * y1 + 0.05 * y0 * y1
        return jnp.asarray([y0_new, y1_new])

    t0 = 0.0
    t1 = 2.0
    u0 = jnp.asarray([20.0, 20.0])

    # Set up a solver.

    ssm = probdiffeq.state_space_model_blockdiag()
    jetexpand = probdiffeq.jetexpand_ode_padded_scan(num=2)
    tcoeffs, _ = jetexpand(vf, (u0,), t=t0)
    iwp = ssm.prior_wiener_integrated(tcoeffs)
    ts1 = ssm.constraint_ode_ts1(vf)
    strategy = probdiffeq.strategy_smoother_fixedpoint()

    solver = probdiffeq.solver_mle(strategy=strategy, constraint=ts1)
    error = probdiffeq.error_residual_std(constraint=ts1)
    solve = ivpsolve.solve_adaptive_save_at(solver=solver, error=error)

    # Solve the ODE.

    ts = jnp.linspace(t0, t1, endpoint=True, num=500)
    sol = jax.jit(solve)(iwp, save_at=ts, dt0=0.1, atol=1e-1, rtol=1e-1)

    # Plot the solution.

    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(iwp.init.mean),
        sharex="col",
        tight_layout=True,
        figsize=(len(sol.u.mean) * 3, 5),
    )
    titles = ["Position", "Velocity", "Acceleration"]

    content = {
        "Solution (smoothed)": sol.u,
        "Fwd. pass (filtered)": sol.solution_full.filtering,
    }

    for label, u in content.items():
        for i, (u_i, std_i, ax_i, title_i) in enumerate(
            zip(u.mean, u.std, axes.T, titles)
        ):
            # Set up titles and axis descriptions
            ax_i[0].set_title(title_i, fontsize="medium")
            if i == 0:
                ax_i[0].set_ylabel("Prey", fontsize="medium")
                ax_i[1].set_ylabel("Predators", fontsize="medium")
                # ax_i[2].set_ylabel("Std.-dev.", fontsize="medium")

            ax_i[-1].set_xlabel("Time", fontsize="medium")

            for m, std, ax in zip(u_i.T, std_i.T, ax_i):
                # Plot the mean
                ax.plot(sol.t, m, label=label)

                # Plot the standard deviation
                lower, upper = m - 3 * std, m + 3 * std
                ax.fill_between(sol.t, lower, upper, alpha=0.3)
                ax.set_xlim((jnp.amin(ts), jnp.amax(ts)))

    for ax in axes.flatten():
        ax.legend(fontsize="x-small")
        ax.grid(linestyle="dotted")

    fig.align_ylabels()
    plt.show()


if __name__ == "__main__":
    main()
