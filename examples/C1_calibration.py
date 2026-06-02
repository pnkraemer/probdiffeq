"""Leverage dynamic calibration."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from probdiffeq import ivpsolve, probdiffeq

# Fail this notebook on NaN detection (to catch those in the CI)
jax.config.update("jax_debug_nans", True)


def main():
    """Solve a linear ODE with dynamic vs MLE solvers."""

    @probdiffeq.ode_vector_field
    def vf(y, /, *, t):
        """Evaluate the affine vector field."""
        del t
        return 2 * y

    t0, t1 = 0.0, 1.0
    u0 = jnp.asarray(1.0)

    tcoeffs = (u0, vf(u0, t=t0))
    ssm = probdiffeq.state_space_model(ssm_fact="dense")
    init, iwp = probdiffeq.prior_wiener_integrated(tcoeffs, ssm=ssm, output_scale=1.0)
    ts1 = probdiffeq.constraint_ode_ts1(vf, ssm=ssm)
    strategy = probdiffeq.strategy_filter(ssm=ssm)
    dynamic = probdiffeq.solver_dynamic(
        strategy=strategy, prior=iwp, constraint=ts1, ssm=ssm
    )
    mle = probdiffeq.solver_mle(strategy=strategy, prior=iwp, constraint=ts1, ssm=ssm)

    t0, t1 = 0.0, 3.0
    num_pts = 200

    ts = jnp.linspace(t0, t1, num=num_pts, endpoint=True)

    solve_dynamic = ivpsolve.solve_fixed_grid(solver=dynamic)
    solution_dynamic = jax.jit(solve_dynamic)(init, grid=ts)

    solve_mle = ivpsolve.solve_fixed_grid(solver=mle)
    solution_mle = jax.jit(solve_mle)(init, grid=ts)

    # Plot the solution.

    fig, (axes_linear, axes_log) = plt.subplots(
        ncols=2, nrows=2, sharex=True, sharey="row"
    )

    u_dynamic = solution_dynamic.u.mean[0]
    u_mle = solution_mle.u.mean[0]

    style_target = {
        "marker": "None",
        "label": "Target",
        "color": "black",
        "linewidth": 0.5,
        "alpha": 1,
        "linestyle": "dashed",
    }
    style_approx = {
        "marker": "None",
        "label": "Posterior mean",
        "color": "C0",
        "linewidth": 1.5,
        "alpha": 0.75,
    }

    axes_linear[0].set_title("Time-varying model", fontsize="medium")
    axes_linear[0].plot(ts, jnp.exp(ts * 2), **style_target)
    axes_linear[0].plot(ts, u_dynamic, **style_approx)
    axes_linear[0].legend()

    axes_linear[1].set_title("Constant model", fontsize="medium")
    axes_linear[1].plot(ts, jnp.exp(ts * 2), **style_target)
    axes_linear[1].plot(ts, u_mle, **style_approx)
    axes_linear[1].legend()

    axes_linear[0].set_ylabel("Linear scale", fontsize="medium")

    axes_linear[0].set_xlim((t0, t1))

    axes_log[0].semilogy(ts, jnp.exp(ts * 2), **style_target)
    axes_log[0].semilogy(ts, u_dynamic, **style_approx)
    axes_log[0].legend()

    axes_log[1].semilogy(ts, jnp.exp(ts * 2), **style_target)
    axes_log[1].semilogy(ts, u_mle, **style_approx)
    axes_log[1].legend()

    axes_log[0].set_ylabel("Logarithmic scale", fontsize="medium")
    axes_log[0].set_xlabel("Time t", fontsize="medium")
    axes_log[1].set_xlabel("Time t", fontsize="medium")

    axes_log[0].set_xlim((t0, t1))

    fig.align_ylabels()
    plt.show()

    # The dynamic solver adapts the output-scale so
    # that both the solution and the output-scale
    # grow exponentially.
    # The ODE-solution fits the truth well.
    #
    # The solver_mle does not have this tool, and
    # the ODE solution is not able to
    # follow the exponential: it drifts
    # back to the origin.
    # (This is expected, we are basically trying to
    # fit an exponential with a piecewise polynomial.)


if __name__ == "__main__":
    main()
