"""Understand the ODE posterior.

Diffuse initialisation of the prior yields samples that do not satisfy the ODE.
Taylor-coefficient initialisation yields samples that approximately satisfy it.
Conditioning on the ODE (forming the posterior) collapses residuals near zero.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from probdiffeq import ivpsolve, probdiffeq

# Fail this notebook on NaN detection (to catch those in the CI)
jax.config.update("jax_debug_nans", True)


def main():
    """Sample from a probabilistic solution and plot residuals."""
    # Create an ODE problem.

    @probdiffeq.ode
    def vector_field(y, /, *, t):
        """Evaluate the logistic ODE vector field."""
        del t
        return 10.0 * y * (2.0 - y)

    t0, t1 = 0.0, 2.5
    u0 = jnp.asarray(0.1)

    # Assemble the discretized prior (with and without the correct Taylor coefficients).

    ts = jnp.linspace(t0, t1, num=500, endpoint=True)

    # "Bad" prior (no Taylor coefficients)
    ssm = probdiffeq.state_space_model_dense()
    iwp_diffuse = ssm.prior_wiener_integrated(
        [u0], diffuse_derivatives=2, output_scale=10.0
    )
    mseq_prior = probdiffeq.MarkovSequence.from_grid(
        iwp_diffuse, grid=ts, reverse=False
    )

    # "Good" prior (Taylor coefficients)
    jetexpand = probdiffeq.jetexpand_ode_padded_scan(num=2)
    tcoeffs, _ = jetexpand(vector_field, (u0,), t=t0)

    iwp = ssm.prior_wiener_integrated(tcoeffs, output_scale=10.0)
    mseq_tcoeffs = probdiffeq.MarkovSequence.from_grid(iwp, grid=ts, reverse=False)

    # Posterior
    ts1 = ssm.constraint_ode_ts1(vector_field)
    strategy = probdiffeq.strategy_smoother_fixedpoint()
    solver = probdiffeq.solver(strategy=strategy, constraint=ts1)
    error = probdiffeq.error_residual_std(constraint=ts1)
    solve = ivpsolve.solve_adaptive_save_at(solver=solver, error=error)
    sol = solve(iwp, save_at=ts, atol=1e-1, rtol=1e-1)
    mseq_posterior = sol.solution_full

    # Compute samples.

    num_samples = 20
    key = jax.random.PRNGKey(seed=1)
    samples_prior = mseq_prior.sample(key, shape=(num_samples,))
    samples_tcoeffs = mseq_tcoeffs.sample(key, shape=(num_samples,))
    samples_posterior = mseq_posterior.sample(key, shape=(num_samples,))

    # Plot the results.

    def residual(x, t):
        """Evaluate the ODE residual."""
        fx = jax.vmap(jax.vmap(lambda s: vector_field(s, t=t)))(x[0])
        return x[1] - fx

    residual_prior = residual(samples_prior, ts)
    residual_tcoeffs = residual(samples_tcoeffs, ts)
    residual_posterior = residual(samples_posterior, ts)

    sample_style = {"marker": "None", "alpha": 0.99, "linewidth": 0.75}
    fig, (axes_state, axes_residual, axes_log_abs) = plt.subplots(
        nrows=3,
        ncols=3,
        sharex=True,
        sharey="row",
        constrained_layout=True,
        figsize=(8, 5),
    )
    axes_state[0].set_title("IWP w/ diffuse initialisation", fontsize="medium")
    axes_state[1].set_title("IWP w/ Taylor coefficients", fontsize="medium")
    axes_state[2].set_title("Posterior", fontsize="medium")

    axes_state[0].set_xticks((t0, (t0 + t1) / 2, t1))
    axes_state[0].set_xlim((t0, t1))
    axes_state[0].set_ylim((-1, 3))
    axes_state[0].set_yticks((-1, 1, 3))
    axes_residual[0].set_ylim((-10.0, 20))
    axes_residual[0].set_yticks((-10.0, 5, 20))
    axes_log_abs[0].set_ylim((-6, 4))
    axes_log_abs[0].set_yticks((-6, -1, 4))

    axes_state[0].set_ylabel("Solution", fontsize="medium")
    axes_residual[0].set_ylabel("Residual", fontsize="medium")
    axes_log_abs[0].set_ylabel("Log-residual", fontsize="medium")
    axes_log_abs[0].set_xlabel("Time $t$", fontsize="medium")
    axes_log_abs[1].set_xlabel("Time $t$", fontsize="medium")
    axes_log_abs[2].set_xlabel("Time $t$", fontsize="medium")

    axes_state[0].plot(ts, samples_prior[0].T, **sample_style, color="C0")
    axes_state[1].plot(ts, samples_tcoeffs[0].T, **sample_style, color="C1")
    axes_state[2].plot(ts, samples_posterior[0].T, **sample_style, color="C2")

    axes_residual[0].plot(ts, residual_prior.T, **sample_style, color="C0")
    axes_residual[1].plot(ts, residual_tcoeffs.T, **sample_style, color="C1")
    axes_residual[2].plot(ts, residual_posterior.T, **sample_style, color="C2")

    axes_log_abs[0].plot(
        ts, jnp.log10(jnp.abs(residual_prior)).T, **sample_style, color="C0"
    )
    axes_log_abs[1].plot(
        ts, jnp.log10(jnp.abs(residual_tcoeffs)).T, **sample_style, color="C1"
    )
    axes_log_abs[2].plot(
        ts, jnp.log10(jnp.abs(residual_posterior)).T, **sample_style, color="C2"
    )

    fig.align_ylabels()
    plt.show()


if __name__ == "__main__":
    main()
