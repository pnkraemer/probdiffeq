"""Choose between prior distributions.

See also: https://nathanaelbosch.github.io/ProbNumDiffEq.jl/stable/priors/
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from probdiffeq import probdiffeq
from probdiffeq.backend import func

# Fail this notebook on NaN detection (to catch those in the CI)
jax.config.update("jax_debug_nans", True)


def main():
    """Sample from various prior distributions."""
    ts = jnp.linspace(0.0, 5.0, num=100, endpoint=True)

    @func.partial(probdiffeq.ode_autonomous_order_arbitrary, num_derivatives_in_args=3)
    def vf_matern(u, du, ddu, /):
        ell = 0.5
        return -(ell**3) * u - 3 * ell**2 * du - 3 * ell * ddu

    @func.partial(probdiffeq.ode_autonomous_order_arbitrary, num_derivatives_in_args=3)
    def vf_oscillator(_u, du, _ddu, /):
        return -5 * du  # always the second highest coefficient

    @func.partial(probdiffeq.ode_autonomous_order_arbitrary, num_derivatives_in_args=3)
    def vf_ioup(_u, _du, ddu, /):
        return -5 * ddu  # always the highest coefficient

    @func.partial(probdiffeq.ode_autonomous_order_arbitrary, num_derivatives_in_args=3)
    def vf_iwp(u, _du, _ddu, /):
        return 0.0 * u  # always zeros

    vf_titles = ["Oscillating", "Matern 5/2", "IOUP", "IWP"]

    _fig, axes = plt.subplots(
        nrows=3, ncols=4, sharex=True, figsize=(8, 5), constrained_layout=True
    )
    for i, (vf_prior, title, ax_col) in enumerate(
        zip([vf_oscillator, vf_matern, vf_ioup, vf_iwp], vf_titles, axes.T)
    ):
        # Match initial distribution to stationary distribution of Matern
        ssm = probdiffeq.state_space_model_dense()

        init, prior = ssm.prior_exponential_diffuse(
            vf_prior, [0.0, 0.0, 0.0], [2.5, 0.7, 0.6], output_scale=1.0
        )
        mseq = probdiffeq.MarkovSequence.from_grid(init, prior, grid=ts, reverse=False)

        num_samples = 3
        key = jax.random.PRNGKey(i)
        sample_fun = jax.jit(mseq.sample, static_argnames=["shape"])
        samples_prior = sample_fun(key, shape=(num_samples,))

        margs = mseq.evaluate_marginals()
        means = margs.mean
        stds = margs.std

        ax_col[0].set_title(title, fontsize="medium")

        for smp, m, std, ax in zip(samples_prior, means, stds, ax_col):
            ax.plot(ts, smp.T, color=f"C{i}", linewidth=1.0)
            ax.fill_between(ts, m - 2 * std, m + 2 * std, color=f"C{i}", alpha=0.25)

    axes[0][0].set_ylabel("State", fontsize="medium")
    axes[1][0].set_ylabel("Velocity", fontsize="medium")
    axes[2][0].set_ylabel("Acceleration", fontsize="medium")

    plt.show()


if __name__ == "__main__":
    main()
