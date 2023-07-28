"""Test the discrete IBM transitions."""

import jax
import jax.numpy as jnp

from probdiffeq.statespace.scalar import extra


def test_marginal_moments_are_correct(num_derivatives=1):
    """Solve a second-order, scalar, linear, separable BVP."""
    output_scale = 10.0
    t0, t1 = 0.0, 3.4123412
    grid = jnp.linspace(t0, t1, endpoint=True, num=20)

    markovseq = extra.ibm_discretise_fwd(
        jnp.diff(grid), num_derivatives=num_derivatives, output_scale=output_scale
    )

    means, stds = _marginal_moments(markovseq)
    _assert_zero_mean(means)
    _assert_monotonously_increasing_std(stds)
    _assert_brownian_motion_std(
        stds[-1, -1],
        std_init=output_scale,
        t0=t0,
        t1=t1,
        output_scale=output_scale,
        num_derivatives=num_derivatives,
    )


def _marginal_moments(precon_mseq):
    def step(rv, model):
        rv = extra.extrapolate_precon(rv, *model)
        return rv, rv

    _, rvs = jax.lax.scan(
        step,
        init=precon_mseq.init,
        xs=(precon_mseq.conditional, precon_mseq.preconditioner),
        reverse=False,
    )
    means, cov_sqrtms = rvs.mean, rvs.cov_sqrtm_lower

    @jax.vmap
    def cov(x):
        return x @ x.T

    covs = cov(cov_sqrtms)
    stds = jnp.sqrt(jax.vmap(jnp.diagonal)(covs))
    return means, stds


def _assert_zero_mean(means):
    assert jnp.allclose(means, 0.0)


def _assert_monotonously_increasing_std(stds):
    diffs = jnp.diff(stds, axis=0)
    assert jnp.all(diffs > 0), diffs


def _assert_brownian_motion_std(
    std_final, std_init, t0, t1, *, output_scale, num_derivatives
):
    received = std_final**2 - std_init**2
    expected = output_scale**2 * (t1 - t0)
    assert jnp.allclose(received, expected)
