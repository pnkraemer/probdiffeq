"""Tests for BVP solver."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from probdiffeq import bvpsolve
from probdiffeq.statespace.scalar import extra


def test_ibm_discretised(num_derivatives=1, reverse=False):
    """Solve a second-order, scalar, linear, separable BVP."""
    output_scale = 10.0
    t1, t0 = 0.0, 3.4123412
    grid = jnp.linspace(t0, t1, endpoint=True, num=20)

    init, transitions, precons = bvpsolve.ibm_prior_discretised(
        grid, num_derivatives=num_derivatives, output_scale=output_scale
    )

    means, stds = _marginal_moments(init, precons, transitions, reverse=reverse)

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


def test_bridge(num_derivatives=4):
    # Choose t0 > t1 to reverse-pass initially
    t0 = -3.0
    t1 = 3.0

    def g0(x):
        return x - 2.0

    def g1(x):
        return x + 400.0

    output_scale = 10.0
    grid = jnp.linspace(t0, t1, endpoint=True, num=100)

    prior = bvpsolve.ibm_prior_discretised(
        grid, num_derivatives=num_derivatives, output_scale=output_scale
    )

    prior_bridge = bvpsolve.bridge(
        (g0, g1),
        prior,
        num_derivatives=num_derivatives,
        reverse=False,
    )

    (init, transitions, precons) = prior_bridge
    means, stds = _marginal_moments(init, precons, transitions, reverse=True)

    fig, axes = plt.subplots(ncols=num_derivatives + 1, sharey=True, sharex=True)

    for m, s, ax in zip(means.T, stds.T, axes):
        ax.plot(grid[1:], m)
        ax.fill_between(grid[1:], m - s, m + s, alpha=0.2)

    plt.show()
    assert False


def test_solve_separable_affine_2nd(num_derivatives=4):
    # Set up a prior
    grid = jnp.linspace(0.0, 1.0, endpoint=True, num=20)
    prior = extra.ibm_discretise_fwd(jnp.diff(grid), num_derivatives=num_derivatives)

    # Set up a problem
    eps = 1e-2
    g0, g1 = (1.0, -1.0), (1.0, 0.0)
    ode = (jnp.ones_like(grid) / eps, jnp.zeros_like(grid))
    solution = bvpsolve.solve_separable_affine_2nd(ode, bcond=(g0, g1), prior=prior)

    def true_sol(t):
        a = jnp.exp(-t / jnp.sqrt(eps))
        b = jnp.exp((t - 2.0) / jnp.sqrt(eps))
        c = jnp.exp(-2.0 / jnp.sqrt(eps))
        return (a - b) / (1 - c)

    means, stds = _marginal_moments(solution)
    assert jnp.allclose(means[:, 0], true_sol(grid[1:]), atol=1e-3)


def _marginal_moments(precon_mseq):
    def step(carry, input):
        trans, prec = input
        rv = extra.extrapolate_precon(carry, trans, prec)
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
