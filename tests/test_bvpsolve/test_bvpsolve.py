"""Tests for BVP solver."""

import diffeqzoo.bvps
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from probdiffeq import bvpsolve
from probdiffeq.statespace.scalar import extra


def test_ibm_discretised(num_derivatives=1, reverse=False):
    """Solve a second-order, scalar, linear, separable BVP."""
    output_scale = 1.0
    t0, t1 = 0.0, 3.4123412
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


def _marginal_moments(init, precons, transitions, *, reverse):
    def step(carry, input):
        trans, prec = input
        rv = extra.extrapolate_precon(carry, trans, prec)
        return rv, rv

    _, rvs = jax.lax.scan(step, init=init, xs=(transitions, precons), reverse=reverse)
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


def test_bridge():
    vf, (g0, g1), (t0, t1), params = diffeqzoo.bvps.pendulum()

    def vf_partial(u, /):
        return vf(u, *params)

    grid = jnp.linspace(t0, t1, num=10)
    solution = bvpsolve.solve_fixed_grid(vf_partial, bcond=(g0, g1), grid=grid)

    init, (process_noise, transitions) = solution
    print(init)
    print(process_noise)
    print(transitions)
    assert False
