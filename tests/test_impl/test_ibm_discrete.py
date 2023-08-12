"""Test the discrete IBM transitions."""

import jax
import jax.numpy as jnp

from probdiffeq.impl import impl


def test_marginal_moments_are_correct(num_derivatives=2):
    """Solve a second-order, scalar, linear, separable BVP."""
    output_scale = 10.0 * jnp.ones_like(impl.prototypes.output_scale())
    t0, t1 = 0.0, 3.4123412
    grid = jnp.linspace(t0, t1, endpoint=True, num=20)

    init = impl.ssm_util.standard_normal(num_derivatives + 1, output_scale)
    discretise = impl.ssm_util.ibm_transitions(num_derivatives, output_scale)
    transitions = jax.vmap(discretise)(jnp.diff(grid))

    means, stds = _marginal_moments(init, transitions)

    _assert_zero_mean(means)
    _assert_monotonously_increasing_std(stds)
    _assert_brownian_motion_std(
        std_final=stds[-1, ..., -1],
        std_init=output_scale,
        t0=t0,
        t1=t1,
        output_scale=output_scale,
    )


def _marginal_moments(init, transitions):
    def step(rv, model):
        cond, (p, p_inv) = model
        rv = impl.ssm_util.preconditioner_apply(rv, p_inv)
        rv = impl.conditional.marginalise(rv, cond)
        rv = impl.ssm_util.preconditioner_apply(rv, p)
        return rv, rv

    _, rvs = jax.lax.scan(step, init=init, xs=transitions, reverse=False)
    means = impl.stats.mean(rvs)
    print(jax.tree_util.tree_map(jnp.shape, rvs))
    # todo: does this conflict with error estimation?
    stds = impl.stats.standard_deviation(rvs)

    return means, stds


def _assert_zero_mean(means):
    assert jnp.allclose(means, 0.0)


def _assert_monotonously_increasing_std(stds):
    diffs = jnp.diff(stds, axis=0)
    assert jnp.all(diffs > 0), diffs


def _assert_brownian_motion_std(std_final, std_init, t0, t1, *, output_scale):
    received = std_final**2 - std_init**2
    expected = output_scale**2 * (t1 - t0)
    assert jnp.allclose(received, expected)
