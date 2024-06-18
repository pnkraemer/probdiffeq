"""Test the discrete IBM transitions."""

from probdiffeq.backend import control_flow, functools
from probdiffeq.backend import numpy as np
from probdiffeq.impl import impl


def test_marginal_moments_are_correct(ssm, num_derivatives=2):  # noqa: ARG001
    """Solve a second-order, scalar, linear, separable BVP."""
    output_scale = 10.0 * np.ones_like(impl.prototypes.output_scale())
    t0, t1 = 0.0, 3.4123412
    grid = np.linspace(t0, t1, endpoint=True, num=20)

    init = impl.normal.standard(num_derivatives + 1, output_scale)
    discretise = impl.conditional.ibm_transitions(num_derivatives, output_scale)
    transitions = functools.vmap(discretise)(np.diff(grid))

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
        rv = impl.normal.preconditioner_apply(rv, p_inv)
        rv = impl.conditional.marginalise(rv, cond)
        rv = impl.normal.preconditioner_apply(rv, p)
        return rv, rv

    _, rvs = control_flow.scan(step, init=init, xs=transitions, reverse=False)
    means = impl.stats.mean(rvs)
    # todo: does this conflict with error estimation?
    stds = impl.stats.standard_deviation(rvs)

    return means, stds


def _assert_zero_mean(means):
    assert np.allclose(means, 0.0)


def _assert_monotonously_increasing_std(stds):
    diffs = np.diff_along_axis(stds, axis=0)
    assert np.all(diffs > 0), diffs


def _assert_brownian_motion_std(std_final, std_init, t0, t1, *, output_scale):
    received = std_final**2 - std_init**2
    expected = output_scale**2 * (t1 - t0)
    assert np.allclose(received, expected)
