"""Tests for inexact approximations for first-order problems."""

from probdiffeq import taylor
from probdiffeq.backend import numpy as np
from probdiffeq.backend import ode, testing
from probdiffeq.impl import impl


@testing.parametrize("num", [1, 4])
@testing.parametrize("fact", ["isotropic"])
def test_initialised_correct_shape_and_values(num, fact):
    vf, (u0,), (t0, _) = ode.ivp_lotka_volterra()

    solution = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), (u0,), num=num)

    tcoeffs = [np.ones((2,))] * (num + 1)
    ssm = impl.choose(fact, tcoeffs_like=tcoeffs)
    rk_starter = taylor.runge_kutta_starter(dt=0.01, ssm=ssm)
    derivatives = rk_starter(lambda y, t: vf(y, t=t), (u0,), t=t0, num=num)

    assert len(derivatives) == len((u0,)) + num
    assert derivatives[0].shape == u0.shape

    for i, (expected, received) in enumerate(zip(derivatives, solution)):
        # Don't compare the highest derivative because the RK starter can't do that
        if i < len(expected):
            # demand at least ~10% accuracy to warn about the most obvious bugs
            assert np.allclose(expected, received, rtol=1e-1), (i, expected, received)
