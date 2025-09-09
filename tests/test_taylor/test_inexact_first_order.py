"""Tests for inexact approximations for first-order problems."""

from probdiffeq import ivpsolvers, taylor
from probdiffeq.backend import functools, ode, testing, tree_util
from probdiffeq.backend import numpy as np


@testing.parametrize("num", [0, 1, 4])
@testing.parametrize("fact", ["isotropic", "dense", "blockdiag"])
def test_initialised_correct_shape_and_values(num, fact):
    vf, (u0,), (t0, _) = ode.ivp_lotka_volterra()
    vf_autonomous = functools.partial(vf, t=t0)
    solution = taylor.odejet_padded_scan(vf_autonomous, (u0,), num=num)

    tcoeffs_like = [tree_util.tree_map(np.zeros_like, u0)] * (num + 1)
    _init, prior, ssm = ivpsolvers.prior_wiener_integrated(tcoeffs_like, ssm_fact=fact)
    rk_starter = taylor.runge_kutta_starter(dt=0.01, prior=prior, ssm=ssm, num=num)
    derivatives = rk_starter(vf, (u0,), t=t0)
    assert len(derivatives) == 1 + num
    assert testing.tree_all_allclose(derivatives[:1], solution[:1], rtol=1e-1)

    if num > 1:
        assert testing.tree_all_allclose(
            derivatives[: num - 1], solution[: num - 1], rtol=1e-1
        )
