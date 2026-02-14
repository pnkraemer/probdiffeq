"""Tests for inexact approximations for first-order problems."""

from probdiffeq import probdiffeq, taylor
from probdiffeq.backend import functools, np, ode, testing, tree_util


@testing.parametrize("num", [0, 1, 3])
@testing.parametrize("fact", ["isotropic", "dense", "blockdiag"])
def test_initialised_correct_shape_and_values(num, fact):
    vf, (u0,), (t0, _) = ode.ivp_lotka_volterra()
    vf_autonomous = functools.partial(vf, t=t0)
    solution = taylor.odejet_padded_scan(vf_autonomous, (u0,), num=num)

    tcoeffs_like = [tree_util.tree_map(np.zeros_like, u0)] * (num + 1)
    _init, prior, ssm = probdiffeq.prior_wiener_integrated(tcoeffs_like, ssm_fact=fact)
    rk_starter = taylor.runge_kutta_starter(dt=0.01, prior=prior, ssm=ssm, num=num)
    derivatives = rk_starter(vf, (u0,), t=t0)
    assert len(derivatives) == 1 + num
    assert testing.allclose(derivatives[:1], solution[:1], rtol=1e-1)

    if num > 1:
        assert testing.allclose(derivatives[: num - 1], solution[: num - 1], rtol=1e-1)
