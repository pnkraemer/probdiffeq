import functools

import jax.numpy as jnp

from probdiffeq.backend import testing
from probdiffeq.statespace import cubature
from probdiffeq.statespace.dense import linearise, variables


@testing.case()
def case_ts0():
    vf, x0 = _setup()
    b = linearise.ts0(vf, x0)
    return b, vf(x0)


@testing.case()
def case_slr0(noise=1e-5):
    vf, x0 = _setup()
    rv0 = variables.DenseNormal(x0, jnp.eye(1) * noise, target_shape=None)

    b = linearise.slr0(vf, rv0, cubature_rule=cubature.gauss_hermite(input_shape=(1,)))
    return b.mean, vf(x0)


def _setup():
    def vf(y, /):
        return y * (1.0 - y)

    x0 = jnp.asarray([0.7])
    return vf, x0


@testing.parametrize_with_cases("a, b", cases=".", prefix="case_")
def test_linearisation_allclose_0th(a, b):
    assert jnp.allclose(a, b)
