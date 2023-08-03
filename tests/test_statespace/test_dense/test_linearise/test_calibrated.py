import functools

import jax.numpy as jnp

from probdiffeq.backend import testing
from probdiffeq.statespace import cubature
from probdiffeq.statespace.dense import linearise, variables


@testing.fixture(name="setup")
def fixture_setup():
    def vf(y, /):
        return y * (1.0 - y)

    x0 = jnp.asarray([0.7])
    return vf, x0


def test_linearisation_calibrated(linearisation, truth):
    return jnp.allclose(linearisation, truth)
