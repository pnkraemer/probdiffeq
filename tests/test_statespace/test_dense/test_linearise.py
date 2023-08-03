"""Tests for linearisation.

Technically, this is tested already (indirectly);
but debugging is a lot easier with separate tests.
"""
import jax.numpy as jnp

from probdiffeq.statespace import cubature
from probdiffeq.statespace.dense import linearise, variables


def test_output_format_ts1():
    def vf(y, /):
        return y * (1. - y)

    x0 = jnp.asarray([0.7])
    A, b = linearise.ts1(vf, x0)

    # Expect: Ax + b ~ f(x), not A(x-x0)+b ~(fx). for x=x0, linearisation is exact.
    assert jnp.allclose(A(x0) + b, vf(x0))
    assert not jnp.allclose(A(x0 - x0) + b, vf(x0))


def test_output_format_ts0():
    def vf(y, /):
        return y * (1. - y)

    x0 = jnp.asarray([0.7])
    b = linearise.ts0(vf, x0)
    assert jnp.allclose(b, vf(x0))


def test_output_format_slr0_almost_exact(noise=1e-5):
    def vf(y, /):
        return y * (1. - y)

    x0 = jnp.asarray([0.7])
    rv0 = variables.DenseNormal(x0, jnp.eye(1) * noise, target_shape=None)

    b = linearise.slr0(vf, rv0, cubature_rule=cubature.gauss_hermite(input_shape=(1,)))
    assert jnp.allclose(b.mean, vf(x0))


def test_output_format_slr0_inexact_but_calibrated(noise=1e-1):
    def vf(y, /):
        return y * (1. - y)

    x0 = jnp.asarray([0.7])
    rv0 = variables.DenseNormal(x0, jnp.eye(1) * noise, target_shape=None)

    b = linearise.slr0(vf, rv0, cubature_rule=cubature.gauss_hermite(input_shape=(1,)))

    error = jnp.abs(b.mean - vf(rv0.mean)) / jnp.abs(b.cov_sqrtm_lower)
    assert 0.1 < error < 10.0
