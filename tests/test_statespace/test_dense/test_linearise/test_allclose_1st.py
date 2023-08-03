import functools

import jax.numpy as jnp

from probdiffeq.backend import testing
from probdiffeq.statespace import cubature
from probdiffeq.statespace.dense import linearise, linearise_ode, variables


@testing.case()
def case_ts1():
    vf, x0 = _setup()
    A, b = linearise.ts1(vf, x0)
    return A(x0) + b, vf(x0)


@testing.case()
def case_slr1(noise=1e-5):
    vf, x0 = _setup()
    rv0 = variables.DenseNormal(x0, jnp.eye(1) * noise, target_shape=None)
    cubature_rule = cubature.gauss_hermite(input_shape=(1,))
    A, b = linearise.slr1(vf, rv0, cubature_rule=cubature_rule)
    return A(x0) + b.mean, vf(x0)


@testing.case()
def case_ode_1st_statistical(noise=1e-5):
    vf, x0 = _setup()
    rv0 = variables.DenseNormal(x0, jnp.eye(1) * noise, target_shape=None)

    cubature_rule = cubature.gauss_hermite(input_shape=(1,))
    fun = functools.partial(linearise.slr1, cubature_rule=cubature_rule)
    ode_linearise = linearise_ode.constraint_1st_statistical(
        fun, ode_shape=(1,), ode_order=1
    )
    A, bias = ode_linearise(vf, rv0)

    rv = variables.DenseNormal(
        jnp.concatenate([x0] * 3), jnp.eye(3), target_shape=(3, 1)
    )
    return A(rv.mean) + bias.mean, jnp.atleast_1d(rv.mean[1] - vf(rv.mean[0]))


@testing.case()
def case_ode_1st():
    vf, x0 = _setup()

    fun = linearise.ts1
    ode_linearise = linearise_ode.constraint_1st(fun, ode_shape=(1,), ode_order=1)
    A, bias = ode_linearise(vf, x0)

    m0 = jnp.concatenate([x0] * 3)
    return A(m0) + bias, jnp.atleast_1d(m0[1] - vf(m0[0]))


@testing.case()
def case_ode_1st_second_order():
    vf, x0 = _setup_2nd()

    fun = linearise.ts1
    ode_linearise = linearise_ode.constraint_1st(fun, ode_shape=(1,), ode_order=2)
    A, bias = ode_linearise(vf, x0)

    m0 = jnp.concatenate([jnp.concatenate(x0)] * 3)
    return A(m0) + bias, jnp.atleast_1d(m0[2] - vf(m0[:2]))


@testing.case()
def case_ode_0th_statistical(noise=1e-5):
    vf, x0 = _setup()
    rv0 = variables.DenseNormal(x0, jnp.eye(1) * noise, target_shape=None)

    cubature_rule = cubature.gauss_hermite(input_shape=(1,))
    fun = functools.partial(linearise.slr0, cubature_rule=cubature_rule)
    ode_linearise = linearise_ode.constraint_0th_statistical(
        fun, ode_shape=(1,), ode_order=1
    )
    A, bias = ode_linearise(vf, rv0)

    rv = variables.DenseNormal(
        jnp.concatenate([x0] * 3), jnp.eye(3), target_shape=(3, 1)
    )
    return A(rv.mean) + bias.mean, jnp.atleast_1d(rv.mean[1] - vf(rv.mean[0]))


@testing.case()
def case_ode_0th():
    vf, x0 = _setup()

    fun = linearise.ts0
    ode_linearise = linearise_ode.constraint_0th(fun, ode_shape=(1,), ode_order=1)
    A, bias = ode_linearise(vf, x0)

    m0 = jnp.concatenate([x0] * 3)
    return A(m0) + bias, jnp.atleast_1d(m0[1] - vf(m0[0]))


@testing.case()
def case_ode_0th_second_order():
    vf, x0 = _setup_2nd()

    fun = linearise.ts0
    ode_linearise = linearise_ode.constraint_0th(fun, ode_shape=(1,), ode_order=2)
    A, bias = ode_linearise(vf, x0)

    m0 = jnp.concatenate([jnp.concatenate(x0)] * 3)
    return A(m0) + bias, jnp.atleast_1d(m0[2] - vf((m0[0], m0[1])))


def _setup():
    def vf(y, /):
        return y * (1.0 - y)

    x0 = jnp.asarray([0.7])
    return vf, x0


def _setup_2nd():
    def vf(y_and_dy, /):
        y, dy = y_and_dy
        return y * (1.0 - dy**2)

    x0 = jnp.asarray([0.7])
    return vf, jnp.stack((x0, x0 + 1.0))


@testing.parametrize_with_cases("a, b", cases=".", prefix="case_")
def test_allclose(a, b):
    assert jnp.shape(a) == jnp.shape(b)
    assert jnp.allclose(a, b)
