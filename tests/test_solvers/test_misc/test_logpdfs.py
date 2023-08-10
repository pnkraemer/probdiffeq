"""Tests for logpdfs.

Necessary because the implementation has been faulty in the past. Never again.
"""

import jax.numpy as jnp
import jax.scipy.stats

from probdiffeq.backend import testing
from probdiffeq.impl import impl


@testing.fixture(name="setup")
@testing.parametrize("d", [1, 10])
def fixture_setup(d):
    m = jnp.arange(1.0, 1.0 + d)

    X = m[:, None] * m[None, :] + jnp.eye(d)
    # X *= jnp.arange(d)[:, None]  # singular

    cov_cholesky = 1e-10 * jnp.linalg.qr(X, mode="r").T
    return m, cov_cholesky


def test_logpdf_dense(setup):
    mean, cholesky = setup
    impl.select("dense", ode_shape=jnp.shape(mean))

    def fn1(x):
        rv = impl.random.variable(mean, cholesky)
        return impl.random.logpdf(x, rv)

    def fn2(x):
        return jax.scipy.stats.multivariate_normal.logpdf(
            x, mean=mean, cov=cholesky @ cholesky.T
        )

    u = mean + 1e-3
    pdf1 = fn1(u)
    pdf2 = fn2(u)
    assert jnp.allclose(pdf1, pdf2)

    pdf1 = jax.grad(fn1)(u)
    pdf2 = jax.grad(fn2)(u)
    assert jnp.allclose(pdf1, pdf2)


def test_logpdf_iso(setup):
    mean, cholesky = setup
    impl.select("isotropic", ode_shape=jnp.shape(mean))

    standard_deviation = jnp.trace(cholesky)

    def fn1(x):
        rv = impl.random.variable(mean, standard_deviation)
        return impl.random.logpdf(x, rv)

    def fn2(x):
        return jax.scipy.stats.multivariate_normal.logpdf(
            x, mean=mean, cov=standard_deviation**2 * jnp.eye(len(mean))
        )

    u = mean + 1e-3
    pdf1 = fn1(u)
    pdf2 = fn2(u)
    assert jnp.allclose(pdf1, pdf2)

    pdf1 = jax.grad(fn1)(u)
    pdf2 = jax.grad(fn2)(u)
    assert jnp.allclose(pdf1, pdf2)


def test_logpdf_blockdiag(setup):
    mean, cholesky = setup
    impl.select("blockdiag", ode_shape=jnp.shape(mean))

    standard_deviation = jnp.diagonal(cholesky)

    def fn1(x):
        rv = impl.random.variable(mean, standard_deviation)
        return impl.random.logpdf(x, rv)

    def fn2(x):
        return jax.scipy.stats.multivariate_normal.logpdf(
            x, mean=mean, cov=jnp.diag(standard_deviation**2)
        )

    u = mean + 1e-3
    pdf1 = fn1(u)
    pdf2 = fn2(u)
    assert jnp.allclose(pdf1, pdf2)

    pdf1 = jax.grad(fn1)(u)
    pdf2 = jax.grad(fn2)(u)
    assert jnp.allclose(pdf1, pdf2)


def test_logpdf_scalar(setup):
    m, cov_cholesky = setup
    mean = jnp.linalg.norm(m)
    impl.select("scalar", ode_shape=jnp.shape(mean))

    standard_deviation = jnp.trace(cov_cholesky)

    def fn1(x):
        rv = impl.random.variable(mean, standard_deviation)
        return impl.random.logpdf(x, rv)

    def fn2(x):
        return jax.scipy.stats.multivariate_normal.logpdf(
            x, mean=mean, cov=standard_deviation**2
        )

    u = m[0] + 1e-3
    pdf1 = fn1(u)
    pdf2 = fn2(u)
    assert jnp.allclose(pdf1, pdf2)

    pdf1 = jax.grad(fn1)(u)
    pdf2 = jax.grad(fn2)(u)
    assert jnp.allclose(pdf1, pdf2)
