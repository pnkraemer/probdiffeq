"""Tests for logpdfs.

Necessary because the implementation has been faulty in the past. Never again.
"""

import jax.numpy as jnp
import jax.scipy.stats

from probdiffeq.backend import statespace, testing


@testing.fixture(name="setup")
@testing.parametrize("d", [1, 10])
def fixture_setup(d):
    m = jnp.arange(1.0, 1.0 + d)

    X = m[:, None] * m[None, :] + jnp.eye(d)
    # X *= jnp.arange(d)[:, None]  # singular

    cov_cholesky = 1e-10 * jnp.linalg.qr(X, mode="r").T
    return m, cov_cholesky


def test_logpdf_dense(setup):
    statespace.select("dense")
    m, cov_cholesky = setup

    def fn1(x):
        rv = statespace.random.variable(mean, cholesky)
        return statespace.random.logpdf(x, rv)

    def fn2(x):
        return jax.scipy.stats.multivariate_normal.logpdf(
            x, mean=mean, cov=cholesky @ cholesky.T
        )

    u = m + 1e-3
    pdf1 = fn1(u)
    pdf2 = fn2(u)
    assert jnp.allclose(pdf1, pdf2)

    pdf1 = jax.grad(fn1)(u)
    pdf2 = jax.grad(fn2)(u)
    assert jnp.allclose(pdf1, pdf2)


def test_logpdf_iso(setup):
    statespace.select("isotropic")

    mean, cholesky = setup
    standard_deviation = jnp.trace(cov_cholesky)

    def fn1(x):
        rv = statespace.random.variable(mean, standard_deviation)
        return statespace.random.logpdf(x, rv)

    def fn2(x):
        return jax.scipy.stats.multivariate_normal.logpdf(
            x, mean=mean, cov=standard_deviation**2 * jnp.eye(len(m))
        )

    u = m + 1e-3
    pdf1 = fn1(u)
    pdf2 = fn2(u)
    assert jnp.allclose(pdf1, pdf2)

    pdf1 = jax.grad(fn1)(u)
    pdf2 = jax.grad(fn2)(u)
    assert jnp.allclose(pdf1, pdf2)


def test_logpdf_scalar(setup):
    statespace.select("scalar")

    m, cov_cholesky = setup
    mean = jnp.linalg.norm(m)
    standard_deviation = jnp.trace(cov_cholesky)

    def fn1(x):
        rv = statespace.random.variable(mean, standard_deviation)
        return statespace.random.logpdf(x, rv)

    def fn2(x):
        return jax.scipy.stats.multivariate_normal.logpdf(
            x, mean=mean, cov=standard_deviation**2
        )

    u = m + 1e-3
    pdf1 = fn1(u)
    pdf2 = fn2(u)
    assert jnp.allclose(pdf1, pdf2)

    pdf1 = jax.grad(fn1)(u)
    pdf2 = jax.grad(fn2)(u)
    assert jnp.allclose(pdf1, pdf2)
