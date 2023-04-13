"""Tests for logpdfs.

Necessary because the implementation has been faulty in the past. Never again.
"""

import jax.numpy as jnp
import jax.scipy.stats

from probdiffeq.backend import testing
from probdiffeq.statespace.dense import _vars as vars_dense
from probdiffeq.statespace.iso import _vars as vars_iso
from probdiffeq.statespace.scalar import _vars as vars_scalar


@testing.fixture(name="setup")
@testing.parametrize("d", [1, 10])
def fixture_setup(d):
    m = jnp.arange(1.0, 1.0 + d)

    X = m[:, None] * m[None, :] + jnp.eye(d)
    # X *= jnp.arange(d)[:, None]  # singular

    cov_cholesky = 1e-10 * jnp.linalg.qr(X, mode="r").T
    return m, cov_cholesky


def test_logpdf_dense(setup):
    m, cov_cholesky = setup

    def fn1(x):
        return vars_dense.DenseNormal(m, cov_cholesky).logpdf(x)

    def fn2(x):
        return jax.scipy.stats.multivariate_normal.logpdf(
            x, mean=m, cov=cov_cholesky @ cov_cholesky.T
        )

    u = m + 1e-3
    pdf1 = fn1(u)
    pdf2 = fn2(u)
    assert jnp.allclose(pdf1, pdf2)

    pdf1 = jax.grad(fn1)(u)
    pdf2 = jax.grad(fn2)(u)
    assert jnp.allclose(pdf1, pdf2)


def test_logpdf_iso(setup):
    m, cov_cholesky = setup
    variance = jnp.trace(cov_cholesky)

    def fn1(x):
        return vars_iso.IsoNormalQOI(m, variance).logpdf(x)

    def fn2(x):
        return jax.scipy.stats.multivariate_normal.logpdf(
            x, mean=m, cov=variance**2 * jnp.eye(len(m))
        )

    u = m + 1e-3
    pdf1 = fn1(u)
    pdf2 = fn2(u)
    assert jnp.allclose(pdf1, pdf2)

    pdf1 = jax.grad(fn1)(u)
    pdf2 = jax.grad(fn2)(u)
    assert jnp.allclose(pdf1, pdf2)


def test_logpdf_scalar(setup):
    m, cov_cholesky = setup
    mean = jnp.linalg.norm(m)
    variance = jnp.trace(cov_cholesky)

    def fn1(x):
        return vars_scalar.NormalQOI(mean, variance).logpdf(x)

    def fn2(x):
        logpdf_fn = jax.scipy.stats.multivariate_normal.logpdf
        return logpdf_fn(x, mean=mean, cov=variance**2)

    u = mean + 1e-3
    pdf1 = fn1(u)
    pdf2 = fn2(u)
    assert jnp.allclose(pdf1, pdf2)

    pdf1 = jax.grad(fn1)(mean)
    pdf2 = jax.grad(fn2)(mean)
    assert jnp.allclose(pdf1, pdf2)
