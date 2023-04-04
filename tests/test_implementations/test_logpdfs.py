"""Tests for logpdfs.

Necessary because the implementation has been faulty in the past. Never again.
"""

import jax.numpy as jnp
import jax.scipy.stats

from probdiffeq.implementations import _scalar
from probdiffeq.implementations.dense import _vars as vars_dense
from probdiffeq.implementations.iso import _vars as vars_iso


def test_logpdf_dense():
    u = jnp.arange(1.0, 4.0)
    m = u + 1.0
    cov_cholesky = u[:, None] * u[None, :] + jnp.eye(3)
    pdf1 = vars_dense.DenseNormal(m, cov_cholesky).logpdf(u)
    pdf2 = jax.scipy.stats.multivariate_normal.logpdf(
        u, mean=m, cov=cov_cholesky @ cov_cholesky.T
    )
    assert jnp.allclose(pdf1, pdf2)


def test_logpdf_iso():
    u = jnp.arange(1.0, 4.0)
    m = u + 1.0
    cov_cholesky = 4.0
    pdf1 = vars_iso.IsoNormalQOI(m, cov_cholesky).logpdf(u)
    pdf2 = jax.scipy.stats.multivariate_normal.logpdf(
        u, mean=m, cov=cov_cholesky**2 * jnp.eye(3)
    )
    assert jnp.allclose(pdf1, pdf2)


def test_logpdf_scalar():
    pdf1 = _scalar.NormalQOI(2.0, 3.0).logpdf(5.0)
    pdf2 = jax.scipy.stats.multivariate_normal.logpdf(5.0, mean=2.0, cov=3.0**2)
    assert jnp.allclose(pdf1, pdf2)
