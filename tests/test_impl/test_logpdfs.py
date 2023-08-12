"""Tests for logpdfs.

Necessary because the implementation has been faulty in the past. Never again.
"""

import jax.numpy as jnp
import jax.scipy.stats

from probdiffeq.backend import testing
from probdiffeq.impl import impl
from tests.setup import setup


@testing.fixture(name="setup")
@testing.parametrize("d", [1, 10])
def fixture_setup(d):
    m = jnp.arange(1.0, 1.0 + d)

    X = m[:, None] * m[None, :] + jnp.eye(d)
    # X *= jnp.arange(d)[:, None]  # singular

    cov_cholesky = 1e-10 * jnp.linalg.qr(X, mode="r").T
    return m, cov_cholesky


def test_logpdf():
    u, rv = setup.rv()
    u_dense, (mean_dense, cov_dense) = impl.variable.to_multivariate_normal(u, rv)
    pdf1 = impl.stats.logpdf(u, rv)
    pdf2 = jax.scipy.stats.multivariate_normal.logpdf(u_dense, mean_dense, cov_dense)
    assert jnp.allclose(pdf1, pdf2)


def test_grad_not_none():
    u, rv = setup.rv()
    pdf = jax.jacrev(impl.stats.logpdf)(u, rv)
    assert not jnp.any(jnp.isinf(pdf))
    assert not jnp.any(jnp.isnan(pdf))
