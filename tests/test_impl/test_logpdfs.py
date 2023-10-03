"""Tests for logpdfs.

Necessary because the implementation has been faulty in the past. Never again.
"""

import jax.numpy as jnp
import jax.scipy.stats

from probdiffeq.impl import impl
from tests.setup import setup


def test_logpdf():
    rv = setup.rv()
    (mean_dense, cov_dense) = impl.variable.to_multivariate_normal(rv)

    u = jnp.ones_like(impl.stats.mean(rv))
    u_dense = jnp.ones_like(mean_dense)

    pdf1 = impl.stats.logpdf(u, rv)
    pdf2 = jax.scipy.stats.multivariate_normal.logpdf(u_dense, mean_dense, cov_dense)
    assert jnp.allclose(pdf1, pdf2)


def test_grad_not_none():
    rv = setup.rv()
    u = jnp.ones_like(impl.stats.mean(rv))

    pdf = jax.jacrev(impl.stats.logpdf)(u, rv)
    assert not jnp.any(jnp.isinf(pdf))
    assert not jnp.any(jnp.isnan(pdf))
