"""Tests for logpdfs.

Necessary because the implementation has been faulty in the past. Never again.
"""

from probdiffeq.backend import functools, stats
from probdiffeq.backend import numpy as np
from probdiffeq.impl import impl


def test_logpdf(ssm):
    rv = ssm.default_rv

    (mean_dense, cov_dense) = impl.stats.to_multivariate_normal(rv)

    u = np.ones_like(impl.stats.mean(rv))
    u_dense = np.ones_like(mean_dense)

    pdf1 = impl.stats.logpdf(u, rv)
    pdf2 = stats.multivariate_normal_logpdf(u_dense, mean_dense, cov_dense)
    assert np.allclose(pdf1, pdf2)


def test_grad_not_none(ssm):
    rv = ssm.default_rv
    u = np.ones_like(impl.stats.mean(rv))

    pdf = functools.jacrev(impl.stats.logpdf)(u, rv)
    assert not np.any(np.isinf(pdf))
    assert not np.any(np.isnan(pdf))
