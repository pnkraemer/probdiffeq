"""Tests for logpdfs.

Necessary because the implementation has been faulty in the past. Never again.
"""

from probdiffeq.backend import functools, random, stats, testing, tree_util
from probdiffeq.backend import numpy as np
from probdiffeq.impl import impl


@testing.parametrize("fact", ["dense", "isotropic", "blockdiag"])
def test_logpdf(fact):
    rv, ssm = random_variable(fact=fact)

    u = np.ones_like(ssm.stats.mean(rv))

    (mean_dense, cov_dense) = ssm.stats.to_multivariate_normal(rv)
    u_dense = np.ones_like(mean_dense)

    pdf1 = ssm.stats.logpdf(u, rv)
    pdf2 = stats.multivariate_normal_logpdf(u_dense, mean_dense, cov_dense)
    assert testing.allclose(pdf1, pdf2)


@testing.parametrize("fact", ["dense", "isotropic", "blockdiag"])
def test_grad_not_none(fact):
    rv, ssm = random_variable(fact=fact)
    u = np.ones_like(ssm.stats.mean(rv))

    pdf = functools.jacrev(ssm.stats.logpdf)(u, rv)
    assert not np.any(np.isinf(pdf))
    assert not np.any(np.isnan(pdf))


def random_variable(fact):
    tcoeffs = [np.ones((3,))] * 5  # values irrelevant
    ssm = impl.choose(fact, tcoeffs_like=tcoeffs)
    output_scale = np.ones_like(ssm.prototypes.output_scale())
    discretize = ssm.conditional.ibm_transitions(output_scale)
    rv = discretize(0.1, output_scale)

    key = random.prng_key(seed=1)
    noise_flat, unravel = tree_util.ravel_pytree(rv.noise)
    noise_flat = random.normal(key, shape=noise_flat.shape)
    noise = unravel(noise_flat)
    return noise, ssm
