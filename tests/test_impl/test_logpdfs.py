"""Tests for logpdfs.

Necessary because the implementation has been faulty in the past. Never again.
"""

from probdiffeq.backend import func, np, random, stats, testing, tree
from probdiffeq.impl import impl


@testing.parametrize("fact", ["dense", "isotropic", "blockdiag"])
def test_logpdf(fact):
    rv, ssm = create_random_variable(fact=fact)

    u = tree.tree_map(np.ones_like, rv.eval_mean())

    (mean_dense, cov_dense) = rv.to_multivariate_normal()
    u_dense = np.ones_like(mean_dense)

    pdf1 = rv.logpdf(u)
    pdf2 = stats.multivariate_normal_logpdf(u_dense, mean_dense, cov_dense)
    assert testing.allclose(pdf1, pdf2)


@testing.parametrize("fact", ["dense", "isotropic", "blockdiag"])
def test_grad_not_none(fact):
    rv, ssm = create_random_variable(fact=fact)
    u = tree.tree_map(np.ones_like, rv.eval_mean())

    pdf = func.jacrev(lambda x, y: y.logpdf(x))(u, rv)
    pdf, _ = tree.ravel_pytree(pdf)

    assert not np.any(np.isinf(pdf))
    assert not np.any(np.isnan(pdf))


def create_random_variable(fact):
    tcoeffs = [np.ones((3,))] * 5  # values irrelevant
    ssm = impl.choose(fact, tcoeffs_like=tcoeffs)

    output_scale = np.ones_like(ssm.prototypes.error_estimate())
    discretize = ssm.conditional.ibm_transitions(output_scale)

    output_scale = np.ones_like(ssm.prototypes.output_scale())
    rv = discretize(0.1, output_scale)

    key = random.prng_key(seed=1)
    noise_flat, unravel = tree.ravel_pytree(rv.noise)
    noise_flat = random.normal(key, shape=noise_flat.shape)
    noise = unravel(noise_flat)
    return noise, ssm
