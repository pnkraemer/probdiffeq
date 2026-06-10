"""Tests for logpdfs.

Necessary because the implementation has been faulty in the past.
"""

from probdiffeq import probdiffeq
from probdiffeq.backend import func, np, random, testing, tree


@testing.parametrize("fact", ["dense", "isotropic", "blockdiag"])
def test_logpdf(fact) -> None:
    rv, _ssm = create_random_variable(fact=fact)

    u = tree.tree_map(np.ones_like, rv.mean)

    (mean_dense, cov_dense) = rv.to_multivariate_normal()
    u_dense = np.ones_like(mean_dense)

    pdf1 = rv.logpdf_tree(u)
    pdf2 = random.logpdf_multivariate_normal(u_dense, mean_dense, cov_dense)
    assert testing.allclose(pdf1, pdf2)


@testing.parametrize("fact", ["dense", "isotropic", "blockdiag"])
def test_grad_not_none(fact) -> None:
    rv, _ssm = create_random_variable(fact=fact)
    u = tree.tree_map(np.ones_like, rv.mean)

    pdf = func.jacrev(lambda x, y: y.logpdf_tree(x))(u, rv)
    pdf, _ = tree.ravel_pytree(pdf)

    assert not np.any(np.isinf(pdf))
    assert not np.any(np.isnan(pdf))


def create_random_variable(fact):
    tcoeffs = [np.ones((3,))] * 5  # values irrelevant
    ssm = probdiffeq.state_space_model(fact)
    _, discretize = ssm.prior_wiener_integrated(
        tcoeffs, is_exact=False, inexact_eps=1.0
    )

    if fact == "dense":
        output_scale = np.ones(())
        rv = discretize(0.1, output_scale)
    elif fact == "blockdiag":
        output_scale = np.ones((3,))
        rv = discretize(0.1, output_scale)
    elif fact == "isotropic":
        output_scale = np.ones(())
        rv = discretize(0.1, output_scale)
    else:
        raise ValueError
    key = random.prng_key(seed=1)
    noise_flat, unravel = tree.ravel_pytree(rv.noise)
    noise_flat = random.normal(key, shape=noise_flat.shape)
    noise = unravel(noise_flat)
    return noise, ssm
