"""Tests for logpdfs.

Necessary because the implementation has been faulty in the past.
"""

from probdiffeq import ssm_impl
from probdiffeq.backend import func, np, random, testing, tree


@testing.parametrize("fact", ["dense", "isotropic", "blockdiag"])
def test_logpdf(fact) -> None:
    rv, _ssm = create_random_variable(fact=fact)

    u = tree.tree_map(np.ones_like, rv.mean_tree)

    (mean_dense, cov_dense) = rv.to_multivariate_normal()
    u_dense = np.ones_like(mean_dense)

    pdf1 = rv.logpdf_tree(u)
    pdf2 = random.logpdf_multivariate_normal(u_dense, mean_dense, cov_dense)
    assert testing.allclose(pdf1, pdf2)


@testing.parametrize("fact", ["dense", "isotropic", "blockdiag"])
def test_grad_not_none(fact) -> None:
    rv, _ssm = create_random_variable(fact=fact)
    u = tree.tree_map(np.ones_like, rv.mean_tree)

    pdf = func.jacrev(lambda x, y: y.logpdf_tree(x))(u, rv)
    pdf, _ = tree.ravel_pytree(pdf)

    assert not np.any(np.isinf(pdf))
    assert not np.any(np.isnan(pdf))


def create_random_variable(fact):
    tcoeffs = [np.ones((3,))] * 5  # values irrelevant

    if fact == "dense":
        stds = [np.ones((3,))] * 5
        _, ssm = ssm_impl.FactSsmImpl.from_tcoeffs_dense(tcoeffs, stds)
        output_scale = np.ones((3,))
        discretize = ssm.prior.transition_wiener_integrated(output_scale)

        output_scale = np.ones(())
        rv = discretize(0.1, output_scale)
    elif fact == "blockdiag":
        stds = [np.ones((3,))] * 5
        _, ssm = ssm_impl.FactSsmImpl.from_tcoeffs_blockdiag(tcoeffs, stds)
        output_scale = np.ones((3,))
        discretize = ssm.prior.transition_wiener_integrated(output_scale)

        output_scale = np.ones((3,))
        rv = discretize(0.1, output_scale)
    elif fact == "isotropic":
        stds = [np.ones(())] * 5
        _, ssm = ssm_impl.FactSsmImpl.from_tcoeffs_isotropic(tcoeffs, stds)
        output_scale = np.ones(())
        discretize = ssm.prior.transition_wiener_integrated(output_scale)

        output_scale = np.ones(())
        rv = discretize(0.1, output_scale)
    else:
        raise ValueError
    key = random.prng_key(seed=1)
    noise_flat, unravel = tree.ravel_pytree(rv.noise)
    noise_flat = random.normal(key, shape=noise_flat.shape)
    noise = unravel(noise_flat)
    return noise, ssm
