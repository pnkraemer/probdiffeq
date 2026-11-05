"""Tests for square-root matrices.

These are so crucial and annoying to debug that they need their own test set.
"""

from probdiffeq.backend import func, linalg, np, random, testing, tree
from probdiffeq.util import cholesky_util

_SHAPES = ([(4, 3), (3, 3), (4, 4)], [(2, 3), (3, 3), (2, 2)])


@testing.parametrize("HCshape, Cshape, Xshape", _SHAPES)
@testing.parametrize("solve_triu", [linalg.solve_triu, linalg.lstsq_svd])
def test_revert_conditional(HCshape, Cshape, Xshape, solve_triu) -> None:
    """Assert that revert_conditional returns the correct gain, joint covariance, and backward noise covariance."""
    HC = _some_array(HCshape) + 1.0
    C = _some_array(Cshape) + 2.0
    X = _some_array(Xshape) + 3.0 + np.eye(Xshape[0])

    S = HC @ HC.T + X @ X.T
    K = C @ HC.T @ linalg.inv(S)
    C1 = C @ C.T - K @ S @ K.T

    extra, (bw_noise, g) = cholesky_util.revert_conditional(
        R_X_F=HC.T, R_X=C.T, R_YX=X.T, solve_triu=solve_triu
    )

    def cov(x):
        return x.T @ x

    assert testing.allclose(cov(extra), S)
    assert testing.allclose(g, K)
    assert testing.allclose(cov(bw_noise), C1)


def _some_array(shape):
    key = random.prng_key(seed=1)
    return random.normal(key, shape=shape)


def test_reverse_conditional_jacrev_zero_matrix() -> None:
    """For zero-valued input covariances, reverse-mode gradients need a trick.

    This resolves issue #668.
    """
    C = _some_array((3, 3)) * 0.0
    HC = _some_array((2, 3)) * 0.0
    X = _some_array((2, 2)) + 3.0 + np.eye(2)

    revert_conditional = func.partial(
        cholesky_util.revert_conditional, solve_triu=linalg.solve_triu
    )
    result = func.jacrev(revert_conditional)(HC.T, C.T, X.T)
    is_not_nan = _tree_is_free_of_nans(result)
    assert is_not_nan


def test_sum_of_sqrtm_factors_jacrev_zero_matrix() -> None:
    """For zero-valued input covariances, reverse-mode gradients need a trick.

    This resolves issue #668.
    """
    C = _some_array((3, 3)) * 0.0
    HC = _some_array((3, 2))

    result = func.jacrev(cholesky_util.sum_of_sqrtm_factors)((C.T, HC.T))
    is_not_nan = _tree_is_free_of_nans(result)
    assert is_not_nan


@testing.parametrize(
    "x, expected",
    [
        ((0.0, 0.0), (1.0, 1.0)),
        ((3.0, 4.0), (0.6, 0.8)),
        ((1e-6, 0.0), (1.0, 0.0)),
        ((0.0, 1e-6), (0.0, 1.0)),
    ],
)
def test_hypot_derivative_well_defined_at_origin(x, expected) -> None:
    """Ensure that the derivative of np.hypot is well-defined at the origin.

    The output-scale calibration evaluates hypot at (0, 0): the running
    output scale starts at zero, and the residual term can be zero, too.
    Reverse-mode derivatives through the solver must not be NaN then (#668).

    Since JAX v0.12, the derivative of jnp.hypot is NaN at the origin
    (before, it was finite). We therefore use a custom JVP, which matches
    the derivative of abs() at zero: it is 1 by convention in JAX, grad(abs)(0) == 1.
    So at the origin, both partial derivatives are 1 (the right-sided limits).
    """

    @func.grad
    def grad_hypot(z):
        return np.hypot(z[0], z[1])

    received = grad_hypot(np.asarray(x))
    assert testing.allclose(received, np.asarray(expected))


def _tree_is_free_of_nans(pytree):
    def contains_no_nan(x):
        return np.logical_not(np.any(np.isnan(x)))

    tree_contains_no_nan = tree.tree_map(contains_no_nan, pytree)
    return tree.tree_all(tree_contains_no_nan)
