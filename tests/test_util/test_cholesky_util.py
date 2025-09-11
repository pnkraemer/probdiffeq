"""Tests for square-root matrices.

These are so crucial and annoying to debug that they need their own test set.
"""

from probdiffeq.backend import functools, linalg, random, testing, tree_util
from probdiffeq.backend import numpy as np
from probdiffeq.util import cholesky_util

_SHAPES = ([(4, 3), (3, 3), (4, 4)], [(2, 3), (3, 3), (2, 2)])


@testing.parametrize("HCshape, Cshape, Xshape", _SHAPES)
def test_revert_conditional(HCshape, Cshape, Xshape):
    HC = _some_array(HCshape) + 1.0
    C = _some_array(Cshape) + 2.0
    X = _some_array(Xshape) + 3.0 + np.eye(Xshape[0])

    S = HC @ HC.T + X @ X.T
    K = C @ HC.T @ linalg.inv(S)
    C1 = C @ C.T - K @ S @ K.T

    extra, (bw_noise, g) = cholesky_util.revert_conditional(
        R_X_F=HC.T, R_X=C.T, R_YX=X.T
    )

    def cov(x):
        return x.T @ x

    assert testing.allclose(cov(extra), S)
    assert testing.allclose(g, K)
    assert testing.allclose(cov(bw_noise), C1)


@testing.parametrize("Cshape, Hshape", ([(3, 3), (2, 3)],))
def test_revert_kernel_noisefree(Cshape, Hshape):
    C = _some_array(Cshape) + 1.0
    H = _some_array(Hshape) + 2.0
    HC = H @ C

    S = HC @ HC.T
    K = C @ HC.T @ linalg.inv(S)

    C1 = (np.eye(Cshape[0]) - K @ H) @ C @ C.T @ (np.eye(Cshape[0]) - K @ H).T

    extra, (bw_noise, g) = cholesky_util.revert_conditional_noisefree(
        R_X_F=HC.T, R_X=C.T
    )

    def cov(x):
        return x.T @ x

    assert testing.allclose(cov(extra), S)
    assert testing.allclose(g, K)
    assert testing.allclose(cov(bw_noise), C1)


def _some_array(shape):
    key = random.prng_key(seed=1)
    return random.normal(key, shape=shape)


def test_sqrt_sum_square_scalar():
    a = 3.0
    b = 4.0
    c = 5.0
    expected = np.sqrt(a**2 + b**2 + c**2)
    received = cholesky_util.sqrt_sum_square_scalar(a, b, c)
    assert testing.allclose(expected, received)


def test_sqrt_sum_square_error():
    a = 3.0 * np.eye(2)
    b = 4.0 * np.eye(2)
    c = 5.0 * np.eye(2)
    with testing.raises(ValueError, match="scalar"):
        _ = cholesky_util.sqrt_sum_square_scalar(a, b, c)


def test_reverse_conditional_jacrev_zero_matrix():
    """For zero-valued input covariances, reverse-mode gradients need a trick.

    This resolves issue #668.
    """
    C = _some_array((3, 3)) * 0.0
    HC = _some_array((2, 3)) * 0.0
    X = _some_array((2, 2)) + 3.0 + np.eye(2)

    result = functools.jacrev(cholesky_util.revert_conditional)(HC.T, C.T, X.T)
    is_not_nan = _tree_is_free_of_nans(result)
    assert is_not_nan


def test_sum_of_sqrtm_factors_jacrev_zero_matrix():
    """For zero-valued input covariances, reverse-mode gradients need a trick.

    This resolves issue #668.
    """
    C = _some_array((3, 3)) * 0.0
    HC = _some_array((3, 2))

    result = functools.jacrev(cholesky_util.sum_of_sqrtm_factors)((C.T, HC.T))
    is_not_nan = _tree_is_free_of_nans(result)
    assert is_not_nan


def test_sqrt_sum_square_scalar_derivative_value_test():
    """Test that the values match previous versions.

    Why? Because we implement custom derivatives for triangularisation
    to resolve specific corner cases, but need to assert that these are correct.
    """

    @functools.grad
    def triu_via_naive_arithmetic_and_autograd(x, y, z):
        return np.sqrt(x**2 + y**2 + z**2)

    @functools.grad
    def triu_via_qr_r(x, y, z):
        return cholesky_util.sqrt_sum_square_scalar(x, y, z)

    a, b, c = 3.0, 4.0, 5.0
    expected = triu_via_naive_arithmetic_and_autograd(a, b, c)
    received = triu_via_qr_r(a, b, c)
    assert testing.allclose(expected, received)


def test_sqrt_sum_square_scalar_derivative_value_test_at_origin():
    """Like the previous test, but for zero inputs.

    This ensures that the QR decomposition is differentiable at the origin,
    which is not the case unless we use custom JVPs.
    """

    # Use square of triu to ensure that the reference is differentiable
    # (np.sqrt is not differentiable at zero)
    @functools.grad
    def triu_via_naive_arithmetic_and_autograd(x, y, z):
        return x**2 + y**2 + z**2

    @functools.grad
    def triu_via_qr_r(x, y, z):
        return cholesky_util.sqrt_sum_square_scalar(x, y, z) ** 2

    a, b, c = 0.0, 0.0, 0.0
    expected = triu_via_naive_arithmetic_and_autograd(a, b, c)
    received = triu_via_qr_r(a, b, c)
    assert testing.allclose(expected, received)


def _tree_is_free_of_nans(tree):
    def contains_no_nan(x):
        return np.logical_not(np.any(np.isnan(x)))

    tree_contains_no_nan = tree_util.tree_map(contains_no_nan, tree)
    return tree_util.tree_all(tree_contains_no_nan)
