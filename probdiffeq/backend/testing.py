"""Test-utilities.

In this file, we pick-and-mix functionality from pytest_cases and pytest.

Some of the functionality provided by both libraries overlaps,
and without bundling them here, choices between
(e.g.) pytest.fixture and pytest_cases.fixture
have been very inconsistent.
This is not good for extendability of the test suite.
"""

import jax
import jax.numpy as jnp
import jax.tree_util
import pytest
import pytest_cases

case = pytest_cases.case
filterwarnings = pytest.mark.filterwarnings
parametrize = pytest.mark.parametrize
parametrize_with_cases = pytest_cases.parametrize_with_cases
raises = pytest.raises
warns = pytest.warns
xfail = pytest.xfail


def skip(reason):
    return pytest.skip(reason=reason)


def fixture(name=None, scope="function"):
    return pytest_cases.fixture(name=name, scope=scope)


def allclose(tree1, tree2, /, *, atol: float | None = None, rtol: float | None = None):
    """Check whether two pytrees are 'close' to each other.

    In contrast to jax.numpy.allclose, this version:
    - Works with pytrees (by comparing the structure and leaves)
    - Uses different tolerances for single precision than for double precision.
      (It adjusts atol and rtol to the floating-point precision of the leaves.)
    """
    trees_is_allclose = _tree_allclose(tree1, tree2, atol=atol, rtol=rtol)
    return jax.tree_util.tree_all(trees_is_allclose)


def _tree_allclose(tree1, tree2, /, *, atol, rtol):
    def allclose_partial(t1, t2, /):
        return _allclose(t1, t2, atol=atol, rtol=rtol)

    return jax.tree_util.tree_map(allclose_partial, tree1, tree2)


def marginals_allclose(
    m1, m2, /, *, ssm, atol: float | None = None, rtol: float | None = None
):
    m1, c1 = ssm.stats.to_multivariate_normal(m1)
    m2, c2 = ssm.stats.to_multivariate_normal(m2)

    means_allclose = _allclose(m1, m2, atol=atol, rtol=rtol)
    covs_allclose = _allclose(c1, c2, atol=atol, rtol=rtol)
    return jnp.logical_and(means_allclose, covs_allclose)


def _allclose(a, b, /, *, atol: float | None, rtol: float | None):
    # promote to float-type to enable finfo.eps
    a = jnp.asarray(1.0 * a)
    b = jnp.asarray(1.0 * b)

    # numpy.allclose uses defaults atol=1e-8 and rtol=1e-5;
    # we mirror this as atol=sqrt(tol) and rtol slightly larger.
    tol = jnp.sqrt(jnp.finfo(b.dtype).eps)
    if atol is None:
        atol = tol
    if rtol is None:
        rtol = 10 * tol
    return jnp.allclose(a, b, atol=atol, rtol=rtol)
