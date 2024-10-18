"""Test-utilities.

In this file, we pick-and-mix functionality from pytest_cases and pytest.

Some of the functionality provided by both libraries overlaps,
and without bundling them here, choices between
(e.g.) pytest.fixture and pytest_cases.fixture
have been very inconsistent.
This is not good for extendability of the test suite.
"""

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


def tree_all_allclose(tree1, tree2, **kwargs):
    trees_is_allclose = tree_allclose(tree1, tree2, **kwargs)
    return jax.tree_util.tree_all(trees_is_allclose)


def tree_allclose(tree1, tree2, **kwargs):
    def allclose_partial(*args):
        return jnp.allclose(*args, **kwargs)

    return jax.tree_util.tree_map(allclose_partial, tree1, tree2)


def marginals_allclose(m1, m2, /, *, ssm):
    m1, c1 = ssm.stats.to_multivariate_normal(m1)
    m2, c2 = ssm.stats.to_multivariate_normal(m2)

    means_allclose = jnp.allclose(m1, m2)
    covs_allclose = jnp.allclose(c1, c2)
    return jnp.logical_and(means_allclose, covs_allclose)
