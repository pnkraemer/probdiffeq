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
import pytest
import pytest_cases

from probdiffeq.impl import impl

case = pytest_cases.case
filterwarnings = pytest.mark.filterwarnings
parametrize = pytest.mark.parametrize
parametrize_with_cases = pytest_cases.parametrize_with_cases
raises = pytest.raises
warns = pytest.warns
xfail = pytest.xfail


def skip(reason):
    return pytest.skip(reason=reason)


def fixture(name=None, scope="module"):
    # We have a different default! Usually, the scope is set to "function".
    # We benefit so much from "module" that we choose this instead.
    return pytest_cases.fixture(name=name, scope=scope)


def tree_all_allclose(tree1, tree2, **kwargs):
    trees_is_allclose = _tree_allclose(tree1, tree2, **kwargs)
    return jax.tree_util.tree_all(trees_is_allclose)


def _tree_allclose(tree1, tree2, **kwargs):
    def allclose_partial(*args):
        return jnp.allclose(*args, **kwargs)

    return jax.tree_util.tree_map(allclose_partial, tree1, tree2)


def marginals_allclose(m1, m2, /):
    mean_allclose = jnp.allclose(impl.stats.mean(m1), impl.stats.mean(m2))
    cov_allclose = jnp.allclose(impl.stats.cov_dense(m1), impl.stats.cov_dense(m2))
    return mean_allclose and cov_allclose
