"""Pytree-array utility functions (e.g. tree_concatenate)."""

import jax.numpy as jnp
import jax.tree_util


def tree_prepend(y, X, /):
    """PyTree-equivalent of y[None, ...].append(X)."""
    Y = jax.tree_util.tree_map(lambda s: s[None, ...], y)
    return tree_concatenate([Y, X])


def tree_append(X, y, /):
    """PyTree-equivalent of X.append(y[None, ...])."""
    Y = jax.tree_util.tree_map(lambda s: s[None, ...], y)
    return tree_concatenate([X, Y])


def tree_concatenate(list_of_trees):
    """Apply  tree_transpose and jnp.stack() to a list of PyTrees."""
    tree_of_lists = _tree_transpose(list_of_trees)
    return jax.tree_util.tree_map(
        jnp.concatenate, tree_of_lists, is_leaf=lambda x: isinstance(x, list)
    )


# TODO: should this be public or not?
def tree_stack(list_of_trees):
    """Apply  tree_transpose and jnp.stack() to a list of PyTrees."""
    tree_of_lists = _tree_transpose(list_of_trees)
    return jax.tree_util.tree_map(
        jnp.stack, tree_of_lists, is_leaf=lambda x: isinstance(x, list)
    )


# From https://jax.readthedocs.io/en/latest/jax-101/05.1-pytrees.html
def _tree_transpose(list_of_trees):
    """Convert a list of trees of identical structure into a single tree of lists."""
    return jax.tree_util.tree_map(lambda *xs: list(xs), *list_of_trees)
