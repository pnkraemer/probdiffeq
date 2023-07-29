"""Custom control flow."""

import jax
import jax.numpy as jnp


def tree_prepend(tree_of_arrays, tree_of_stacked_arrays):
    broadcast = jax.tree_util.tree_map(lambda x: x[None, ...], tree_of_arrays)
    return tree_concatenate([broadcast, tree_of_stacked_arrays])


def tree_concatenate(list_of_trees):
    """Apply  tree_transpose and jnp.stack() to a list of PyTrees."""
    tree_of_lists = _tree_transpose(list_of_trees)
    return jax.tree_util.tree_map(
        jnp.concatenate, tree_of_lists, is_leaf=lambda x: isinstance(x, list)
    )


# todo: should this be public or not?
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
