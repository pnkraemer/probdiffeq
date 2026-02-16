"""PyTree utilities."""

import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.tree_util


def register_pytree_node(node_type, /, flatten_func, unflatten_func):
    return jax.tree_util.register_pytree_node(node_type, flatten_func, unflatten_func)


def register_pytree_node_class(node_cls, /):
    return jax.tree_util.register_pytree_node_class(node_cls)


def Partial(func, *args, **kwargs):
    return jax.tree_util.Partial(func, *args, **kwargs)


def tree_map(func, tree, *rest):
    return jax.tree.map(func, tree, *rest)


def tree_all(tree, /):
    return jax.tree_util.tree_all(tree)


def ravel_pytree(tree, /):
    return jax.flatten_util.ravel_pytree(tree)


def tree_flatten(tree, /):
    return jax.tree_util.tree_flatten(tree)


def tree_leaves(tree, /):
    return jax.tree_util.tree_leaves(tree)


def register_dataclass(datacls):
    return jax.tree_util.register_dataclass(datacls)


def tree_array_prepend(y, X, /):
    """PyTree-equivalent of y[None, ...].append(X)."""
    Y = jax.tree.map(lambda s: s[None, ...], y)
    return tree_array_concatenate([Y, X])


def tree_array_append(X, y, /):
    """PyTree-equivalent of X.append(y[None, ...])."""
    Y = jax.tree.map(lambda s: s[None, ...], y)
    return tree_array_concatenate([X, Y])


def tree_array_concatenate(list_of_trees):
    """Apply  tree_array_transpose and jnp.stack() to a list of PyTrees."""
    tree_array_of_lists = _tree_array_transpose(list_of_trees)

    def is_leaf(x):
        return isinstance(x, list) and isinstance(x[0], jax.Array)

    return jax.tree.map(jnp.concatenate, tree_array_of_lists, is_leaf=is_leaf)


# TODO: should this be public or not?
def tree_array_stack(list_of_trees):
    """Apply  tree_array_transpose and jnp.stack() to a list of PyTrees."""
    tree_array_of_lists = _tree_array_transpose(list_of_trees)

    def is_leaf(x):
        return isinstance(x, list) and isinstance(x[0], jax.Array)

    return jax.tree.map(jnp.stack, tree_array_of_lists, is_leaf=is_leaf)


# From https://jax.readthedocs.io/en/latest/jax-101/05.1-pytrees.html
def _tree_array_transpose(list_of_trees):
    """Convert a list of trees of identical structure into a single tree of lists."""
    return jax.tree.map(lambda *xs: list(xs), *list_of_trees)
