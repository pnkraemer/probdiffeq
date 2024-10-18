"""PyTree utilities."""

import jax.flatten_util
import jax.tree_util


def register_pytree_node(node_type, /, flatten_func, unflatten_func):
    return jax.tree_util.register_pytree_node(node_type, flatten_func, unflatten_func)


def register_pytree_node_class(node_cls, /):
    return jax.tree_util.register_pytree_node_class(node_cls)


def Partial(func, *args, **kwargs):
    return jax.tree_util.Partial(func, *args, **kwargs)


def tree_map(func, tree, *rest):
    return jax.tree_util.tree_map(func, tree, *rest)


def tree_all(tree, /):
    return jax.tree_util.tree_all(tree)


def ravel_pytree(tree, /):
    return jax.flatten_util.ravel_pytree(tree)
