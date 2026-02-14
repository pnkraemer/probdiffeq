"""PyTree utilities."""

import jax.flatten_util
import jax.tree


def register_pytree_node(node_type, /, flatten_func, unflatten_func):
    return jax.tree.register_pytree_node(node_type, flatten_func, unflatten_func)


def register_pytree_node_class(node_cls, /):
    return jax.tree.register_pytree_node_class(node_cls)


def Partial(func, *args, **kwargs):
    return jax.tree.Partial(func, *args, **kwargs)


def tree_map(func, tree, *rest):
    return jax.tree.tree_map(func, tree, *rest)


def tree_all(tree, /):
    return jax.tree.tree_all(tree)


def ravel_pytree(tree, /):
    return jax.flatten_util.ravel_pytree(tree)


def tree_flatten(tree, /):
    return jax.tree.tree_flatten(tree)


def tree_leaves(tree, /):
    return jax.tree.tree_leaves(tree)


def register_dataclass(datacls):
    return jax.tree.register_dataclass(datacls)
