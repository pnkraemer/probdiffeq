"""Custom control flow."""

import jax.lax
import jax.numpy as jnp
import jax.tree_util


def scan_with_init(*, f, init, xs, reverse=False, **kwargs):
    """Scan, but include the ``init`` parameter into the output.

    Needed to compute checkpoint-ODE-solutions.
    """
    carry, ys = jax.lax.scan(f=f, init=init, xs=xs, reverse=reverse, **kwargs)

    init_broadcast = jax.tree_util.tree_map(lambda x: jnp.asarray(x)[None, ...], init)
    if reverse:
        stack_list = _tree_transpose([ys, init_broadcast])
    else:
        stack_list = _tree_transpose([init_broadcast, ys])
    ys_stacked = jax.tree_util.tree_map(
        jnp.concatenate, stack_list, is_leaf=lambda x: isinstance(x, list)
    )
    return carry, ys_stacked


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
