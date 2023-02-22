"""Custom control flow. Extends the functionality of jax.lax."""

import jax
import jax.numpy as jnp


# todo: this function only works if the output of scan has the same
#  pytree structure as the init;
#  i.e., if the carry type is the solution type.
def scan_with_init(*, f, init, xs, reverse=False):
    """Scan, but include the ``init`` parameter into the output.

    Often, we loop over grid-points but the initial state is part of
    what we consider a solution. For instance, solve_and_save_at does this.
    But backwards-marginalisation or sampling do as well.
    """
    carry, ys = jax.lax.scan(f=f, init=init, xs=xs, reverse=reverse)
    init_broadcast = jax.tree_util.tree_map(lambda x: jnp.asarray(x)[None, ...], init)

    def stack(x):
        stack_list = _tree_transpose([x[0], x[1]])
        return jax.tree_util.tree_map(
            jnp.concatenate, stack_list, is_leaf=lambda s: isinstance(s, list)
        )

    def stack_rev(x):
        stack_list = _tree_transpose([x[1], x[0]])
        return jax.tree_util.tree_map(
            jnp.concatenate, stack_list, is_leaf=lambda s: isinstance(s, list)
        )

    ys_stacked = jax.lax.cond(reverse, stack, stack_rev, (ys, init_broadcast))
    return carry, ys_stacked


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
