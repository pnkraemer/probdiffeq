"""Tests for specific containers."""

import jax.numpy as jnp
import jax.tree_util

from probdiffeq.backend import containers


def test_dataclass_tree():
    @containers.dataclass
    class MyClass:
        a: float
        b: float

    @containers.dataclass_pytree_node
    class MyPyTree:
        a: float
        b: float

    # Traditional dataclasses are leaves
    dataclass_instance = MyClass([2.0], (3.0, 4.0, 5.0, 6.0))
    children, aux = jax.tree_util.tree_flatten(dataclass_instance)
    assert len(children) == 1

    # containers.dataclass_tree yields a pytree, which has two leaves (not one)
    dataclass_pytree = MyPyTree([2.0], (3.0, 4.0, 5.0, 6.0))
    children, aux = jax.tree_util.tree_flatten(dataclass_pytree)
    assert len(children) > 1  # because MyClass has > 1 leaves (5 in this example)

    # tree_unflatten works as well.
    back = jax.tree_util.tree_unflatten(aux, children)
    assert jax.tree_util.tree_all(jax.tree_map(jnp.allclose, back, dataclass_pytree))
