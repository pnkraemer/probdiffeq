"""Container types."""
import dataclasses

import jax

dataclass = dataclasses.dataclass


# See https://github.com/google/jax/issues/2371
def _register_pytree_node_dataclass(clz, /):
    def flatten(obj, /):
        return jax.tree_util.tree_flatten(dataclasses.asdict(obj))

    def unflatten(aux, children):
        return clz(**aux.unflatten(children))

    jax.tree_util.register_pytree_node(clz, flatten, unflatten)
    return clz


def dataclass_pytree_node(clz, /):
    return _register_pytree_node_dataclass(dataclass(frozen=True)(clz))
