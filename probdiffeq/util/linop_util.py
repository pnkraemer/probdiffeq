"""Matrix-free API."""
import dataclasses
from typing import Any, Callable

import jax


def parametrised_linop(func, /, params=None):
    return CallableLinOp(func=func, params=params)


@dataclasses.dataclass(frozen=True)
class CallableLinOp:
    """Matrix-free linear operator."""

    func: Callable
    params: Any

    def __matmul__(self, other):
        return self.func(other, self.params)


def _linop_flatten(linop):
    children = (linop.params,)
    aux = (linop.func,)
    return children, aux


def _linop_unflatten(aux, children):
    (func,) = aux
    (params,) = children
    return parametrised_linop(func, params=params)


jax.tree_util.register_pytree_node(CallableLinOp, _linop_flatten, _linop_unflatten)
