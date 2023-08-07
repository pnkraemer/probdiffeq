"""Matrix-free stuff."""

import dataclasses
from dataclasses import dataclass
from typing import Callable

import jax


def linop_from_matmul(matrix):
    return LinOp(lambda s, m: m @ s, params=matrix)


def linop_from_callable(func):
    return LinOp(lambda s, m: func(s), params=())


# Why? Because transformations/conditionals can either be
# matrices or matrix-free linear operators.
# We have to build a generic version of those that can
# 1) Behave like matmul (i.e. behave *exactly* like A @ x, where x can be any array)
# 2) Are vmap'able
# 3) Can be used as inputs to scan(...xs=?)
# 4) Can be implemented matrix-free (we do a lot of slicing and a lot of JVPs)


@dataclasses.dataclass(frozen=True)
class LinOp:
    func: Callable
    params: jax.Array

    # Do we provide both call and matmul?

    def __matmul__(self, other):
        return self.func(other, self.params)

    def __iter__(self):
        for p in self.params:
            yield LinOp(self.func, p)


def _flatten(linop):
    children = linop.params
    aux = linop.matmul
    return children, aux


def _unflatten(aux, children):
    matmul = aux
    params = children
    return LinOp(matmul=matmul, params=params)


jax.tree_util.register_pytree_node(LinOp, _flatten, _unflatten)
