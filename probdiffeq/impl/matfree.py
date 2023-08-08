# """Matrix-free stuff."""
#
# import abc
# import dataclasses
# from dataclasses import dataclass
# from typing import Callable, TypeVar
#
import jax


#
#
# def linop_from_matmul(matrix):
#     return MatrixLinOp(matrix)
#
#
def linop_from_callable(func):
    return CallableLinOp(func)


#
#
# # Why? Because transformations/conditionals can either be
# # matrices or matrix-free linear operators.
# #
# # We have to build a generic version of those that can
# # 1) Behave like matmul (i.e. behave *exactly* like A @ x, where x can be any array)
# # 2) Are vmap'able; the vmap-output can be used as an input to scan(...xs=)
# # 3) Can be implemented matrix-free (we do a lot of slicing and a lot of JVPs)
# #
# # Always using a matrix fails because QOI-conditioning must use indexing/slicing.
# # Always using an operator fails because merging and vmapping becomes awkward.
# #
# # But we also don't want to have two implementations
# # (one for matrices and one for callables)
# # for all transformations and conditionals. So we need to find the common denominator.
# #
# #
# # Can we simplify this?
#
#
# class LinOp(abc.ABC):
#     @abc.abstractmethod
#     def __matmul__(self, other):
#         raise NotImplementedError
#
#
# class MatrixLinOp(LinOp):
#     def __init__(self, matrix, /):
#         self.matrix = matrix
#
#     def __repr__(self):
#         return f"MatrixLinOp({self.matrix})"
#
#     def __matmul__(self, other):
#         return self.matrix @ other
#


class CallableLinOp:
    def __init__(self, func, /):
        self.func = func

    def __matmul__(self, other):
        return self.func(other)


#
# T = TypeVar("T")
#
#
# def merge_linops(linop1: T, linop2: T) -> T:
#     if isinstance(linop1, MatrixLinOp):
#         return MatrixLinOp(linop1.matrix @ linop2.matrix)
#     raise ValueError


def _linop_flatten(linop):
    children = ()
    aux = (linop.func,)
    return children, aux


def _linop_unflatten(aux, _children):
    (func,) = aux
    return linop_from_callable(func)


jax.tree_util.register_pytree_node(CallableLinOp, _linop_flatten, _linop_unflatten)

# jax.tree_util.register_pytree_node(MatrixLinOp, _linop_flatten, _linop_unflatten)
#
# #
# # def merge_matmul_linops(linop1, linop2):
# #     assert not linop1.is_matfree
# #     assert not linop2.is_matfree
# #     return linop_from_matmul(linop1.params @ linop2.params)
# #
