"""Various interfaces."""

import abc
from typing import Generic, TypeVar

from probdiffeq.statespace import _collections

# todo: split into multiple files.


SSVTypeVar = TypeVar("SSVTypeVar", bound=_collections.SSV)
"""A type-variable to alias appropriate state-space variable types."""


# todo: remove
CacheTypeVar = TypeVar("CacheTypeVar")
"""A type-variable to alias extrapolation- and correction-caches."""


class AbstractExtrapolation(abc.ABC, Generic[SSVTypeVar, CacheTypeVar]):
    """Extrapolation model interface."""

    def __init__(self, a, q_sqrtm_lower, preconditioner_scales, preconditioner_powers):
        self.a = a
        self.q_sqrtm_lower = q_sqrtm_lower

        self.preconditioner_scales = preconditioner_scales
        self.preconditioner_powers = preconditioner_powers

    def tree_flatten(self):
        children = (
            self.a,
            self.q_sqrtm_lower,
            self.preconditioner_scales,
            self.preconditioner_powers,
        )
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        a, q_sqrtm_lower, scales, powers = children
        return cls(
            a=a,
            q_sqrtm_lower=q_sqrtm_lower,
            preconditioner_scales=scales,
            preconditioner_powers=powers,
        )

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    @abc.abstractmethod
    def promote_output_scale(self, output_scale) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def solution_from_tcoeffs_without_reversal(
        self,
        taylor_coefficients,
        /,
    ) -> SSVTypeVar:
        raise NotImplementedError

    @abc.abstractmethod
    def solution_from_tcoeffs_with_reversal(self, taylor_coefficients, /) -> SSVTypeVar:
        raise NotImplementedError

    @abc.abstractmethod
    def init_without_reversal(self, rv, /):
        raise NotImplementedError

    @abc.abstractmethod
    def init_with_reversal(self, rv, cond, /):
        raise NotImplementedError

    @abc.abstractmethod
    def init_with_reversal_and_reset(self, rv, cond, /):
        raise NotImplementedError

    @abc.abstractmethod
    def extract_without_reversal(self, s, /):
        raise NotImplementedError

    @abc.abstractmethod
    def extract_with_reversal(self, s, /):
        raise NotImplementedError

    @abc.abstractmethod
    def begin(self, s0, /, dt) -> SSVTypeVar:
        raise NotImplementedError

    @abc.abstractmethod
    def complete_without_reversal(
        self,
        output_begin: SSVTypeVar,
        /,
        s0,
        output_scale,
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def complete_with_reversal(
        self,
        output_begin: SSVTypeVar,
        /,
        s0,
        output_scale,
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def replace_backward_model(self, s: SSVTypeVar, /, backward_model) -> SSVTypeVar:
        raise NotImplementedError

    @abc.abstractmethod
    def duplicate_with_unit_backward_model(self, s: SSVTypeVar, /) -> SSVTypeVar:
        raise NotImplementedError
