"""Various interfaces."""

import abc
from typing import Any, Generic, NamedTuple, Tuple, TypeVar

from probdiffeq.statespace import _collections

# todo: split into multiple files.


class State(NamedTuple):
    backward_model: Any
    cache: Any

    def scale_covariance(self, s, /):
        if self.backward_model is not None:
            bw = self.backward_model.scale_covariance(s)
            return State(backward_model=bw, cache=self.cache)
        return self


S = TypeVar("S", bound=_collections.SSV)
"""A type-variable to alias appropriate state-space variable types."""

E = TypeVar("E", bound=State)


class Extrapolation(abc.ABC, Generic[S, E]):
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
    def solution_from_tcoeffs_without_reversal(self, taylor_coefficients, /) -> S:
        raise NotImplementedError

    @abc.abstractmethod
    def solution_from_tcoeffs_with_reversal(self, taylor_coefficients, /) -> S:
        raise NotImplementedError

    @abc.abstractmethod
    def init_without_reversal(self, rv, /) -> Tuple[S, E]:
        raise NotImplementedError

    @abc.abstractmethod
    def init_with_reversal(self, rv, cond, /) -> Tuple[S, E]:
        raise NotImplementedError

    @abc.abstractmethod
    def init_with_reversal_and_reset(self, rv, cond, /) -> Tuple[S, E]:
        raise NotImplementedError

    @abc.abstractmethod
    def extract_without_reversal(self, s: S, e: E, /):
        raise NotImplementedError

    @abc.abstractmethod
    def extract_with_reversal(self, s: S, e: E, /):
        raise NotImplementedError

    @abc.abstractmethod
    def begin(self, s: S, e: E, /, dt) -> Tuple[S, E]:
        raise NotImplementedError

    @abc.abstractmethod
    def complete_without_reversal(self, s: S, e: E, /, output_scale) -> Tuple[S, E]:
        raise NotImplementedError

    @abc.abstractmethod
    def complete_with_reversal(self, s: S, e: E, /, output_scale) -> Tuple[S, E]:
        raise NotImplementedError

    @abc.abstractmethod
    def replace_backward_model(self, e: E, /, backward_model) -> E:
        raise NotImplementedError

    @abc.abstractmethod
    def duplicate_with_unit_backward_model(self, e: E, /) -> E:
        raise NotImplementedError
