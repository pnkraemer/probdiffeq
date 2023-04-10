"""Interface for estimation strategies."""

import abc
from typing import Generic, TypeVar

import jax

P = TypeVar("P")
"""A type-variable to indicate strategy-state ("posterior") types."""


@jax.tree_util.register_pytree_node_class
class Strategy(abc.ABC, Generic[P]):
    """Inference strategy interface."""

    def __init__(self, implementation):
        self.implementation = implementation

    def __repr__(self):
        args = f"implementation={self.implementation}"
        return f"{self.__class__.__name__}({args})"

    @abc.abstractmethod
    def init(self, *, taylor_coefficients) -> P:
        raise NotImplementedError

    @abc.abstractmethod
    def extract_u(self, posterior: P, /):
        raise NotImplementedError

    @abc.abstractmethod
    def extract_marginals(self, posterior: P, /):
        raise NotImplementedError

    @abc.abstractmethod
    def extract_marginals_terminal_values(self, posterior: P, /):
        raise NotImplementedError

    @abc.abstractmethod
    def case_right_corner(self, *, p0: P, p1: P, t, t0, t1, output_scale):
        raise NotImplementedError

    @abc.abstractmethod
    def case_interpolate(self, *, p0: P, p1: P, t, t0, t1, output_scale):
        raise NotImplementedError

    @abc.abstractmethod
    def offgrid_marginals(
        self, *, t, marginals, posterior, posterior_previous: P, t0, t1, output_scale
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, key, *, posterior: P, shape):
        raise NotImplementedError

    @abc.abstractmethod
    def begin_extrapolation(self, posterior: P, /, *, dt):
        raise NotImplementedError

    @abc.abstractmethod
    def complete_extrapolation(
        self, output_extra: P, /, *, output_scale, posterior_previous: P
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def begin_correction(self, output_extra: P, /, *, vector_field, t, p):
        raise NotImplementedError

    @abc.abstractmethod
    def complete_correction(self, extrapolated: P, /, *, cache_obs):
        raise NotImplementedError

    def tree_flatten(self):
        children = (self.implementation,)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        (implementation,) = children
        return cls(implementation=implementation)

    def init_error_estimate(self):
        return self.implementation.extrapolation.init_error_estimate()

    def init_output_scale(self, *args, **kwargs):
        init_fn = self.implementation.extrapolation.init_output_scale
        return init_fn(*args, **kwargs)
