"""Collections of interfaces."""

import abc
from typing import Generic, Tuple, TypeVar

import jax


class RandomVariable(abc.ABC):
    @abc.abstractmethod
    def logpdf(self, u, /):
        raise NotImplementedError

    @abc.abstractmethod
    def transform_unit_sample(self, x, /):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def sample_shape(self):
        raise NotImplementedError

    @abc.abstractmethod
    def scale_covariance(self, scale_sqrtm):
        raise NotImplementedError

    @abc.abstractmethod
    def norm_of_whitened_residual_sqrtm(self):
        raise NotImplementedError

    @abc.abstractmethod
    def Ax_plus_y(self, A, x, y):
        raise NotImplementedError


# todo: make "u" a property?
class StateSpaceVariable(abc.ABC):
    def __init__(self, hidden_state):
        self.hidden_state = hidden_state

    def tree_flatten(self):
        children = (self.hidden_state,)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        (hidden_state,) = children
        return cls(hidden_state=hidden_state)

    def __repr__(self):
        return f"{self.__class__.__name__}(hidden_state={self.hidden_state})"

    @abc.abstractmethod
    def condition_on_qoi_observation(self, u, /, observation_std):
        raise NotImplementedError

    @abc.abstractmethod
    def extract_qoi(self):
        raise NotImplementedError

    @abc.abstractmethod
    def extract_qoi_from_sample(self, u, /):
        raise NotImplementedError

    @abc.abstractmethod
    def scale_covariance(self, scale_sqrtm):
        raise NotImplementedError


SSVTypeVar = TypeVar("SSVTypeVar", bound=StateSpaceVariable)
"""A type-variable to alias appropriate state-space variable types."""

CacheTypeVar = TypeVar("CacheTypeVar")


class AbstractExtrapolation(abc.ABC, Generic[SSVTypeVar, CacheTypeVar]):
    """Extrapolation model interface."""

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    @abc.abstractmethod
    def init_corrected(self, taylor_coefficients) -> SSVTypeVar:
        raise NotImplementedError

    @abc.abstractmethod
    def init_output_scale_sqrtm(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def init_error_estimate(self) -> jax.Array:
        raise NotImplementedError

    @abc.abstractmethod
    def init_conditional(self, rv_proto):
        raise NotImplementedError

    @abc.abstractmethod
    def begin_extrapolation(self, m0, /, dt) -> Tuple[SSVTypeVar, CacheTypeVar]:
        raise NotImplementedError

    @abc.abstractmethod
    def complete_extrapolation(
        self,
        linearisation_pt: SSVTypeVar,
        l0,
        cache: CacheTypeVar,
        output_scale_sqrtm,
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def revert_markov_kernel(
        self,
        linearisation_pt: SSVTypeVar,
        l0,
        cache: CacheTypeVar,
        output_scale_sqrtm,
    ):
        raise NotImplementedError


class AbstractConditional(abc.ABC, Generic[SSVTypeVar]):
    """Backward model for backward-Gauss--Markov process representations."""

    @abc.abstractmethod
    def __call__(self, x, /):
        raise NotImplementedError

    @abc.abstractmethod
    def scale_covariance(self, scale_sqrtm):
        raise NotImplementedError

    @abc.abstractmethod
    def merge_with_incoming_conditional(self, incoming, /):
        raise NotImplementedError

    @abc.abstractmethod
    def marginalise(self, rv, /):
        raise NotImplementedError


@jax.tree_util.register_pytree_node_class
class AbstractCorrection(abc.ABC, Generic[SSVTypeVar, CacheTypeVar]):
    """Correction model interface."""

    def __init__(self, ode_order):
        self.ode_order = ode_order

    def __repr__(self):
        return f"{self.__class__.__name__}(ode_order={self.ode_order})"

    def tree_flatten(self):
        children = ()
        aux = (self.ode_order,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, _children):
        (ode_order,) = aux
        return cls(ode_order=ode_order)

    @abc.abstractmethod
    def begin_correction(
        self, x: SSVTypeVar, /, vector_field, t, p
    ) -> Tuple[jax.Array, float, CacheTypeVar]:
        raise NotImplementedError

    @abc.abstractmethod
    def complete_correction(self, extrapolated: SSVTypeVar, cache: CacheTypeVar):
        raise NotImplementedError
