"""Collections of interfaces."""

import abc
from typing import Any, Generic, Tuple, TypeVar

import jax

from odefilter import _tree_utils

# todo: make "u" a property?


class StateSpaceVariable(abc.ABC):
    @abc.abstractmethod
    def logpdf(self, u, /):
        raise NotImplementedError

    @abc.abstractmethod
    def norm_of_whitened_residual_sqrtm(self):
        raise NotImplementedError

    @abc.abstractmethod
    def condition_on_qoi_observation(self, u, /, *, observation_std):
        raise NotImplementedError

    @abc.abstractmethod
    def extract_qoi(self):
        raise NotImplementedError

    @abc.abstractmethod
    def extract_qoi_from_sample(self, u, /):
        raise NotImplementedError

    @abc.abstractmethod
    def scale_covariance(self, *, scale_sqrtm):
        raise NotImplementedError

    @abc.abstractmethod
    def transform_unit_sample(self, x, /):
        raise NotImplementedError

    @abc.abstractmethod
    def Ax_plus_y(self, *, A, x, y):
        raise NotImplementedError


SSVTypeVar = TypeVar("SSVTypeVar", bound=StateSpaceVariable)
"""A type-variable to alias appropriate state-space variable types."""

CacheTypeVar = TypeVar("CacheTypeVar")


@jax.tree_util.register_pytree_node_class
class AbstractExtrapolation(
    abc.ABC, Generic[SSVTypeVar, CacheTypeVar], _tree_utils.TreeEqualMixIn
):
    """Extrapolation model interface."""

    def tree_flatten(self):
        return (), ()

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls()

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    @abc.abstractmethod
    def init_corrected(self, *, taylor_coefficients) -> SSVTypeVar:
        raise NotImplementedError

    @abc.abstractmethod
    def init_output_scale_sqrtm(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def init_error_estimate(self) -> jax.Array:
        raise NotImplementedError

    @abc.abstractmethod
    def init_conditional(self, *, rv_proto):
        raise NotImplementedError

    @abc.abstractmethod
    def begin_extrapolation(self, m0, /, *, dt) -> Tuple[SSVTypeVar, CacheTypeVar]:
        raise NotImplementedError

    @abc.abstractmethod
    def complete_extrapolation(
        self,
        *,
        linearisation_pt: SSVTypeVar,
        l0,
        cache: CacheTypeVar,
        output_scale_sqrtm,
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def revert_markov_kernel(
        self,
        *,
        linearisation_pt: SSVTypeVar,
        l0,
        cache: CacheTypeVar,
        output_scale_sqrtm,
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def condense_backward_models(
        self,
        *,
        transition_init,
        noise_init: SSVTypeVar,
        transition_state,
        noise_state: SSVTypeVar,
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def marginalise_backwards(self, *, init: SSVTypeVar, linop, noise: SSVTypeVar):
        raise NotImplementedError

    @abc.abstractmethod
    def marginalise_model(self, *, init: SSVTypeVar, linop, noise: SSVTypeVar):
        raise NotImplementedError


@jax.tree_util.register_pytree_node_class
class BackwardModel(Generic[SSVTypeVar]):
    """Backward model for backward-Gauss--Markov process representations."""

    def __init__(self, transition: Any, *, noise: SSVTypeVar):
        self.transition = transition
        self.noise = noise

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(transition={self.transition}, noise={self.noise})"

    def tree_flatten(self):
        children = self.transition, self.noise
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        transition, noise = children
        return cls(transition=transition, noise=noise)

    def scale_covariance(self, *, scale_sqrtm):
        noise = self.noise.scale_covariance(scale_sqrtm=scale_sqrtm)
        return BackwardModel(transition=self.transition, noise=noise)


@jax.tree_util.register_pytree_node_class
class AbstractCorrection(
    abc.ABC, Generic[SSVTypeVar, CacheTypeVar], _tree_utils.TreeEqualMixIn
):
    """Correction model interface."""

    def __init__(self, *, ode_order):
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
        self, x: SSVTypeVar, /, *, vector_field, t, p
    ) -> Tuple[jax.Array, float, CacheTypeVar]:
        raise NotImplementedError

    @abc.abstractmethod
    def complete_correction(self, *, extrapolated: SSVTypeVar, cache: CacheTypeVar):
        raise NotImplementedError
