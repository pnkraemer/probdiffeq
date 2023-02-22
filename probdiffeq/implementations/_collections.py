"""Various interfaces."""

import abc
from typing import Generic, Tuple, TypeVar

import jax


class AbstractNormal(abc.ABC):
    """Normal-distributed random variables.

    Means, covariances, log-probability-density functions, sampling, and so on.
    """

    def __init__(self, mean, cov_sqrtm_lower):
        self.mean = mean
        self.cov_sqrtm_lower = cov_sqrtm_lower

    def __repr__(self):
        name = f"{self.__class__.__name__}"
        args = f"mean={self.mean}, cov_sqrtm_lower={self.cov_sqrtm_lower}"
        return f"{name}({args})"

    def tree_flatten(self) -> Tuple:
        children = self.mean, self.cov_sqrtm_lower
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        mean, cov_sqrtm_lower = children
        return cls(mean=mean, cov_sqrtm_lower=cov_sqrtm_lower)

    @abc.abstractmethod
    def logpdf(self, u, /):
        raise NotImplementedError

    @abc.abstractmethod
    def transform_unit_sample(self, x, /):
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

    @property
    def sample_shape(self) -> Tuple[int]:
        return self.mean.shape


class StateSpaceVar(abc.ABC):
    """State-space variables.

    Hidden states, and knowledge about extracting a quantity of interest.

    For example, the state-space variable of an integrated Wiener process is (x, x'),
    whereas the quantity of interest is (x, x') -> x.
    Or, the sum of the output of two integrated Wiener processes tracks (x, x', y, y'),
    and the quantity of interest is (x, x', y, y') -> x+y
    """

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

    @abc.abstractmethod
    def marginal_nth_derivative(self, n):
        raise NotImplementedError


SSVTypeVar = TypeVar("SSVTypeVar", bound=StateSpaceVar)
"""A type-variable to alias appropriate state-space variable types."""

CacheTypeVar = TypeVar("CacheTypeVar")
"""A type-variable to alias extrapolation- and correction-caches."""


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
    def init_conditional(self, ssv_proto):
        raise NotImplementedError

    @abc.abstractmethod
    def begin_extrapolation(self, p0, /, dt) -> Tuple[SSVTypeVar, CacheTypeVar]:
        raise NotImplementedError

    @abc.abstractmethod
    def complete_extrapolation(
        self,
        linearisation_pt: SSVTypeVar,
        p0,
        cache: CacheTypeVar,
        output_scale_sqrtm,
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def revert_markov_kernel(
        self,
        linearisation_pt: SSVTypeVar,
        p0,
        cache: CacheTypeVar,
        output_scale_sqrtm,
    ):
        raise NotImplementedError


class AbstractConditional(abc.ABC, Generic[SSVTypeVar]):
    """Conditional distribution interface.

    Used as a backward model for backward-Gauss--Markov process representations.
    """

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
