"""Various interfaces."""

import abc
from typing import Generic, Tuple, TypeVar

# todo: split into multiple files.


# todo: necessary? All "normal" information should be
#  encapsulated in the implementations.
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
    def scale_covariance(self, output_scale):
        raise NotImplementedError

    @abc.abstractmethod
    def mahalanobis_norm(self, u, /):
        raise NotImplementedError

    @property
    def sample_shape(self) -> Tuple[int]:
        return self.mean.shape


class SSV(abc.ABC):
    """State-space variables.

    Hidden states, and knowledge about extracting a quantity of interest.

    For example, the state-space variable of an integrated Wiener process is (x, x'),
    whereas the quantity of interest is (x, x') -> x.
    Or, the sum of the output of two integrated Wiener processes tracks (x, x', y, y'),
    and the quantity of interest is (x, x', y, y') -> x+y
    """

    def __init__(self, hidden_state, /, *, hidden_shape=None):
        self.hidden_shape = hidden_shape
        self.hidden_state = hidden_state  # todo: 'hidden'

    def tree_flatten(self):
        children = (self.hidden_state,)
        aux = (self.hidden_shape,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (hidden_state,) = children
        (hidden_shape,) = aux
        return cls(
            hidden_state,
            hidden_shape=hidden_shape,
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"{self.hidden_shape},"
            f"hidden_state={self.hidden_state}"
            f")"
        )

    @abc.abstractmethod
    def observe_qoi(self, observation_std):
        raise NotImplementedError

    @abc.abstractmethod
    def extract_qoi(self):
        raise NotImplementedError

    @abc.abstractmethod
    def extract_qoi_from_sample(self, u, /):
        raise NotImplementedError

    @abc.abstractmethod
    def scale_covariance(self, output_scale):
        raise NotImplementedError

    @abc.abstractmethod
    def marginal_nth_derivative(self, n):
        raise NotImplementedError


S = TypeVar("S", bound=SSV)
"""A type-variable to alias appropriate state-space variable types."""


class AbstractConditional(abc.ABC, Generic[S]):
    """Conditional distribution interface.

    Used as a backward model for backward-Gauss--Markov process representations.
    """

    def __init__(self, transition, noise):
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

    @abc.abstractmethod
    def __call__(self, x, /):
        raise NotImplementedError

    def scale_covariance(self, output_scale):
        raise NotImplementedError

    def merge_with_incoming_conditional(self, incoming, /):
        raise NotImplementedError

    def marginalise(self, rv, /):
        raise NotImplementedError
