"""Various interfaces."""

import abc
from typing import Generic, Tuple, TypeVar

import jax

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

    # todo: hidden_state & hidden_shape are important for extrapolation and correction.
    #  The others uniquely correspond to either one, so why don't we access
    #  ssv.hidden_shape, ssv.hidden_shape
    #  ssv.extrapolation.backward_model, ssv.extrapolation.cache
    #  ssv.correction.error_estimate, ssv.correction.output_scale_dynamic,
    #  ssv.correction.cache, ssv.correction.observed, ...
    #  which would simplify SSV to
    #  SSV(hidden_state, /, *, hidden_shape, extrapolation: T, correction: S)
    #  and we think less about which quantity is None at which step.
    #  In other words: SSV() is getting too powerful.
    #
    # todo: change SSV to only contain hidden_shape and hidden_state.
    #  then, extrapolation and correction models have their own types
    #  and we aim for:
    #     # x, ex, co = strategy.init(**sol)
    #     x, ex = extra.init(*sol)
    #     x, co = corr.init(x)
    #     for _ in range(10):
    #         # x, ex, co = strategy.begin(x, **pro_ex, **pro_co)
    #         x, ex = extra.begin(x, ex, **pro_ex)
    #         x, co = corr.begin(x, co, **pro_co)
    #         # x, ex, co = strategy.complete(x, **pro_ex, **pro_co)
    #         x, ex = extra.complete(x, ex, **pro_ex)
    #         x, co = corr.complete(x, co, **pro_sol)
    #         # sol = strategy.extract(x, ex, co)
    #         yield extra.extract(corr.extract(x, co), ex)
    #     x: hidden_state, hidden_shape(optional)
    #     ex: backward_model(optional), cache_extra(tmp)
    #     co: error_estimate, observed, output_scale_dynamic(tmp?), cache_corr(tmp)
    #     # interpolation
    #     x, ex = extra.init(*sol)
    #     for _ in range(10):
    #         x, ex = extra.begin(x, ex, **pro_ex)
    #         x, ex = extra.complete(x, ex, **pro_ex)
    #         yield extra.extract(x, ex)

    def __init__(
        self,
        hidden_state,
        *,
        hidden_shape,  # not always = hidden_state.shape! E.g. vectorised SSMs.
        observed_state,
        error_estimate,
        backward_model,
        # temporary:
        output_scale_dynamic,
        cache_extra,
        cache_corr,
    ):
        self.hidden_shape = hidden_shape
        self.hidden_state = hidden_state  # todo: 'hidden'
        self.backward_model = backward_model
        self.observed_state = observed_state  # todo: 'observed'

        # todo: add conditional here
        #  (and make init_with_reversal, extract_with_reversal methods in extrapolation)
        self.output_scale_dynamic = output_scale_dynamic
        self.error_estimate = error_estimate
        self.cache_extra = cache_extra
        self.cache_corr = cache_corr

    def tree_flatten(self):
        children = (
            self.hidden_state,
            self.observed_state,
            self.output_scale_dynamic,
            self.error_estimate,
            self.cache_extra,
            self.cache_corr,
            self.backward_model,
        )
        aux = (self.hidden_shape,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (
            hidden_state,
            observed_state,
            output_scale_dynamic,
            error_estimate,
            cache_e,
            cache_c,
            backward_model,
        ) = children
        (hidden_shape,) = aux
        return cls(
            hidden_state=hidden_state,
            hidden_shape=hidden_shape,
            observed_state=observed_state,
            output_scale_dynamic=output_scale_dynamic,
            error_estimate=error_estimate,
            cache_extra=cache_e,
            cache_corr=cache_c,
            backward_model=backward_model,
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"hidden_shape={self.hidden_shape},"
            f"hidden_state={self.hidden_state},"
            f"observed_state={self.observed_state},"
            f"output_scale_dynamic={self.output_scale_dynamic},"
            f"error_estimate={self.error_estimate},"
            f"cache_extra={self.cache_extra},"
            f"cache_corr={self.cache_corr},"
            f"backward_model={self.backward_model}"
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


SSVTypeVar = TypeVar("SSVTypeVar", bound=SSV)
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


class AbstractConditional(abc.ABC, Generic[SSVTypeVar]):
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


@jax.tree_util.register_pytree_node_class
class AbstractCorrection(abc.ABC, Generic[SSVTypeVar, CacheTypeVar]):
    """Correction model interface."""

    def __init__(self, ode_order):
        self.ode_order = ode_order

    def tree_flatten(self):
        children = ()
        aux = (self.ode_order,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, _children):
        (ode_order,) = aux
        return cls(ode_order=ode_order)

    @abc.abstractmethod
    def init(self, s, /):
        raise NotImplementedError

    @abc.abstractmethod
    def begin(self, x: SSVTypeVar, /, vector_field, t, p) -> SSVTypeVar:
        raise NotImplementedError

    @abc.abstractmethod
    def complete(self, x: SSVTypeVar, /, vector_field, t, p) -> SSVTypeVar:
        raise NotImplementedError
