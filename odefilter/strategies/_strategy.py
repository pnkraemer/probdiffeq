"""Inference interface."""

import abc

import jax
import jax.tree_util


@jax.tree_util.register_pytree_node_class
class Strategy(abc.ABC):
    """Inference strategy interface."""

    def __init__(self, *, implementation):
        self.implementation = implementation

    @abc.abstractmethod
    def init_posterior(self, *, taylor_coefficients):
        raise NotImplementedError

    @abc.abstractmethod
    def extract_sol_terminal_value(self, *, posterior):
        raise NotImplementedError

    @abc.abstractmethod
    def extract_sol_from_marginals(self, *, marginals):
        raise NotImplementedError

    @abc.abstractmethod
    def case_right_corner(self, *, p0, p1, t, t0, t1, scale_sqrtm):
        raise NotImplementedError

    @abc.abstractmethod
    def case_interpolate(self, *, p0, rv1, t, t0, t1, scale_sqrtm):  # noqa: D102
        raise NotImplementedError

    @abc.abstractmethod
    def marginals(self, *, posterior):  # todo: rename to marginalise?
        raise NotImplementedError

    @abc.abstractmethod
    def marginals_terminal_value(self, *, posterior):  # todo: rename to marginalise?
        raise NotImplementedError

    @abc.abstractmethod
    def offgrid_marginals(
        self, *, t, marginals, posterior_previous, t0, t1, scale_sqrtm
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, key, *, posterior, shape):
        raise NotImplementedError

    @abc.abstractmethod
    def begin_extrapolation(self, *, posterior, dt):
        raise NotImplementedError

    @abc.abstractmethod
    def complete_extrapolation(
        self, linearisation_pt, cache, *, output_scale_sqrtm, posterior_previous
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def complete_correction(self, *, info_op, extrapolated, cache_obs, obs_pt):
        raise NotImplementedError

    @abc.abstractmethod
    def scale_marginals(self, marginals, *, output_scale_sqrtm):
        raise NotImplementedError

    @abc.abstractmethod
    def scale_posterior(self, posterior, *, output_scale_sqrtm):
        raise NotImplementedError

    def tree_flatten(self):
        children = (self.implementation,)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        (implementation,) = children
        return cls(implementation=implementation)

    def _base_samples(self, key, *, shape):
        base_samples = jax.random.normal(key=key, shape=shape)
        return base_samples

    def init_error_estimate(self):
        return self.implementation.init_error_estimate()

    def init_output_scale_sqrtm(self):
        return self.implementation.init_output_scale_sqrtm()

    # this stuff should move to the info op...

    def begin_correction(self, linearisation_pt, *, info_op, t, p):
        obs_pt, cache_obs = info_op.linearize(linearisation_pt, t=t, p=p)
        scale_sqrtm, error = info_op.estimate_error(cache_obs=cache_obs, obs_pt=obs_pt)
        return error * scale_sqrtm, scale_sqrtm, (obs_pt, *cache_obs)
