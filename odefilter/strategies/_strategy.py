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
    def begin_extrapolation(self, *, posterior, p_inv, p):
        raise NotImplementedError

    @abc.abstractmethod
    def complete_extrapolation(
        self, ext_for_lin, cache, *, output_scale_sqrtm, p, p_inv, posterior_previous
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def final_correction(self, *, info_op, extrapolated, cache_obs, m_obs):
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

    def estimate_error(self, *, info_op, cache_obs, m_obs, p):
        return self.implementation.estimate_error(
            info_op=info_op, cache_obs=cache_obs, m_obs=m_obs, p=p
        )

    def assemble_preconditioner(self, *, dt):
        return self.implementation.assemble_preconditioner(dt=dt)

    def evidence_sqrtm(self, *, observed):
        return self.implementation.evidence_sqrtm(observed=observed)
