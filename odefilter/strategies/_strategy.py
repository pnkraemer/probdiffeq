"""Inference interface."""

import abc

import jax
import jax.numpy as jnp
import jax.tree_util

from odefilter.implementations import isotropic


@jax.tree_util.register_pytree_node_class
class Strategy(abc.ABC):
    """Inference strategy interface."""

    def __init__(self, *, extrapolation=None, correction=None):
        if extrapolation is None:
            self.extrapolation = isotropic.IsotropicIBM.from_params()
        else:
            self.extrapolation = extrapolation

        if correction is None:
            self.correction = isotropic.TaylorConstant()
        else:
            self.correction = correction

    def __repr__(self):
        args = f"extrapolation={self.extrapolation}, correction={self.correction}"
        return f"{self.__class__.__name__}({args})"

    def __eq__(self, other):
        equal = jax.tree_util.tree_map(lambda a, b: jnp.all(a == b), self, other)
        return jax.tree_util.tree_all(equal)

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
    def complete_correction(self, *, extrapolated, cache_obs):
        raise NotImplementedError

    @abc.abstractmethod
    def scale_marginals(self, marginals, *, output_scale_sqrtm):
        raise NotImplementedError

    @abc.abstractmethod
    def scale_posterior(self, posterior, *, output_scale_sqrtm):
        raise NotImplementedError

    def tree_flatten(self):
        children = (self.extrapolation, self.correction)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        (extrapolation, correction) = children
        return cls(extrapolation=extrapolation, correction=correction)

    def _base_samples(self, key, *, shape):
        base_samples = jax.random.normal(key=key, shape=shape)
        return base_samples

    def init_error_estimate(self):
        return self.extrapolation.init_error_estimate()

    def init_output_scale_sqrtm(self):
        return self.extrapolation.init_output_scale_sqrtm()

    def begin_correction(self, linearisation_pt, *, vector_field, t, p):
        return self.correction.begin_correction(
            linearisation_pt, vector_field=vector_field, t=t, p=p
        )
