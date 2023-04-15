"""Corrections."""
from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from probdiffeq.statespace import _corr, cubature
from probdiffeq.statespace.scalar import _vars as scalar_vars
from probdiffeq.statespace.scalar import corr as scalar_corr

_SLR1CacheType = Tuple[Callable]
"""Type-variable for the correction-cache."""


def statistical_order_one(ode_shape, ode_order):
    cubature_fn = cubature.blockdiag(cubature.third_order_spherical)
    cubature_rule = cubature_fn(input_shape=ode_shape)
    return _BlockDiagStatisticalFirstOrder(
        ode_shape=ode_shape, ode_order=ode_order, cubature_rule=cubature_rule
    )


@jax.tree_util.register_pytree_node_class
class _BlockDiagStatisticalFirstOrder(_corr.AbstractCorrection):
    """First-order statistical linear regression in state-space models \
     with block-diagonal covariance structure.

    !!! warning "Warning: highly EXPERIMENTAL feature!"
        This feature is highly experimental.
        There is no guarantee that it works correctly.
        It might be deleted tomorrow
        and without any deprecation policy.

    """

    def __init__(self, ode_shape, ode_order, cubature_rule):
        if ode_order > 1:
            raise ValueError

        super().__init__(ode_order=ode_order)
        self.ode_shape = ode_shape

        self._mm = scalar_corr.StatisticalFirstOrder(
            ode_order=ode_order, cubature_rule=cubature_rule
        )

    @property
    def cubature_rule(self):
        return self._mm.cubature_rule

    def tree_flatten(self):
        # todo: should this call super().tree_flatten()?
        children = (self.cubature_rule,)
        aux = self.ode_order, self.ode_shape
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (cubature_rule,) = children
        ode_order, ode_shape = aux
        return cls(
            ode_order=ode_order, ode_shape=ode_shape, cubature_rule=cubature_rule
        )

    def init(self, x, /):
        mean_like = jnp.zeros(self.ode_shape)
        cholesky_like = jnp.zeros(self.ode_shape)
        observed_like = scalar_vars.NormalQOI(
            mean=mean_like, cov_sqrtm_lower=cholesky_like
        )
        error_estimate = jnp.zeros(self.ode_shape)
        return scalar_vars.SSV(
            observed_state=observed_like,
            error_estimate=error_estimate,
            hidden_state=x.hidden_state,
            hidden_shape=x.hidden_shape,
            backward_model=x.backward_model,
            output_scale_dynamic=None,
            cache_extra=None,
            cache_corr=None,
        )

    def begin(self, extrapolated, /, vector_field, t, p):
        # Vmap relevant functions
        vmap_f = jax.vmap(jax.tree_util.Partial(vector_field, t=t, p=p))

        # Evaluate vector field at sigma-points
        sigma_points_fn = jax.vmap(
            scalar_corr.StatisticalFirstOrder.transform_sigma_points
        )
        sigma_points, _, _ = sigma_points_fn(self._mm, extrapolated.hidden_state)

        fx = vmap_f(sigma_points.T).T  # (d, S).T = (S, d) -> (S, d) -> transpose again
        center_fn = jax.vmap(scalar_corr.StatisticalFirstOrder.center)
        fx_mean, _, fx_centered_normed = center_fn(self._mm, fx)

        # Compute output scale and error estimate
        calibrate_fn = jax.vmap(scalar_corr.StatisticalFirstOrder.calibrate)
        error_estimate, output_scale = calibrate_fn(
            self._mm, fx_mean, fx_centered_normed, extrapolated.hidden_state
        )
        return scalar_vars.SSV(
            error_estimate=output_scale * error_estimate,
            output_scale_dynamic=output_scale,
            hidden_state=extrapolated.hidden_state,
            hidden_shape=extrapolated.hidden_shape,
            observed_state=extrapolated.observed_state,
            cache_extra=extrapolated.cache_extra,
            backward_model=extrapolated.backward_model,
            cache_corr=None,
        )

    def complete(self, extrapolated, vector_field, t, p):
        vmap_f = jax.vmap(jax.tree_util.Partial(vector_field, t=t, p=p))

        H, noise = self.linearize(extrapolated, vmap_f)

        compl_fn = scalar_corr.StatisticalFirstOrder.complete_post_linearize
        fn = jax.vmap(compl_fn)
        return fn(self._mm, H, extrapolated, noise)

    def linearize(self, extrapolated, vmap_f):
        # Transform the sigma-points
        sigma_points_fn = jax.vmap(
            scalar_corr.StatisticalFirstOrder.transform_sigma_points
        )
        sigma_points, _, sigma_points_centered_normed = sigma_points_fn(
            self._mm, extrapolated.hidden_state
        )

        # Evaluate the vector field at the sigma-points
        fx = vmap_f(sigma_points.T).T  # (d, S).T = (S, d) -> (S, d) -> transpose again
        center_fn = jax.vmap(scalar_corr.StatisticalFirstOrder.center)
        fx_mean, _, fx_centered_normed = center_fn(self._mm, fx)

        # Complete the linearization
        lin_fn = jax.vmap(scalar_corr.StatisticalFirstOrder.linearization_matrices)
        return lin_fn(
            self._mm,
            fx_centered_normed,
            fx_mean,
            sigma_points_centered_normed,
            extrapolated.hidden_state,
        )


def taylor_order_zero(*args, **kwargs):
    ts0 = scalar_corr.taylor_order_zero(*args, **kwargs)
    return _BlockDiag(ts0)


_TS0CacheType = Tuple[jax.Array]


@jax.tree_util.register_pytree_node_class
class _BlockDiag(_corr.AbstractCorrection):
    def __init__(self, corr, /):
        super().__init__(ode_order=corr.ode_order)
        self.corr = corr

    def __repr__(self):
        return f"{self.__class__.__name__}({self.corr})"

    def tree_flatten(self):
        children = (self.corr,)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        (corr,) = children
        return cls(corr)

    def init(self, s, /):
        fn = jax.vmap(type(self.corr).init)
        return fn(self.corr, s)

    def begin(self, x, /, vector_field, t, p):
        select_fn = jax.vmap(type(self.corr).select_derivatives)
        m0, m1 = select_fn(self.corr, x.hidden_state)

        fx = vector_field(*m0.T, t=t, p=p)

        marginalise_fn = jax.vmap(type(self.corr).marginalise_observation)
        cache, obs_unbatch = marginalise_fn(self.corr, fx, m1, x.hidden_state)

        mahalanobis_fn = scalar_vars.NormalQOI.mahalanobis_norm
        mahalanobis_fn_vmap = jax.vmap(mahalanobis_fn)
        output_scale = mahalanobis_fn_vmap(obs_unbatch, jnp.zeros_like(m1))
        error_estimate = obs_unbatch.cov_sqrtm_lower

        return scalar_vars.SSV(
            error_estimate=output_scale * error_estimate,
            output_scale_dynamic=output_scale,
            cache_corr=cache,
            hidden_state=x.hidden_state,
            hidden_shape=x.hidden_shape,
            cache_extra=x.cache_extra,
            backward_model=x.backward_model,
            observed_state=None,
        )

    def complete(self, extrapolated, /, vector_field, t, p):
        fn = jax.vmap(type(self.corr).complete, in_axes=(0, 0, None, None, None))
        return fn(self.corr, extrapolated, vector_field, t, p)

    def _cov_sqrtm_lower(self, cov_sqrtm_lower):
        return cov_sqrtm_lower[:, self.ode_order, ...]
