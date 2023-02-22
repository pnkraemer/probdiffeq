"""Corrections."""
from typing import Callable, Tuple

import jax

from probdiffeq import cubature as cubature_module
from probdiffeq.implementations import _collections, _scalar

_SLR1CacheType = Tuple[Callable]
"""Type-variable for the correction-cache."""


@jax.tree_util.register_pytree_node_class
class BlockDiagStatisticalFirstOrder(_collections.AbstractCorrection):
    """First-order statistical linear regression in state-space models \
     with block-diagonal covariance structure.

    !!! warning "Warning: highly EXPERIMENTAL feature!"
        This feature is highly experimental.
        There is no guarantee that it works correctly.
        It might be deleted tomorrow
        and without any deprecation policy.

    """

    def __init__(self, ode_shape, ode_order, cubature):
        if ode_order > 1:
            raise ValueError

        super().__init__(ode_order=ode_order)
        self.ode_shape = ode_shape

        self._mm = _scalar.StatisticalFirstOrder(ode_order=ode_order, cubature=cubature)

    @property
    def cubature(self):
        return self._mm.cubature

    def tree_flatten(self):
        # todo: should this call super().tree_flatten()?
        children = (self.cubature,)
        aux = self.ode_order, self.ode_shape
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (cubature,) = children
        ode_order, ode_shape = aux
        return cls(ode_order=ode_order, ode_shape=ode_shape, cubature=cubature)

    @classmethod
    def from_params(cls, ode_shape, ode_order):
        cubature_fn = cubature_module.ThirdOrderSpherical.from_params_blockdiag
        cubature = cubature_fn(input_shape=ode_shape)
        return cls(ode_shape=ode_shape, ode_order=ode_order, cubature=cubature)

    def begin_correction(self, extrapolated, /, vector_field, t, p):
        # Vmap relevant functions
        vmap_f = jax.vmap(jax.tree_util.Partial(vector_field, t=t, p=p))
        cache = (vmap_f,)

        # Evaluate vector field at sigma-points
        sigma_points_fn = jax.vmap(_scalar.StatisticalFirstOrder.transform_sigma_points)
        sigma_points, _, _ = sigma_points_fn(self._mm, extrapolated.hidden_state)

        fx = vmap_f(sigma_points.T).T  # (d, S).T = (S, d) -> (S, d) -> transpose again
        center_fn = jax.vmap(_scalar.StatisticalFirstOrder.center)
        fx_mean, _, fx_centered_normed = center_fn(self._mm, fx)

        # Compute output scale and error estimate
        calibrate_fn = jax.vmap(_scalar.StatisticalFirstOrder.calibrate)
        error_estimate, output_scale_sqrtm = calibrate_fn(
            self._mm, fx_mean, fx_centered_normed, extrapolated.hidden_state
        )
        return output_scale_sqrtm * error_estimate, output_scale_sqrtm, cache

    def complete_correction(self, extrapolated, cache):
        (vmap_f,) = cache

        H, noise = self.linearize(extrapolated, vmap_f)

        fn = jax.vmap(_scalar.StatisticalFirstOrder.complete_correction_post_linearize)
        return fn(self._mm, H, extrapolated.hidden_state, noise)

    def linearize(self, extrapolated, vmap_f):
        # Transform the sigma-points
        sigma_points_fn = jax.vmap(_scalar.StatisticalFirstOrder.transform_sigma_points)
        sigma_points, _, sigma_points_centered_normed = sigma_points_fn(
            self._mm, extrapolated.hidden_state
        )

        # Evaluate the vector field at the sigma-points
        fx = vmap_f(sigma_points.T).T  # (d, S).T = (S, d) -> (S, d) -> transpose again
        center_fn = jax.vmap(_scalar.StatisticalFirstOrder.center)
        fx_mean, _, fx_centered_normed = center_fn(self._mm, fx)

        # Complete the linearization
        lin_fn = jax.vmap(_scalar.StatisticalFirstOrder.linearization_matrices)
        return lin_fn(
            self._mm,
            fx_centered_normed,
            fx_mean,
            sigma_points_centered_normed,
            extrapolated.hidden_state,
        )


_TS0CacheType = Tuple[jax.Array]


@jax.tree_util.register_pytree_node_class
class BlockDiagTaylorZerothOrder(_collections.AbstractCorrection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ts0 = _scalar.TaylorZerothOrder(*args, **kwargs)

    def begin_correction(self, x, /, vector_field, t, p):
        select_fn = jax.vmap(_scalar.TaylorZerothOrder.select_derivatives)
        m0, m1 = select_fn(self._ts0, x.hidden_state)

        fx = vector_field(*m0.T, t=t, p=p)

        marginalise_fn = jax.vmap(_scalar.TaylorZerothOrder.marginalise_observation)
        cache, obs_unbatch = marginalise_fn(self._ts0, fx, m1, x.hidden_state)

        output_scale_sqrtm_fn = jax.vmap(
            _scalar.ScalarNormal.norm_of_whitened_residual_sqrtm
        )
        output_scale_sqrtm = output_scale_sqrtm_fn(obs_unbatch)
        error_estimate = obs_unbatch.cov_sqrtm_lower
        return output_scale_sqrtm * error_estimate, output_scale_sqrtm, cache

    def complete_correction(self, extrapolated, cache: _TS0CacheType):
        fn = jax.vmap(_scalar.TaylorZerothOrder.complete_correction)
        return fn(self._ts0, extrapolated, cache)

    def _cov_sqrtm_lower(self, cov_sqrtm_lower):
        return cov_sqrtm_lower[:, self.ode_order, ...]
