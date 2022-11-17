"""Batch-style corrections."""
from typing import Callable, Tuple

import jax

from probdiffeq import cubature as cubature_module
from probdiffeq.implementations import _collections, _scalar
from probdiffeq.implementations.batch import _vars

_MM1CacheType = Tuple[Callable]
"""Type of the correction-cache."""


@jax.tree_util.register_pytree_node_class
class BatchMomentMatching(
    _collections.AbstractCorrection[_vars.BatchNormal, _MM1CacheType]
):
    def __init__(self, ode_shape, ode_order, cubature):
        if ode_order > 1:
            raise ValueError

        super().__init__(ode_order=ode_order)
        self.ode_shape = ode_shape

        self._mm = _scalar.MomentMatching(ode_order=ode_order, cubature=cubature)

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
        cubature_fn = cubature_module.ThirdOrderSpherical.from_params_batch
        cubature = cubature_fn(input_shape=ode_shape)
        return cls(ode_shape=ode_shape, ode_order=ode_order, cubature=cubature)

    def begin_correction(self, x: _vars.BatchNormal, /, vector_field, t, p):
        # Unvmap
        extrapolated = x.to_normal()

        # Vmap relevant functions
        vmap_f = jax.vmap(jax.tree_util.Partial(vector_field, t=t, p=p))
        cache = (vmap_f,)

        # Evaluate vector field at sigma-points
        sigma_points_fn = jax.vmap(_scalar.MomentMatching.transform_sigma_points)
        sigma_points, _, _ = sigma_points_fn(self._mm, extrapolated)

        fx = vmap_f(sigma_points.T).T  # (d, S).T = (S, d) -> (S, d) -> transpose again
        center_fn = jax.vmap(_scalar.MomentMatching.center)
        fx_mean, _, fx_centered_normed = center_fn(self._mm, fx)

        # Compute output scale and error estimate
        calibrate_fn = jax.vmap(_scalar.MomentMatching.calibrate)
        error_estimate, output_scale_sqrtm = calibrate_fn(
            self._mm, fx_mean, fx_centered_normed, extrapolated
        )
        return output_scale_sqrtm * error_estimate, output_scale_sqrtm, cache

    def complete_correction(self, extrapolated, cache):
        # Unvmap
        extra = extrapolated.to_normal()
        (vmap_f,) = cache

        H, noise = self.linearize(extra, vmap_f)

        fn = jax.vmap(_scalar.MomentMatching.complete_correction_post_linearize)
        obs_unb, (cor_unb, gain) = fn(self._mm, H, extra, noise)

        # Vmap
        obs = _vars.BatchScalarNormal.from_scalar_normal(obs_unb)
        cor = _vars.BatchNormal.from_normal(cor_unb)
        return obs, (cor, gain)

    def linearize(self, extrapolated, vmap_f):
        # Transform the sigma-points
        sigma_points_fn = jax.vmap(_scalar.MomentMatching.transform_sigma_points)
        sigma_points, _, sigma_points_centered_normed = sigma_points_fn(
            self._mm, extrapolated
        )

        # Evaluate the vector field at the sigma-points
        fx = vmap_f(sigma_points.T).T  # (d, S).T = (S, d) -> (S, d) -> transpose again
        center_fn = jax.vmap(_scalar.MomentMatching.center)
        fx_mean, _, fx_centered_normed = center_fn(self._mm, fx)

        # Complete the linearization
        lin_fn = jax.vmap(_scalar.MomentMatching.linearization_matrices)
        H, noise_unb = lin_fn(
            self._mm,
            fx_centered_normed,
            fx_mean,
            sigma_points_centered_normed,
            extrapolated,
        )
        noise = _vars.BatchScalarNormal.from_scalar_normal(noise_unb)
        return H, noise


_TS0CacheType = Tuple[jax.Array]

_BatchTS0Base = _collections.AbstractCorrection[_vars.BatchNormal, _TS0CacheType]


@jax.tree_util.register_pytree_node_class
class BatchTaylorZerothOrder(_BatchTS0Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ts0 = _scalar.TaylorZerothOrder(*args, **kwargs)

    def begin_correction(self, x: _vars.BatchNormal, /, vector_field, t, p):
        x_unbatch = _scalar.Normal(x.mean, x.cov_sqrtm_lower)

        select_fn = jax.vmap(_scalar.TaylorZerothOrder.select_derivatives)
        m0, m1 = select_fn(self._ts0, x_unbatch)

        fx = vector_field(*m0.T, t=t, p=p)

        marginalise_fn = jax.vmap(_scalar.TaylorZerothOrder.marginalise_observation)
        cache, obs_unbatch = marginalise_fn(self._ts0, fx, m1, x)
        observed = _vars.BatchScalarNormal(
            obs_unbatch.mean, obs_unbatch.cov_sqrtm_lower
        )

        output_scale_sqrtm = observed.norm_of_whitened_residual_sqrtm()
        error_estimate = observed.cov_sqrtm_lower
        return output_scale_sqrtm * error_estimate, output_scale_sqrtm, cache

    def complete_correction(
        self, extrapolated: _vars.BatchNormal, cache: _TS0CacheType
    ):
        extra_unbatch = _scalar.Normal(extrapolated.mean, extrapolated.cov_sqrtm_lower)
        fn = jax.vmap(_scalar.TaylorZerothOrder.complete_correction)
        obs_unbatch, (cor_unbatch, gain) = fn(self._ts0, extra_unbatch, cache)

        obs = _vars.BatchScalarNormal(obs_unbatch.mean, obs_unbatch.cov_sqrtm_lower)
        cor = _vars.BatchNormal(cor_unbatch.mean, cor_unbatch.cov_sqrtm_lower)
        return obs, (cor, gain)

    def _cov_sqrtm_lower(self, cov_sqrtm_lower):
        return cov_sqrtm_lower[:, self.ode_order, ...]
