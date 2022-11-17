"""Batch-style implementations."""
from typing import Callable, Generic, Tuple, TypeVar

import jax
import jax.numpy as jnp

from odefilter import cubature as cubature_module
from odefilter.implementations import _collections, _ibm_util, _scalar

# todo: reconsider naming!

SSV = TypeVar("SSV")


@jax.tree_util.register_pytree_node_class
class BatchNormal(_collections.StateSpaceVariable, Generic[SSV]):
    # Shapes: (d, n), (d, n, n). QOI: n=0

    def __init__(self, mean, cov_sqrtm_lower):
        self._normal = _scalar.Normal(mean, cov_sqrtm_lower)

    def __repr__(self):
        name = f"{self.__class__.__name__}"
        args = f"mean={self.mean}, cov_sqrtm_lower={self.cov_sqrtm_lower}"
        return f"{name}({args})"

    @property
    def mean(self):
        return self._normal.mean

    @property
    def cov_sqrtm_lower(self):
        return self._normal.cov_sqrtm_lower

    def tree_flatten(self):
        children = (self._normal,)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        (normal,) = children
        return cls(normal.mean, normal.cov_sqrtm_lower)

    def to_normal(self):
        return _scalar.Normal(self.mean, self.cov_sqrtm_lower)

    @classmethod
    def from_normal(cls, normal):
        return cls(normal.mean, normal.cov_sqrtm_lower)

    @property
    def sample_shape(self):
        return self._normal.sample_shape  # mean is (d, n)

    def logpdf(self, u, /):
        batch_logpdf = jax.vmap(_scalar.Normal.logpdf)(self._normal, u)
        return jnp.sum(batch_logpdf)

    def norm_of_whitened_residual_sqrtm(self) -> jax.Array:
        fn = jax.vmap(_scalar.Normal.norm_of_whitened_residual_sqrtm)
        return fn(self._normal)

    def condition_on_qoi_observation(self, u, /, observation_std):
        fn = jax.vmap(_scalar.Normal.condition_on_qoi_observation, in_axes=(0, 0, None))
        obs, (cor, gain) = fn(self._normal, u, observation_std)

        obs_batch = BatchScalarNormal(obs.mean, obs.cov_sqrtm_lower)
        cor_batch = BatchNormal(cor.mean, cor.cov_sqrtm_lower)
        return obs_batch, (cor_batch, gain)

    def extract_qoi(self):
        return jax.vmap(_scalar.Normal.extract_qoi)(self._normal)

    def extract_qoi_from_sample(self, u, /):
        fn = jax.vmap(_scalar.Normal.extract_qoi_from_sample)
        return fn(self._normal, u)

    def Ax_plus_y(self, A, x, y):
        fn = jax.vmap(_scalar.Normal.Ax_plus_y)
        return fn(self._normal, A, x, y)

    def scale_covariance(self, scale_sqrtm):
        fn = jax.vmap(_scalar.Normal.scale_covariance)
        scaled = fn(self._normal, scale_sqrtm)
        return BatchNormal(scaled.mean, scaled.cov_sqrtm_lower)

    def transform_unit_sample(self, x, /):
        fn = jax.vmap(_scalar.Normal.transform_unit_sample)
        return fn(self._normal, x)


@jax.tree_util.register_pytree_node_class
class BatchScalarNormal(_collections.StateSpaceVariable, Generic[SSV]):
    def __init__(self, mean, cov_sqrtm_lower):
        self._normal = _scalar.ScalarNormal(mean, cov_sqrtm_lower)

    @property
    def mean(self):
        return self._normal.mean

    @property
    def cov_sqrtm_lower(self):
        return self._normal.cov_sqrtm_lower

    def tree_flatten(self):
        children = (self._normal,)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        (normal,) = children
        return cls(normal.mean, normal.cov_sqrtm_lower)

    @classmethod
    def from_scalar_normal(cls, normal):
        return cls(normal.mean, normal.cov_sqrtm_lower)

    @property
    def sample_shape(self):
        return self._normal.sample_shape  # mean is (d, n)

    def logpdf(self, u, /):
        batch_logpdf = jax.vmap(_scalar.ScalarNormal.logpdf)(self._normal, u)
        return jnp.sum(batch_logpdf)

    def norm_of_whitened_residual_sqrtm(self) -> jax.Array:
        fn = jax.vmap(_scalar.ScalarNormal.norm_of_whitened_residual_sqrtm)
        return fn(self._normal)

    def condition_on_qoi_observation(self, u, /, observation_std):
        raise NotImplementedError

    def extract_qoi(self):
        raise NotImplementedError

    def extract_qoi_from_sample(self, u, /):
        raise NotImplementedError

    def Ax_plus_y(self, A, x, y):
        raise NotImplementedError

    def scale_covariance(self, scale_sqrtm):
        raise NotImplementedError

    def transform_unit_sample(self, x, /):
        raise NotImplementedError


BatchMM1CacheType = Tuple[Callable]
"""Type of the correction-cache."""


@jax.tree_util.register_pytree_node_class
class BatchMomentMatching(
    _collections.AbstractCorrection[BatchNormal, BatchMM1CacheType]
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

    def begin_correction(self, x: BatchNormal, /, vector_field, t, p):
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
        obs = BatchScalarNormal.from_scalar_normal(obs_unb)
        cor = BatchNormal.from_normal(cor_unb)
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
        noise = BatchScalarNormal.from_scalar_normal(noise_unb)
        return H, noise


BatchTS0CacheType = Tuple[jax.Array]

_BatchTS0Base = _collections.AbstractCorrection[BatchNormal, BatchTS0CacheType]


@jax.tree_util.register_pytree_node_class
class BatchTaylorZerothOrder(_BatchTS0Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ts0 = _scalar.TaylorZerothOrder(*args, **kwargs)

    def begin_correction(self, x: BatchNormal, /, vector_field, t, p):
        x_unbatch = _scalar.Normal(x.mean, x.cov_sqrtm_lower)

        select_fn = jax.vmap(_scalar.TaylorZerothOrder.select_derivatives)
        m0, m1 = select_fn(self._ts0, x_unbatch)

        fx = vector_field(*m0.T, t=t, p=p)

        marginalise_fn = jax.vmap(_scalar.TaylorZerothOrder.marginalise_observation)
        cache, obs_unbatch = marginalise_fn(self._ts0, fx, m1, x)
        observed = BatchScalarNormal(obs_unbatch.mean, obs_unbatch.cov_sqrtm_lower)

        output_scale_sqrtm = observed.norm_of_whitened_residual_sqrtm()
        error_estimate = observed.cov_sqrtm_lower
        return output_scale_sqrtm * error_estimate, output_scale_sqrtm, cache

    def complete_correction(self, extrapolated: BatchNormal, cache: BatchTS0CacheType):
        extra_unbatch = _scalar.Normal(extrapolated.mean, extrapolated.cov_sqrtm_lower)
        fn = jax.vmap(_scalar.TaylorZerothOrder.complete_correction)
        obs_unbatch, (cor_unbatch, gain) = fn(self._ts0, extra_unbatch, cache)

        obs = BatchScalarNormal(obs_unbatch.mean, obs_unbatch.cov_sqrtm_lower)
        cor = BatchNormal(cor_unbatch.mean, cor_unbatch.cov_sqrtm_lower)
        return obs, (cor, gain)

    def _cov_sqrtm_lower(self, cov_sqrtm_lower):
        return cov_sqrtm_lower[:, self.ode_order, ...]


@jax.tree_util.register_pytree_node_class
class BatchConditional(_collections.AbstractConditional):
    def __init__(self, transition, noise):
        noise = _scalar.Normal(noise.mean, noise.cov_sqrtm_lower)
        self.conditional = _scalar.Conditional(transition, noise=noise)

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(transition={self.transition}, noise={self.noise})"

    @property
    def transition(self):
        return self.conditional.transition

    @property
    def noise(self):
        return self.conditional.noise

    def tree_flatten(self):
        children = (self.conditional,)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        (conditional,) = children
        return cls(transition=conditional.transition, noise=conditional.noise)

    def __call__(self, x, /):
        out = jax.vmap(_scalar.Conditional.__call__)(self.conditional, x)
        return BatchNormal(out.mean, out.cov_sqrtm_lower)

    def scale_covariance(self, scale_sqrtm):
        out = jax.vmap(_scalar.Conditional.scale_covariance)(
            self.conditional, scale_sqrtm
        )
        noise = BatchNormal(out.noise.mean, out.noise.cov_sqrtm_lower)
        return BatchConditional(transition=out.transition, noise=noise)

    def merge_with_incoming_conditional(self, incoming, /):
        fn = jax.vmap(_scalar.Conditional.merge_with_incoming_conditional)
        merged = fn(self.conditional, incoming.conditional)
        noise = BatchNormal(merged.noise.mean, merged.noise.cov_sqrtm_lower)
        return BatchConditional(transition=merged.transition, noise=noise)

    def marginalise(self, rv, /):
        marginalised = jax.vmap(_scalar.Conditional.marginalise)(self.conditional, rv)
        return BatchNormal(marginalised.mean, marginalised.cov_sqrtm_lower)


BatchIBMCacheType = Tuple[jax.Array]  # Cache type
"""Type of the extrapolation-cache."""


@jax.tree_util.register_pytree_node_class
class BatchIBM(_collections.AbstractExtrapolation[BatchNormal, BatchIBMCacheType]):
    def __init__(self, a, q_sqrtm_lower):
        self.ibm = _scalar.IBM(a, q_sqrtm_lower)

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(a={self.a}, q_sqrtm_lower={self.q_sqrtm_lower})"

    @property
    def num_derivatives(self):
        return self.ibm.a.shape[1] - 1

    @property
    def ode_shape(self):
        return (
            self.ibm.a.shape[0],
        )  # todo: this does not scale to matrix-valued problems

    def tree_flatten(self):
        children = (self.ibm,)
        return children, ()

    @classmethod
    def tree_unflatten(cls, _aux, children):
        (ibm,) = children
        return cls(a=ibm.a, q_sqrtm_lower=ibm.q_sqrtm_lower)

    @classmethod
    def from_params(cls, ode_shape, num_derivatives):
        """Create a strategy from hyperparameters."""
        assert len(ode_shape) == 1
        (n,) = ode_shape
        a, q_sqrtm = _ibm_util.system_matrices_1d(num_derivatives=num_derivatives)
        a_stack, q_sqrtm_stack = _tree_stack_duplicates((a, q_sqrtm), n=n)
        return cls(a=a_stack, q_sqrtm_lower=q_sqrtm_stack)

    def begin_extrapolation(self, m0, /, dt):
        fn = jax.vmap(_scalar.IBM.begin_extrapolation, in_axes=(0, 0, None))
        extra, cache = fn(self.ibm, m0, dt)
        extra_batch = BatchNormal(extra.mean, extra.cov_sqrtm_lower)
        return extra_batch, cache

    def complete_extrapolation(self, linearisation_pt, cache, l0, output_scale_sqrtm):
        fn = jax.vmap(_scalar.IBM.complete_extrapolation)
        ext = fn(self.ibm, linearisation_pt, cache, l0, output_scale_sqrtm)
        return BatchNormal(ext.mean, ext.cov_sqrtm_lower)

    def init_conditional(self, rv_proto):
        conditional = jax.vmap(_scalar.IBM.init_conditional)(self.ibm, rv_proto)
        noise = BatchNormal(conditional.noise.mean, conditional.noise.cov_sqrtm_lower)
        return BatchConditional(conditional.transition, noise=noise)

    def init_corrected(self, taylor_coefficients):
        cor = jax.vmap(_scalar.IBM.init_corrected)(self.ibm, taylor_coefficients)
        return BatchNormal(cor.mean, cor.cov_sqrtm_lower)

    # todo: move to correction?
    def init_error_estimate(self):
        return jax.vmap(_scalar.IBM.init_error_estimate)(self.ibm)

    # todo: move to correction?
    def init_output_scale_sqrtm(self):
        return jax.vmap(_scalar.IBM.init_output_scale_sqrtm)(self.ibm)

    def revert_markov_kernel(self, linearisation_pt, l0, output_scale_sqrtm, cache):
        fn = jax.vmap(_scalar.IBM.revert_markov_kernel)
        ext, bw_model = fn(self.ibm, linearisation_pt, cache, l0, output_scale_sqrtm)

        ext_batched = BatchNormal(ext.mean, ext.cov_sqrtm_lower)
        bw_noise = BatchNormal(bw_model.noise.mean, bw_model.noise.cov_sqrtm_lower)
        bw_model_batched = BatchConditional(bw_model.transition, bw_noise)
        return ext_batched, bw_model_batched


def _tree_stack_duplicates(tree, n):
    return jax.tree_util.tree_map(lambda s: jnp.vstack([s[None, ...]] * n), tree)
