"""Batch-style implementations."""
from typing import Callable, Generic, Tuple, TypeVar

import jax
import jax.numpy as jnp

from odefilter import cubature as cubature_module
from odefilter.implementations import _collections, _ibm_util, _scalar, _sqrtm

# todo: reconsider naming!

SSV = TypeVar("SSV")


@jax.tree_util.register_pytree_node_class
class BatchNormal(_collections.StateSpaceVariable, Generic[SSV]):
    # Shapes: (d, n), (d, n, n). QOI: n=0

    def __init__(self, mean, cov_sqrtm_lower):
        self._normal = _scalar.Normal(mean, cov_sqrtm_lower)

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
    def __init__(self, ode_dimension, ode_order, cubature):
        if ode_order > 1:
            raise ValueError

        super().__init__(ode_order=ode_order)
        self.cubature = cubature
        self.ode_dimension = ode_dimension

    def tree_flatten(self):
        # todo: should this call super().tree_flatten()?
        children = (self.cubature,)
        aux = self.ode_order, self.ode_dimension
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (cubature,) = children
        ode_order, ode_dimension = aux
        return cls(ode_order=ode_order, ode_dimension=ode_dimension, cubature=cubature)

    @classmethod
    def from_params(cls, ode_dimension, ode_order):
        cubature = cubature_module.SphericalCubatureIntegration.from_params(
            ode_dimension=ode_dimension
        )
        return cls(ode_dimension=ode_dimension, ode_order=ode_order, cubature=cubature)

    def begin_correction(self, x: BatchNormal, /, vector_field, t, p):

        # Vmap relevant functions
        vmap_f = jax.vmap(jax.tree_util.Partial(vector_field, t=t, p=p))
        cache = (vmap_f,)

        # 1. x -> (e0x, e1x)
        L_X = x.cov_sqrtm_lower
        L0 = L_X[:, 0, :]  # (d, n)
        r_marg1_x = jax.vmap(_sqrtm.sqrtm_to_upper_triangular)(
            R=L0[:, :, None]
        )  # (d, 1, 1)
        r_marg1_x_squeeze = jnp.reshape(r_marg1_x, (self.ode_dimension,))
        m_marg1_x, m_marg1_y = x.mean[:, 0], x.mean[:, 1]

        # 2. (x, y) -> (f(x), y)
        x_centered = (
            self.cubature.points * r_marg1_x_squeeze[None, :]
        )  # (S, d) * (1, d) = (S, d)
        sigma_points = m_marg1_x[None, :] + x_centered
        fx = vmap_f(sigma_points)
        m_marg2 = self.cubature.weights_sqrtm**2 @ fx
        fx_centered = fx - m_marg2[None, :]
        fx_centered_normed = (
            fx_centered * self.cubature.weights_sqrtm[:, None]
        )  # (S, d)

        # 3. (x, y) -> y - x (last one)
        m_marg = m_marg1_y - m_marg2
        L1 = L_X[:, 1, :]  # (d, n)
        # (d, n) @ (n, d) + (d, S) @ (S, d) = (d, d)
        # becomes (d, 1, n) @ (d, n, 1) + (d, 1, S) @ (d, S, 1) = (D, 1, 1)
        l_marg_block = jax.vmap(_sqrtm.sum_of_sqrtm_factors)(
            R1=L1[:, :, None], R2=fx_centered_normed.T[:, :, None]
        )
        l_marg = jnp.reshape(l_marg_block, (self.ode_dimension,))

        # Summarise
        marginals = BatchScalarNormal(m_marg, l_marg)
        output_scale_sqrtm = marginals.norm_of_whitened_residual_sqrtm()

        # Compute error estimate
        error_estimate = l_marg
        return output_scale_sqrtm * error_estimate, output_scale_sqrtm, cache

    def complete_correction(self, extrapolated, cache):
        # The correction step for the cubature Kalman filter implementation
        # is quite complicated. The reason is that the observation model
        # is x -> e1(x) - f(e0(x)), i.e., a composition of a linear/nonlinear/linear
        # model, and that we _only_ want to cubature-linearise the nonlinearity.
        # So what we do is that we compute marginals, gains, and posteriors
        # for each of the three transitions and merge them in the end.
        # This uses the fewest sigma-points possible, and ultimately should
        # lead to the fastest, most stable implementation.

        # Compute the linearisation as in
        # Eq. (9) in https://arxiv.org/abs/2102.00514
        H, noise = self._linearize(x=extrapolated, cache=cache)
        # (d,), ((d,), (d,))

        # Compute the CKF correction
        L = extrapolated.cov_sqrtm_lower
        L0 = L[:, 0, :]  # (d, n)
        L1 = L[:, 1, :]  # (d, n)
        HL = L1 - H[:, None] * L0[:, :]  # (d, n) - (d, 1) @ (d, n) = (d, n)
        r_marg, (r_bw, gain) = jax.vmap(_sqrtm.revert_conditional)(
            R_X_F=HL[:, :, None],
            R_X=_transpose(L),
            R_YX=noise.cov_sqrtm_lower[:, None, None],
        )
        r_marg_squeezed = jnp.reshape(r_marg, (self.ode_dimension,))

        # Catch up the marginals
        x = extrapolated  # alias for readability in this code-block
        x0 = x.mean[:, 0]
        x1 = x.mean[:, 1]
        m_marg = x1 - (H * x0 + noise.mean)
        marginals = BatchScalarNormal(m_marg, r_marg_squeezed)

        # Catch up the backward noise and return result
        m_bw = extrapolated.mean - (gain @ m_marg[:, None, None])[:, :, 0]
        backward_noise = BatchNormal(m_bw, _transpose(r_bw))

        return marginals, (backward_noise, gain)

    def _linearize(self, x, cache):
        vmap_f, *_ = cache

        # Create sigma points
        m_0 = x.mean[:, 0]
        l_0 = x.cov_sqrtm_lower[:, 0, :]  # (d, n)
        r_0_square = jax.vmap(_sqrtm.sqrtm_to_upper_triangular)(
            R=l_0[:, :, None]
        )  # (d, n, 1) -> (d, 1, 1)
        r_0_reshaped = jnp.reshape(r_0_square, (self.ode_dimension,))  # (d,)
        pts_centered = (
            self.cubature.points * r_0_reshaped[None, :]
        )  # (S, d) * (1, d) = (S, d)
        pts = m_0[None, :] + pts_centered  # (S, d)

        # Evaluate the vector-field
        fx = vmap_f(pts)
        fx_mean = self.cubature.weights_sqrtm**2 @ fx
        fx_centered = fx - fx_mean[None, :]  # (S, d)

        # Revert the transition to get H and Omega
        # This is a pure sqrt-implementation of
        # Eq. (9) in https://arxiv.org/abs/2102.00514
        # It seems to be different to Section VI.B in
        # https://arxiv.org/abs/2207.00426,
        # because the implementation below avoids sqrt-down-dates
        pts_centered_normed = pts_centered * self.cubature.weights_sqrtm[:, None]
        fx_centered_normed = fx_centered * self.cubature.weights_sqrtm[:, None]
        # todo: with R_X_F = r_0_square, we would save a qr decomposition, right?
        #  (but would it still be valid?)
        _, (r_Om, H) = jax.vmap(_sqrtm.revert_conditional_noisefree)(
            R_X_F=pts_centered_normed.T[:, :, None],
            R_X=fx_centered_normed.T[:, :, None],
        )
        r_Om_reshaped = jnp.reshape(r_Om, (self.ode_dimension,))  # why (d,)??
        H_reshaped = jnp.reshape(H, (self.ode_dimension,))  # why (d,)??

        # Catch up the transition-mean and return the result
        d = fx_mean - H_reshaped * m_0
        return H_reshaped, BatchNormal(d, r_Om_reshaped)


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
    def ode_dimension(self):
        return self.ibm.a.shape[0]

    def tree_flatten(self):
        children = (self.ibm,)
        return children, ()

    @classmethod
    def tree_unflatten(cls, _aux, children):
        (ibm,) = children
        return cls(a=ibm.a, q_sqrtm_lower=ibm.q_sqrtm_lower)

    @classmethod
    def from_params(cls, ode_dimension, num_derivatives):
        """Create a strategy from hyperparameters."""
        a, q_sqrtm = _ibm_util.system_matrices_1d(num_derivatives=num_derivatives)
        a_stack = jnp.stack([a] * ode_dimension)
        q_sqrtm_stack = jnp.stack([q_sqrtm] * ode_dimension)
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


def _transpose(x):
    return jnp.swapaxes(x, -1, -2)
