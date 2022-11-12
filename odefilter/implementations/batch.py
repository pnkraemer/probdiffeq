"""Batch-style implementations."""
from typing import Any, Callable, Generic, Tuple, TypeVar

import jax
import jax.numpy as jnp

from odefilter import cubature as cubature_module
from odefilter.implementations import _collections, _ibm_util, _scalar, _sqrtm

# todo: reconsider naming!

SSV = TypeVar("SSV")


@jax.tree_util.register_pytree_node_class
class BatchVariable(_collections.StateSpaceVariable, Generic[SSV]):
    def __init__(self, normal: SSV, /):
        self.normal = normal
        self.wrap_type = type(normal)

    def tree_flatten(self):
        children = (self.normal,)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        (normal,) = children
        return cls(normal)

    @property
    def sample_shape(self):
        return self.normal.sample_shape  # mean is (d, n)

    @property
    def mean(self):
        return self.normal.mean

    @property
    def cov_sqrtm_lower(self):
        return self.normal.cov_sqrtm_lower

    def logpdf(self, u, /) -> float:
        batch_logpdf = jax.vmap(self.wrap_type.logpdf)(self.normal, u)
        return jnp.sum(batch_logpdf)

    def norm_of_whitened_residual_sqrtm(self) -> jax.Array:
        fn = jax.vmap(self.wrap_type.norm_of_whitened_residual_sqrtm)
        return fn(self.normal)

    def condition_on_qoi_observation(
        self, u, /, observation_std
    ) -> Tuple["BatchVariable[Any]", Tuple["BatchVariable[SSV]", jax.Array]]:
        fn = jax.vmap(self.wrap_type.condition_on_qoi_observation, in_axes=(0, 0, None))
        print(self.normal.sample_shape, u.shape, observation_std.shape)
        obs, (cor, gain) = fn(self.normal, u, observation_std)
        return BatchVariable(obs), (BatchVariable(cor), gain)

    def extract_qoi(self) -> "BatchVariable[Any]":
        return jax.vmap(self.wrap_type.extract_qoi)(self.normal)

    def extract_qoi_from_sample(self, u, /):
        fn = jax.vmap(self.wrap_type.extract_qoi_from_sample)
        return fn(self.normal, u)

    def Ax_plus_y(self, A, x, y):
        fn = jax.vmap(self.wrap_type.Ax_plus_y)
        return fn(self.normal, A, x, y)

    def scale_covariance(self, scale_sqrtm) -> "BatchVariable[SSV]":
        fn = jax.vmap(self.wrap_type.scale_covariance)
        scaled = fn(self.normal, scale_sqrtm)
        return BatchVariable(scaled)

    def transform_unit_sample(self, x, /):
        fn = jax.vmap(self.wrap_type.transform_unit_sample)
        return fn(self.normal, x)


@jax.tree_util.register_pytree_node_class
class BatchNormal(_collections.StateSpaceVariable):
    """Batched normally-distributed random variables."""

    def __init__(self, mean, cov_sqrtm_lower):
        self.mean = mean  # (d, k) shape
        self.cov_sqrtm_lower = cov_sqrtm_lower  # (d, k, k) shape

    def __repr__(self):
        name = f"{self.__class__.__name__}"
        args = f"mean={self.mean}, cov_sqrtm_lower={self.cov_sqrtm_lower}"
        return f"{name}({args})"

    def tree_flatten(self):
        children = self.mean, self.cov_sqrtm_lower
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        mean, cov_sqrtm_lower = children
        return cls(mean=mean, cov_sqrtm_lower=cov_sqrtm_lower)

    def logpdf(self, u, /):
        m_obs, l_obs = self.mean, self.cov_sqrtm_lower

        # todo: is this correct??
        res_white = (m_obs - u) / jnp.reshape(l_obs, m_obs.shape)
        x1 = jnp.dot(res_white, res_white.T)
        x2 = jnp.sum(jnp.reshape(l_obs, m_obs.shape) ** 2)
        x3 = res_white.size * jnp.log(jnp.pi * 2)
        return -0.5 * (x1 + x2 + x3)

    def norm_of_whitened_residual_sqrtm(self):
        obs_pt, l_obs = self.mean, self.cov_sqrtm_lower  # (d,), (d,)

        res_white = obs_pt / l_obs  # (d, )
        evidence_sqrtm = res_white / jnp.sqrt(res_white.size)

        return evidence_sqrtm

    def condition_on_qoi_observation(self, u, /, *, observation_std):
        hc = self.cov_sqrtm_lower[:, 0, ...]  # (d, k)
        m_obs = self.mean[:, 0]  # (d,)
        d = m_obs.shape[0]
        r_yx = observation_std * jnp.ones((d, 1, 1))

        r_x_f = hc[..., None]
        r_x = _transpose(self.cov_sqrtm_lower)
        r_obs, (r_cor, gain) = jax.vmap(_sqrtm.revert_conditional)(
            R_X_F=r_x_f, R_X=r_x, R_YX=r_yx
        )
        m_cor = self.mean - (gain @ (m_obs - u)[:, None, None])[..., 0]

        obs = BatchNormal(m_obs, _transpose(r_obs))
        cor = BatchNormal(m_cor, _transpose(r_cor))
        return obs, (cor, gain)

    def extract_qoi(self):
        return self.mean[..., 0]

    def extract_qoi_from_sample(self, u, /):
        return u[..., 0]

    def scale_covariance(self, *, scale_sqrtm):
        # Endpoint: (d, 1, 1) * (d, k, k) -> (d, k, k)
        return BatchNormal(
            mean=self.mean,
            cov_sqrtm_lower=scale_sqrtm[..., None, None] * self.cov_sqrtm_lower,
        )

    def transform_unit_sample(self, x, /):
        # (d,k) + ((d,k,k) @ (d,k,1))[..., 0] = (d,k)
        return self.mean + (self.cov_sqrtm_lower @ x[..., None])[..., 0]

    def Ax_plus_y(self, *, A, x, y):
        return (A @ x[..., None])[..., 0] + y

    @property
    def sample_shape(self):
        return self.mean.shape


BatchMM1CacheType = Tuple[Callable]
"""Type of the correction-cache."""


@jax.tree_util.register_pytree_node_class
class BatchMomentMatching(
    _collections.AbstractCorrection[BatchNormal, BatchMM1CacheType]
):
    def __init__(self, *, ode_dimension, ode_order, cubature):
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
    def from_params(cls, *, ode_dimension, ode_order):
        cubature = cubature_module.SphericalCubatureIntegration.from_params(
            ode_dimension=ode_dimension
        )
        return cls(ode_dimension=ode_dimension, ode_order=ode_order, cubature=cubature)

    def begin_correction(self, x: BatchNormal, /, *, vector_field, t, p):

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
        marginals = BatchNormal(m_marg, l_marg)
        output_scale_sqrtm = marginals.norm_of_whitened_residual_sqrtm()

        # Compute error estimate
        error_estimate = l_marg
        return output_scale_sqrtm * error_estimate, output_scale_sqrtm, cache

    def complete_correction(self, *, extrapolated, cache):
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
        marginals = BatchNormal(m_marg, r_marg_squeezed)

        # Catch up the backward noise and return result
        m_bw = extrapolated.mean - (gain @ m_marg[:, None, None])[:, :, 0]
        backward_noise = BatchNormal(m_bw, _transpose(r_bw))

        return marginals, (backward_noise, gain)

    def _linearize(self, *, x, cache):
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
    def begin_correction(
        self, x: BatchNormal, /, *, vector_field, t, p
    ) -> Tuple[jax.Array, float, BatchTS0CacheType]:
        m = x.mean

        # m has shape (d, n)
        m1 = m[..., self.ode_order]
        m0 = m[..., : self.ode_order].T
        bias = m1 - vector_field(*m0, t=t, p=p)
        l_obs_nonsquare = self._cov_sqrtm_lower(x.cov_sqrtm_lower)

        l_obs_nonsquare_1 = l_obs_nonsquare[..., None]  # (d, k, 1)

        # (d, 1, 1)
        l_obs_raw = jax.vmap(_sqrtm.sqrtm_to_upper_triangular)(R=l_obs_nonsquare_1)
        l_obs = l_obs_raw[..., 0, 0]  # (d,)

        observed = BatchVariable(_scalar.ScalarNormal(mean=bias, cov_sqrtm_lower=l_obs))
        output_scale_sqrtm = observed.norm_of_whitened_residual_sqrtm()

        error_estimate = l_obs  # (d,)
        return output_scale_sqrtm * error_estimate, output_scale_sqrtm, (bias,)

    def complete_correction(
        self, *, extrapolated: BatchNormal, cache: BatchTS0CacheType
    ):
        (bias,) = cache

        # (d, k), (d, k, k)
        m_ext, l_ext = extrapolated.mean, extrapolated.cov_sqrtm_lower
        l_obs_nonsquare = self._cov_sqrtm_lower(l_ext)  # (d, k)

        # (d, 1, 1)
        l_obs = _sqrtm.sqrtm_to_upper_triangular(R=l_obs_nonsquare[..., None])
        l_obs_scalar = l_obs[..., 0, 0]  # (d,)

        # (d,), (d,)
        observed = BatchVariable(
            _scalar.ScalarNormal(mean=bias, cov_sqrtm_lower=l_obs_scalar)
        )

        # (d, k)
        crosscov = (l_ext @ l_obs_nonsquare[..., None])[..., 0]

        gain = crosscov / (l_obs_scalar[..., None]) ** 2  # (d, k)

        m_cor = m_ext - (gain * bias[..., None])  # (d, k)
        l_cor = l_ext - gain[..., None] * l_obs_nonsquare[..., None, :]  # (d, k, k)

        corrected = BatchVariable(_scalar.Normal(mean=m_cor, cov_sqrtm_lower=l_cor))
        return observed, (corrected, gain)

    def _cov_sqrtm_lower(self, cov_sqrtm_lower):
        return cov_sqrtm_lower[:, self.ode_order, ...]


@jax.tree_util.register_pytree_node_class
class BatchConditional(_collections.AbstractConditional):
    def __call__(self, x, /):
        m = (self.transition @ x[..., None])[..., 0] + self.noise.mean
        return BatchVariable(_scalar.Normal(m, self.noise.cov_sqrtm_lower))

    def scale_covariance(self, *, scale_sqrtm):
        noise = self.noise.scale_covariance(scale_sqrtm=scale_sqrtm)
        return BatchConditional(transition=self.transition, noise=noise)

    def merge_with_incoming_conditional(self, incoming, /):
        # (d, k, k); (d, k), (d, k, k)
        A = self.transition
        (b, B_sqrtm) = (self.noise.mean, self.noise.cov_sqrtm_lower)

        # (d, k, k); (d, k), (d, k, k)
        C = incoming.transition
        (d, D_sqrtm) = (incoming.noise.mean, incoming.noise.cov_sqrtm_lower)

        g = A @ C
        xi = (A @ d[..., None])[..., 0] + b
        Xi_r = jax.vmap(_sqrtm.sum_of_sqrtm_factors)(
            R1=_transpose(A @ D_sqrtm), R2=_transpose(B_sqrtm)
        )
        Xi = _transpose(Xi_r)

        noise = BatchVariable(_scalar.Normal(mean=xi, cov_sqrtm_lower=Xi))
        return BatchConditional(g, noise=noise)

    def marginalise(self, rv, /):

        # Read
        m0_p = rv.mean
        l0_p = rv.cov_sqrtm_lower

        # Apply transition
        m_new = (self.transition @ m0_p[..., None])[..., 0] + self.noise.mean
        r_new = jax.vmap(_sqrtm.sum_of_sqrtm_factors)(
            R1=_transpose(self.transition @ l0_p),
            R2=_transpose(self.noise.cov_sqrtm_lower),
        )
        l_new = _transpose(r_new)

        return BatchVariable(_scalar.Normal(mean=m_new, cov_sqrtm_lower=l_new))


BatchIBMCacheType = Tuple[jax.Array]  # Cache type
"""Type of the extrapolation-cache."""


@jax.tree_util.register_pytree_node_class
class BatchIBM(_collections.AbstractExtrapolation[BatchNormal, BatchIBMCacheType]):
    """Handle block-diagonal covariances."""

    def __init__(self, a, q_sqrtm_lower):
        self.a = a
        self.q_sqrtm_lower = q_sqrtm_lower

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(a={self.a}, q_sqrtm_lower={self.q_sqrtm_lower})"

    @property
    def num_derivatives(self):
        return self.a.shape[1] - 1

    @property
    def ode_dimension(self):
        return self.a.shape[0]

    def tree_flatten(self):
        children = self.a, self.q_sqrtm_lower
        return children, ()

    @classmethod
    def tree_unflatten(cls, _aux, children):
        a, q_sqrtm_lower = children
        return cls(a=a, q_sqrtm_lower=q_sqrtm_lower)

    @classmethod
    def from_params(cls, *, ode_dimension, num_derivatives):
        """Create a strategy from hyperparameters."""
        a, q_sqrtm = _ibm_util.system_matrices_1d(num_derivatives=num_derivatives)
        a = jnp.stack([a] * ode_dimension)
        q_sqrtm = jnp.stack([q_sqrtm] * ode_dimension)
        return cls(a=a, q_sqrtm_lower=q_sqrtm)

    def _assemble_preconditioner(self, *, dt):
        p, p_inv = _ibm_util.preconditioner_diagonal(
            dt=dt, num_derivatives=self.num_derivatives
        )
        p = jnp.stack([p] * self.ode_dimension)
        p_inv = jnp.stack([p_inv] * self.ode_dimension)
        return p, p_inv

    def complete_extrapolation(
        self, *, linearisation_pt, cache, l0, output_scale_sqrtm
    ):
        *_, p, p_inv = cache
        m_ext = linearisation_pt.mean
        r_ext_p = jax.vmap(_sqrtm.sum_of_sqrtm_factors)(
            R1=_transpose(self.a @ (p_inv[..., None] * l0)),
            R2=_transpose(output_scale_sqrtm[:, None, None] * self.q_sqrtm_lower),
        )
        l_ext_p = _transpose(r_ext_p)
        l_ext = p[..., None] * l_ext_p
        return BatchVariable(_scalar.Normal(mean=m_ext, cov_sqrtm_lower=l_ext))

    def begin_extrapolation(self, m0, /, *, dt):
        p, p_inv = self._assemble_preconditioner(dt=dt)
        m0_p = p_inv * m0  # (d, k)
        m_ext_p = (self.a @ m0_p[..., None])[..., 0]  # (d, k)
        m_ext = p * m_ext_p

        q_ext = p[..., None] * self.q_sqrtm_lower
        return BatchVariable(_scalar.Normal(m_ext, q_ext)), (m_ext_p, m0_p, p, p_inv)

    def init_conditional(self, *, rv_proto):
        noi = self._init_backward_noise(rv_proto=rv_proto)
        op = self._init_backward_transition()
        return BatchConditional(op, noise=noi)

    def _init_backward_noise(self, *, rv_proto):
        return BatchVariable(
            _scalar.Normal(
                mean=jnp.zeros_like(rv_proto.mean),
                cov_sqrtm_lower=jnp.zeros_like(rv_proto.cov_sqrtm_lower),
            )
        )

    def _init_backward_transition(self):
        return jnp.stack([jnp.eye(self.num_derivatives + 1)] * self.ode_dimension)

    def init_corrected(self, *, taylor_coefficients):
        """Initialise the "corrected" RV by stacking Taylor coefficients."""
        m0_matrix = jnp.vstack(taylor_coefficients).T
        c_sqrtm0_corrected = jnp.zeros_like(self.q_sqrtm_lower)
        return BatchVariable(
            _scalar.Normal(mean=m0_matrix, cov_sqrtm_lower=c_sqrtm0_corrected)
        )

    # todo: move to correction?
    def init_error_estimate(self):
        return jnp.zeros((self.ode_dimension,))  # the initialisation is error-free

    # todo: move to correction?
    def init_output_scale_sqrtm(self):
        return jnp.ones((self.ode_dimension,))

    def revert_markov_kernel(self, *, linearisation_pt, l0, output_scale_sqrtm, cache):
        m_ext_p, m0_p, p, p_inv = cache
        m_ext = linearisation_pt.mean

        # (d, k, 1) * (d, k, k) = (d, k, k)
        l0_p = p_inv[..., None] * l0

        r_ext_p, (r_bw_p, g_bw_p) = jax.vmap(_sqrtm.revert_conditional)(
            R_X_F=_transpose(self.a @ l0_p),
            R_X=_transpose(l0_p),
            # transpose((d, 1, 1) * (d, k, k)) = (d, k, k)
            R_YX=_transpose(output_scale_sqrtm[..., None, None] * self.q_sqrtm_lower),
        )
        l_ext_p, l_bw_p = _transpose(r_ext_p), _transpose(r_bw_p)
        m_bw_p = m0_p - (g_bw_p @ m_ext_p[..., None])[..., 0]

        # Un-apply the pre-conditioner.
        # The backward models remains preconditioned, because
        # we do backward passes in preconditioner-space.
        l_ext = p[..., None] * l_ext_p
        m_bw = p * m_bw_p
        l_bw = p[..., None] * l_bw_p
        g_bw = p[..., None] * g_bw_p * p_inv[:, None, :]

        backward_noise = BatchVariable(_scalar.Normal(mean=m_bw, cov_sqrtm_lower=l_bw))
        bw_model = BatchConditional(g_bw, noise=backward_noise)
        extrapolated = BatchVariable(_scalar.Normal(mean=m_ext, cov_sqrtm_lower=l_ext))
        return extrapolated, bw_model


def _transpose(x):
    return jnp.swapaxes(x, -1, -2)
