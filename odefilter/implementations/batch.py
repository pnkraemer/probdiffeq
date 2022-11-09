"""Batch-style extrapolations."""
import dataclasses
from typing import Any, Callable, Generic, Tuple, TypeVar

import jax
import jax.numpy as jnp

from odefilter import _control_flow
from odefilter import cubature as cubature_module
from odefilter.implementations import (
    _ibm_util,
    _sqrtm,
    correction,
    extrapolation,
    variable,
)

# todo: reconsider naming!
# todo: extract _BatchCorrection methods into functions
# todo: sort the function order a little bit. Make the docs useful.


@jax.tree_util.register_pytree_node_class
class BatchNormal(variable.StateSpaceVariable):
    """Batched normally-distributed random variables."""

    def __init__(self, mean, cov_sqrtm_lower):
        self.mean = mean  # (d, k) shape
        self.cov_sqrtm_lower = cov_sqrtm_lower  # (d, k, k) shape

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


# todo: extract the below into functions.

BatchCacheTypeVar = TypeVar("BatchCacheTypeVar")
"""Cache-type variable."""


@jax.tree_util.register_pytree_node_class
class _BatchCorrection(
    correction.AbstractCorrection[BatchNormal, BatchCacheTypeVar],
    Generic[BatchCacheTypeVar],
):
    pass


BatchMM1CacheType = Tuple[Callable]
"""Type of the correction-cache."""


@jax.tree_util.register_pytree_node_class
class BatchMomentMatching(_BatchCorrection[BatchMM1CacheType]):
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


@jax.tree_util.register_pytree_node_class
class BatchTaylorZerothOrder(_BatchCorrection[BatchTS0CacheType]):
    """TaylorZerothOrder-linearise an ODE assuming a linearisation-point with\
     isotropic Kronecker structure."""

    def begin_correction(
        self, x: BatchNormal, /, *, vector_field, t, p
    ) -> Tuple[jax.Array, float, BatchTS0CacheType]:
        m = x.mean

        # m has shape (d, n)
        bias = m[..., self.ode_order] - vector_field(
            *(m[..., : self.ode_order]).T, t=t, p=p
        )
        l_obs_nonsquare = self._cov_sqrtm_lower(
            cache=(), cov_sqrtm_lower=x.cov_sqrtm_lower
        )

        l_obs_nonsquare_1 = l_obs_nonsquare[..., None]  # (d, k, 1)

        # (d, 1, 1)
        l_obs_raw = jax.vmap(_sqrtm.sqrtm_to_upper_triangular)(R=l_obs_nonsquare_1)
        l_obs = l_obs_raw[..., 0, 0]  # (d,)

        observed = BatchNormal(mean=bias, cov_sqrtm_lower=l_obs)
        output_scale_sqrtm = observed.norm_of_whitened_residual_sqrtm()

        error_estimate = l_obs  # (d,)
        return output_scale_sqrtm * error_estimate, output_scale_sqrtm, (bias,)

    def complete_correction(
        self, *, extrapolated: BatchNormal, cache: BatchTS0CacheType
    ):
        (bias,) = cache

        # (d, k), (d, k, k)
        m_ext, l_ext = extrapolated.mean, extrapolated.cov_sqrtm_lower
        l_obs_nonsquare = self._cov_sqrtm_lower(
            cache=(), cov_sqrtm_lower=l_ext
        )  # (d, k)

        # (d, 1, 1)
        l_obs = _sqrtm.sqrtm_to_upper_triangular(R=l_obs_nonsquare[..., None])
        l_obs_scalar = l_obs[..., 0, 0]  # (d,)

        # (d,), (d,)
        observed = BatchNormal(mean=bias, cov_sqrtm_lower=l_obs_scalar)

        # (d, k)
        crosscov = (l_ext @ l_obs_nonsquare[..., None])[..., 0]

        gain = crosscov / (l_obs_scalar[..., None]) ** 2  # (d, k)

        m_cor = m_ext - (gain * bias[..., None])  # (d, k)
        l_cor = l_ext - gain[..., None] * l_obs_nonsquare[..., None, :]  # (d, k, k)

        corrected = BatchNormal(mean=m_cor, cov_sqrtm_lower=l_cor)
        return observed, (corrected, gain)

    def _cov_sqrtm_lower(self, *, cache, cov_sqrtm_lower):
        return cov_sqrtm_lower[:, self.ode_order, ...]


BatchIBMCacheType = Tuple[jax.Array]  # Cache type
"""Type of the extrapolation-cache."""


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class BatchIBM(extrapolation.AbstractExtrapolation[BatchNormal, BatchIBMCacheType]):
    """Handle block-diagonal covariances."""

    a: Any
    q_sqrtm_lower: Any

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

    def _assemble_preconditioner(self, *, dt):  # noqa: D102
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
        return BatchNormal(mean=m_ext, cov_sqrtm_lower=l_ext)

    def condense_backward_models(
        self, *, transition_init, noise_init, transition_state, noise_state
    ):  # noqa: D102

        A = transition_init  # (d, k, k)
        # (d, k), (d, k, k)
        (b, B_sqrtm) = (noise_init.mean, noise_init.cov_sqrtm_lower)

        C = transition_state  # (d, k, k)
        # (d, k), (d, k, k)
        (d, D_sqrtm) = (noise_state.mean, noise_state.cov_sqrtm_lower)

        g = A @ C
        xi = (A @ d[..., None])[..., 0] + b
        Xi_r = jax.vmap(_sqrtm.sum_of_sqrtm_factors)(
            R1=_transpose(A @ D_sqrtm), R2=_transpose(B_sqrtm)
        )
        Xi = _transpose(Xi_r)

        noise = BatchNormal(mean=xi, cov_sqrtm_lower=Xi)
        return noise, g

    def extract_mean_from_marginals(self, mean):
        return mean[..., 0]

    def extract_sol(self, *, rv):  # noqa: D102
        return self.extract_mean_from_marginals(mean=rv.mean)

    def begin_extrapolation(self, m0, /, *, dt):

        p, p_inv = self._assemble_preconditioner(dt=dt)
        m0_p = p_inv * m0  # (d, k)
        m_ext_p = (self.a @ m0_p[..., None])[..., 0]  # (d, k)
        m_ext = p * m_ext_p

        q_ext = p[..., None] * self.q_sqrtm_lower
        return BatchNormal(m_ext, q_ext), (m_ext_p, m0_p, p, p_inv)

    # todo: make into init_backward_model?
    def init_backward_noise(self, *, rv_proto):  # noqa: D102
        return BatchNormal(
            mean=jnp.zeros_like(rv_proto.mean),
            cov_sqrtm_lower=jnp.zeros_like(rv_proto.cov_sqrtm_lower),
        )

    def init_backward_transition(self):  # noqa: D102
        return jnp.stack([jnp.eye(self.num_derivatives + 1)] * self.ode_dimension)

    def init_corrected(self, *, taylor_coefficients):
        """Initialise the "corrected" RV by stacking Taylor coefficients."""
        m0_matrix = jnp.vstack(taylor_coefficients).T
        c_sqrtm0_corrected = jnp.zeros_like(self.q_sqrtm_lower)
        return BatchNormal(mean=m0_matrix, cov_sqrtm_lower=c_sqrtm0_corrected)

    # todo: move to correction?
    def init_error_estimate(self):  # noqa: D102
        return jnp.zeros((self.ode_dimension,))  # the initialisation is error-free

    # todo: move to correction?
    def init_output_scale_sqrtm(self):
        return jnp.ones((self.ode_dimension,))

    def marginalise_backwards(self, *, init, linop, noise):
        def body_fun(carry, x):
            op, noi = x
            out = self.marginalise_model(init=carry, linop=op, noise=noi)
            return out, out

        # Initial condition does not matter
        bw_models = jax.tree_util.tree_map(lambda x: x[1:, ...], (linop, noise))

        _, rvs = _control_flow.scan_with_init(
            f=body_fun, init=init, xs=bw_models, reverse=True
        )
        return rvs

    def marginalise_model(self, *, init, linop, noise):
        # todo: add preconditioner?

        # Pull into preconditioned space
        m0_p = init.mean
        l0_p = init.cov_sqrtm_lower

        # Apply transition
        m_new_p = (linop @ m0_p[..., None])[..., 0] + noise.mean
        r_new_p = jax.vmap(_sqrtm.sum_of_sqrtm_factors)(
            R1=_transpose(linop @ l0_p), R2=_transpose(noise.cov_sqrtm_lower)
        )
        l_new_p = _transpose(r_new_p)

        # Push back into non-preconditioned space
        m_new = m_new_p
        l_new = l_new_p

        return BatchNormal(mean=m_new, cov_sqrtm_lower=l_new)

    def revert_markov_kernel(  # noqa: D102
        self, *, linearisation_pt, l0, output_scale_sqrtm, cache
    ):
        m_ext_p, m0_p, p, p_inv = cache
        m_ext = linearisation_pt.mean

        # (d, k, 1) * (d, k, k) = (d, k, k)
        l0_p = p_inv[..., None] * l0

        r_ext_p, (r_bw_p, g_bw_p) = jax.vmap(_sqrtm.revert_conditional)(
            R_X_F=_transpose(self.a @ l0_p),
            R_X=_transpose(l0_p),
            # transpose((d, 1, 1) * (d, k, k)) = tranpose((d,k,k)) = (d, k, k)
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

        backward_op = g_bw
        backward_noise = BatchNormal(mean=m_bw, cov_sqrtm_lower=l_bw)
        extrapolated = BatchNormal(mean=m_ext, cov_sqrtm_lower=l_ext)
        return extrapolated, (backward_noise, backward_op)

    def sample_backwards(self, init, linop, noise, base_samples):
        def body_fun(carry, x):
            op, noi = x
            out = (op @ carry[..., None])[..., 0] + noi
            return out, out

        linop_sample, noise_ = jax.tree_util.tree_map(
            lambda x: x[1:, ...], (linop, noise)
        )
        noise_sample = self._transform_samples(noise_, base_samples[..., :-1, :, :])
        init_sample = self._transform_samples(init, base_samples[..., -1, :, :])

        # todo: should we use an associative scan here?
        _, samples = _control_flow.scan_with_init(
            f=body_fun, init=init_sample, xs=(linop_sample, noise_sample), reverse=True
        )
        return samples

    # automatically batched because of numpy's broadcasting rules?
    def _transform_samples(self, rvs, base):
        # (d,k) + ((d,k,k) @ (d,k,1))[..., 0] = (d,k)
        return rvs.mean + (rvs.cov_sqrtm_lower @ base[..., None])[..., 0]

    def scale_covariance(self, *, rv, scale_sqrtm):
        # Endpoint: (d, 1, 1) * (d, k, k) -> (d, k, k)
        if jnp.ndim(scale_sqrtm) == 1:
            return BatchNormal(
                mean=rv.mean,
                cov_sqrtm_lower=scale_sqrtm[:, None, None] * rv.cov_sqrtm_lower,
            )

        # Time series: (N, d, 1, 1) * (N, d, k, k) -> (N, d, k, k)
        return BatchNormal(
            mean=rv.mean,
            cov_sqrtm_lower=scale_sqrtm[..., None, None] * rv.cov_sqrtm_lower,
        )


def _transpose(x):
    return jnp.swapaxes(x, -1, -2)
