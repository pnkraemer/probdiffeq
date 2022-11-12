"""State-space models with isotropic covariance structure."""

import jax
import jax.numpy as jnp

from odefilter.implementations import _collections, _ibm_util, _sqrtm


@jax.tree_util.register_pytree_node_class
class IsoNormal(_collections.StateSpaceVariable):
    def __init__(self, mean, cov_sqrtm_lower):
        self.mean = mean  # (n, d) shape
        self.cov_sqrtm_lower = cov_sqrtm_lower  # (n, n) shape

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

        res_white = (m_obs - u) / jnp.reshape(l_obs, ())

        x1 = jnp.dot(res_white, res_white.T)
        x2 = jnp.reshape(l_obs, ()) ** 2
        x3 = res_white.size * jnp.log(jnp.pi * 2)
        return -0.5 * (x1 + x2 + x3)

    def norm_of_whitened_residual_sqrtm(self):
        obs_pt, l_obs = self.mean, self.cov_sqrtm_lower
        res_white = obs_pt / l_obs
        evidence_sqrtm = jnp.sqrt(jnp.dot(res_white, res_white.T) / res_white.size)
        return evidence_sqrtm

    def condition_on_qoi_observation(self, u, /, observation_std):
        hc = self.cov_sqrtm_lower[0, ...].reshape((1, -1))
        m_obs = self.mean[0, ...]

        r_yx = observation_std * jnp.ones((1, 1))
        r_obs, (r_cor, gain) = _sqrtm.revert_conditional(
            R_X_F=hc.T, R_X=self.cov_sqrtm_lower.T, R_YX=r_yx
        )
        m_cor = self.mean - gain * (m_obs - u)[None, :]

        return IsoNormal(m_obs, r_obs.T), (IsoNormal(m_cor, r_cor.T), gain)

    def extract_qoi(self):
        m = self.mean[..., 0, :]
        return m

    def extract_qoi_from_sample(self, u, /):
        return u[..., 0, :]

    # todo: split those functions into a batch and a non-batch version?

    def scale_covariance(self, scale_sqrtm):
        if jnp.ndim(scale_sqrtm) == 0:
            return IsoNormal(
                mean=self.mean, cov_sqrtm_lower=scale_sqrtm * self.cov_sqrtm_lower
            )
        return IsoNormal(
            mean=self.mean,
            cov_sqrtm_lower=scale_sqrtm[:, None, None] * self.cov_sqrtm_lower,
        )

    def transform_unit_sample(self, base, /):
        return self.mean + self.cov_sqrtm_lower @ base

    def Ax_plus_y(self, A, x, y):
        return A @ x + y

    @property
    def sample_shape(self):
        return self.mean.shape


@jax.tree_util.register_pytree_node_class
class IsoTaylorZerothOrder(_collections.AbstractCorrection):
    def begin_correction(self, x: IsoNormal, /, vector_field, t, p):
        m = x.mean
        m0, m1 = m[: self.ode_order, ...], m[self.ode_order, ...]
        bias = m1 - vector_field(*m0, t=t, p=p)
        cov_sqrtm_lower = x.cov_sqrtm_lower[self.ode_order, ...]

        l_obs = jnp.reshape(
            _sqrtm.sqrtm_to_upper_triangular(R=cov_sqrtm_lower[:, None]), ()
        )
        res_white = (bias / l_obs) / jnp.sqrt(bias.size)

        # jnp.sqrt(\|res_white\|^2/d) without forming the square
        output_scale_sqrtm = jnp.reshape(
            _sqrtm.sqrtm_to_upper_triangular(R=res_white[:, None]), ()
        )

        error_estimate = l_obs
        return output_scale_sqrtm * error_estimate, output_scale_sqrtm, (bias,)

    def complete_correction(self, extrapolated, cache):
        (bias,) = cache

        m_ext, l_ext = extrapolated.mean, extrapolated.cov_sqrtm_lower
        l_obs = l_ext[self.ode_order, ...]

        l_obs_scalar = jnp.reshape(
            _sqrtm.sqrtm_to_upper_triangular(R=l_obs[:, None]), ()
        )
        c_obs = l_obs_scalar**2

        observed = IsoNormal(mean=bias, cov_sqrtm_lower=l_obs_scalar)

        g = (l_ext @ l_obs.T) / c_obs  # shape (n,)
        m_cor = m_ext - g[:, None] * bias[None, :]
        l_cor = l_ext - g[:, None] * l_obs[None, :]
        corrected = IsoNormal(mean=m_cor, cov_sqrtm_lower=l_cor)
        return observed, (corrected, g)


@jax.tree_util.register_pytree_node_class
class IsoConditional(_collections.AbstractConditional):
    def __init__(self, transition, noise):
        self.transition = transition
        self.noise = noise

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(transition={self.transition}, noise={self.noise})"

    def tree_flatten(self):
        children = self.transition, self.noise
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        transition, noise = children
        return cls(transition=transition, noise=noise)

    def __call__(self, x, /):
        m = self.transition @ x + self.noise.mean
        return IsoNormal(m, self.noise.cov_sqrtm_lower)

    def scale_covariance(self, scale_sqrtm):
        noise = self.noise.scale_covariance(scale_sqrtm=scale_sqrtm)
        return IsoConditional(transition=self.transition, noise=noise)

    def merge_with_incoming_conditional(self, incoming, /):
        A = self.transition
        (b, B_sqrtm) = self.noise.mean, self.noise.cov_sqrtm_lower

        C = incoming.transition
        (d, D_sqrtm) = (incoming.noise.mean, incoming.noise.cov_sqrtm_lower)

        g = A @ C
        xi = A @ d + b
        Xi = _sqrtm.sum_of_sqrtm_factors(R1=(A @ D_sqrtm).T, R2=B_sqrtm.T).T

        noise = IsoNormal(mean=xi, cov_sqrtm_lower=Xi)
        bw_model = IsoConditional(g, noise=noise)
        return bw_model

    def marginalise(self, rv, /):
        """Marginalise the output of a linear model."""
        # Read
        m0_p = rv.mean
        l0_p = rv.cov_sqrtm_lower

        # Apply transition
        m_new = self.transition @ m0_p + self.noise.mean
        l_new = _sqrtm.sum_of_sqrtm_factors(
            R1=(self.transition @ l0_p).T, R2=self.noise.cov_sqrtm_lower.T
        ).T

        return IsoNormal(mean=m_new, cov_sqrtm_lower=l_new)


@jax.tree_util.register_pytree_node_class
class IsoIBM(_collections.AbstractExtrapolation):
    def __init__(self, a, q_sqrtm_lower):
        self.a = a
        self.q_sqrtm_lower = q_sqrtm_lower

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(a={self.a}, q_sqrtm_lower={self.q_sqrtm_lower})"

    def tree_flatten(self):
        children = self.a, self.q_sqrtm_lower
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        a, q_sqrtm_lower = children
        return cls(a=a, q_sqrtm_lower=q_sqrtm_lower)

    @classmethod
    def from_params(cls, num_derivatives):
        a, q_sqrtm = _ibm_util.system_matrices_1d(num_derivatives=num_derivatives)
        return cls(a=a, q_sqrtm_lower=q_sqrtm)

    @property
    def num_derivatives(self):
        return self.a.shape[0] - 1

    def init_corrected(self, taylor_coefficients):
        m0_corrected = jnp.vstack(taylor_coefficients)
        c_sqrtm0_corrected = jnp.zeros_like(self.q_sqrtm_lower)
        return IsoNormal(mean=m0_corrected, cov_sqrtm_lower=c_sqrtm0_corrected)

    def init_rv(self, ode_dimension):
        m0 = jnp.zeros((self.num_derivatives + 1, ode_dimension))
        c0 = jnp.eye(self.num_derivatives + 1)
        return IsoNormal(m0, c0)

    def init_error_estimate(self):
        return jnp.zeros(())  # the initialisation is error-free

    def init_output_scale_sqrtm(self):
        return 1.0

    def begin_extrapolation(self, m0, /, dt):
        p, p_inv = self._assemble_preconditioner(dt=dt)
        m0_p = p_inv[:, None] * m0
        m_ext_p = self.a @ m0_p
        m_ext = p[:, None] * m_ext_p
        q_sqrtm = p[:, None] * self.q_sqrtm_lower
        return IsoNormal(m_ext, q_sqrtm), (m_ext_p, m0_p, p, p_inv)

    def _assemble_preconditioner(self, dt):
        return _ibm_util.preconditioner_diagonal(
            dt=dt, num_derivatives=self.num_derivatives
        )

    def complete_extrapolation(self, linearisation_pt, l0, cache, output_scale_sqrtm):
        _, _, p, p_inv = cache
        m_ext = linearisation_pt.mean

        l0_p = p_inv[:, None] * l0
        l_ext_p = _sqrtm.sum_of_sqrtm_factors(
            R1=(self.a @ l0_p).T,
            R2=(output_scale_sqrtm * self.q_sqrtm_lower).T,
        ).T
        l_ext = p[:, None] * l_ext_p
        return IsoNormal(m_ext, l_ext)

    def revert_markov_kernel(self, linearisation_pt, l0, cache, output_scale_sqrtm):
        m_ext_p, m0_p, p, p_inv = cache
        m_ext = linearisation_pt.mean

        l0_p = p_inv[:, None] * l0
        r_ext_p, (r_bw_p, g_bw_p) = _sqrtm.revert_conditional(
            R_X_F=(self.a @ l0_p).T,
            R_X=l0_p.T,
            R_YX=(output_scale_sqrtm * self.q_sqrtm_lower).T,
        )
        l_ext_p, l_bw_p = r_ext_p.T, r_bw_p.T
        m_bw_p = m0_p - g_bw_p @ m_ext_p

        # Un-apply the pre-conditioner.
        # The backward models remains preconditioned, because
        # we do backward passes in preconditioner-space.
        l_ext = p[:, None] * l_ext_p
        m_bw = p[:, None] * m_bw_p
        l_bw = p[:, None] * l_bw_p
        g_bw = p[:, None] * g_bw_p * p_inv[None, :]

        backward_noise = IsoNormal(mean=m_bw, cov_sqrtm_lower=l_bw)
        bw_model = IsoConditional(g_bw, noise=backward_noise)
        extrapolated = IsoNormal(mean=m_ext, cov_sqrtm_lower=l_ext)
        return extrapolated, bw_model

    # todo: should this be a classmethod in IsoConditional?
    def init_conditional(self, rv_proto):
        op = self._init_backward_transition()
        noi = self._init_backward_noise(rv_proto=rv_proto)
        return IsoConditional(op, noise=noi)

    def _init_backward_transition(self):
        return jnp.eye(*self.a.shape)

    def _init_backward_noise(self, rv_proto):
        return IsoNormal(
            mean=jnp.zeros_like(rv_proto.mean),
            cov_sqrtm_lower=jnp.zeros_like(rv_proto.cov_sqrtm_lower),
        )
