"""Scalar implementations."""

import jax
import jax.numpy as jnp

from odefilter.implementations import _collections, _ibm_util, _sqrtm


@jax.tree_util.register_pytree_node_class
class ScalarNormal(_collections.StateSpaceVariable):
    # Normal RV. Shapes (), (). No QOI.

    def __init__(self, mean, cov_sqrtm_lower):
        self.mean = mean
        self.cov_sqrtm_lower = cov_sqrtm_lower

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

    @property
    def sample_shape(self):
        return self.mean.shape

    def transform_unit_sample(self, base, /):
        m, l_sqrtm = self.mean, self.cov_sqrtm_lower
        return m + l_sqrtm * base

    def Ax_plus_y(self, A, x, y):
        return A * x + y

    def condition_on_qoi_observation(self, u, /, observation_std):
        raise NotImplementedError

    def extract_qoi(self):
        raise NotImplementedError

    def extract_qoi_from_sample(self, u, /):
        raise NotImplementedError

    def scale_covariance(self, scale_sqrtm):
        return ScalarNormal(self.mean, scale_sqrtm * self.cov_sqrtm_lower)

    def logpdf(self, u, /):
        m_obs, l_obs = self.mean, self.cov_sqrtm_lower
        res_white = (m_obs - u) / l_obs
        x1 = l_obs**2
        x2 = jnp.dot(res_white, res_white.T)
        x3 = res_white.size * jnp.log(jnp.pi * 2)
        return -0.5 * (x1 + x2 + x3)

    def norm_of_whitened_residual_sqrtm(self):
        obs_pt, l_obs = self.mean, self.cov_sqrtm_lower
        res_white = obs_pt / l_obs
        evidence_sqrtm = jnp.sqrt(jnp.dot(res_white, res_white.T) / res_white.size)
        return evidence_sqrtm


@jax.tree_util.register_pytree_node_class
class Normal(_collections.StateSpaceVariable):
    # Normal RV. Shapes (n,), (n,n); zeroth state is the QOI.

    def __init__(self, mean, cov_sqrtm_lower):
        self.mean = mean
        self.cov_sqrtm_lower = cov_sqrtm_lower

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

    @property
    def sample_shape(self):
        return self.mean.shape

    def logpdf(self, u, /):
        m_obs, l_obs = self.mean, self.cov_sqrtm_lower
        res_white = jax.scipy.linalg.solve_triangular(l_obs.T, (m_obs - u), lower=False)
        x1 = jnp.linalg.slogdet(l_obs)[1] ** 2
        x2 = jnp.dot(res_white, res_white.T)
        x3 = res_white.size * jnp.log(jnp.pi * 2)
        return -0.5 * (x1 + x2 + x3)

    def norm_of_whitened_residual_sqrtm(self):
        obs_pt, l_obs = self.mean, self.cov_sqrtm_lower
        res_white = jax.scipy.linalg.solve_triangular(l_obs.T, obs_pt, lower=False)
        evidence_sqrtm = jnp.sqrt(jnp.dot(res_white, res_white.T) / res_white.size)
        return evidence_sqrtm

    def condition_on_qoi_observation(self, u, /, observation_std):
        hc = self.cov_sqrtm_lower[0]
        m_obs = self.mean[0]

        r_yx = observation_std  # * jnp.eye(1)
        r_obs_mat, (r_cor, gain_mat) = _sqrtm.revert_conditional(
            R_X=self.cov_sqrtm_lower.T, R_X_F=hc[:, None], R_YX=r_yx[None, None]
        )
        r_obs = jnp.reshape(r_obs_mat, ())
        gain = jnp.reshape(gain_mat, (-1,))

        m_cor = self.mean - gain * (m_obs - u)

        obs = ScalarNormal(m_obs, r_obs.T)
        cor = Normal(m_cor, r_cor.T)
        return obs, (cor, gain)

    def extract_qoi(self):
        return self.mean[..., 0]

    def extract_qoi_from_sample(self, u, /):

        if u.ndim == 1:
            return u[0]
        return jax.vmap(self.extract_qoi_from_sample)(u)

    def scale_covariance(self, scale_sqrtm):
        # todo: this if should not be necessary
        #  whether this function is called in batch mode or not should
        #  be the caller's concern.
        if jnp.ndim(scale_sqrtm) == 0:
            return Normal(
                mean=self.mean,
                cov_sqrtm_lower=scale_sqrtm * self.cov_sqrtm_lower,
            )
        return Normal(
            mean=self.mean,
            cov_sqrtm_lower=scale_sqrtm[:, None, None] * self.cov_sqrtm_lower,
        )

    def transform_unit_sample(self, base, /):
        m, l_sqrtm = self.mean, self.cov_sqrtm_lower
        return (m[..., None] + l_sqrtm @ base[..., None])[..., 0]

    def Ax_plus_y(self, A, x, y):
        return A @ x + y


@jax.tree_util.register_pytree_node_class
class TaylorZerothOrder(_collections.AbstractCorrection):
    def begin_correction(self, x: Normal, /, vector_field, t, p):
        m0, m1 = self.select_derivatives(x)
        fx = vector_field(*m0, t=t, p=p)
        cache, observed = self.marginalise_observation(fx, m1, x)

        output_scale_sqrtm = observed.norm_of_whitened_residual_sqrtm()
        error_estimate = observed.cov_sqrtm_lower
        return output_scale_sqrtm * error_estimate, output_scale_sqrtm, cache

    def marginalise_observation(self, fx, m1, x):
        b = m1 - fx
        cov_sqrtm_lower = x.cov_sqrtm_lower[self.ode_order, :]
        l_obs_raw = _sqrtm.sqrtm_to_upper_triangular(R=cov_sqrtm_lower[:, None])
        l_obs = jnp.reshape(l_obs_raw, ())
        observed = ScalarNormal(b, l_obs)
        cache = (b,)
        return cache, observed

    def select_derivatives(self, x):
        m0, m1 = x.mean[: self.ode_order], x.mean[self.ode_order]
        return m0, m1

    def complete_correction(self, extrapolated, cache):
        (b,) = cache
        m_ext, l_ext = extrapolated.mean, extrapolated.cov_sqrtm_lower

        l_obs_nonsquare = l_ext[self.ode_order, :]
        r_obs_mat, (r_cor, gain_mat) = _sqrtm.revert_conditional_noisefree(
            R_X_F=l_obs_nonsquare[:, None], R_X=l_ext.T
        )
        r_obs = jnp.reshape(r_obs_mat, (-1,))
        gain = jnp.reshape(gain_mat, (-1,))
        m_cor = m_ext - gain * b

        observed = ScalarNormal(mean=b, cov_sqrtm_lower=r_obs.T)
        corrected = Normal(mean=m_cor, cov_sqrtm_lower=r_cor.T)
        return observed, (corrected, gain)


@jax.tree_util.register_pytree_node_class
class Conditional(_collections.AbstractConditional):
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
        return Normal(m, self.noise.cov_sqrtm_lower)

    def scale_covariance(self, scale_sqrtm):
        noise = self.noise.scale_covariance(scale_sqrtm=scale_sqrtm)
        return Conditional(transition=self.transition, noise=noise)

    def merge_with_incoming_conditional(self, incoming, /):
        A = self.transition
        (b, B_sqrtm) = self.noise.mean, self.noise.cov_sqrtm_lower

        C = incoming.transition
        (d, D_sqrtm) = (incoming.noise.mean, incoming.noise.cov_sqrtm_lower)

        g = A @ C
        xi = A @ d + b
        Xi = _sqrtm.sum_of_sqrtm_factors(R1=(A @ D_sqrtm).T, R2=B_sqrtm.T).T

        noise = Normal(mean=xi, cov_sqrtm_lower=Xi)
        return Conditional(g, noise=noise)

    def marginalise(self, rv, /):
        m0, l0 = rv.mean, rv.cov_sqrtm_lower

        m_new = self.transition @ m0 + self.noise.mean
        l_new = _sqrtm.sum_of_sqrtm_factors(
            R1=(self.transition @ l0).T, R2=self.noise.cov_sqrtm_lower.T
        ).T

        return Normal(m_new, l_new)


@jax.tree_util.register_pytree_node_class
class IBM(_collections.AbstractExtrapolation):
    def __init__(self, a, q_sqrtm_lower):
        self.a = a
        self.q_sqrtm_lower = q_sqrtm_lower

    def __repr__(self):
        name = self.__class__.__name__
        args1 = f"a={self.a}, q={self.q_sqrtm_lower}"
        return f"{name}({args1})"

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
        """Create a strategy from hyperparameters."""
        a, q_sqrtm = _ibm_util.system_matrices_1d(num_derivatives=num_derivatives)
        return cls(a=a, q_sqrtm_lower=q_sqrtm)

    @property
    def num_derivatives(self):
        return self.a.shape[0] - 1

    def init_corrected(self, taylor_coefficients):
        """Initialise the "corrected" RV by stacking Taylor coefficients."""
        m0_matrix = jnp.vstack(taylor_coefficients)
        m0_corrected = jnp.reshape(m0_matrix, (-1,), order="F")
        c_sqrtm0_corrected = jnp.zeros_like(self.q_sqrtm_lower)
        return Normal(
            mean=m0_corrected,
            cov_sqrtm_lower=c_sqrtm0_corrected,
        )

    def init_error_estimate(self):
        return jnp.zeros(())

    def begin_extrapolation(self, m0, /, dt):
        p, p_inv = self._assemble_preconditioner(dt=dt)
        m0_p = p_inv * m0
        m_ext_p = self.a @ m0_p
        m_ext = p * m_ext_p
        q_sqrtm = p[:, None] * self.q_sqrtm_lower
        extrapolated = Normal(m_ext, q_sqrtm)
        return extrapolated, (m_ext_p, m0_p, p, p_inv)

    def _assemble_preconditioner(self, dt):
        return _ibm_util.preconditioner_diagonal(
            dt=dt, num_derivatives=self.num_derivatives
        )

    def complete_extrapolation(self, linearisation_pt, cache, l0, output_scale_sqrtm):
        _, _, p, p_inv = cache
        m_ext = linearisation_pt.mean
        l_ext_p = _sqrtm.sum_of_sqrtm_factors(
            R1=(self.a @ (p_inv[:, None] * l0)).T,
            R2=(output_scale_sqrtm * self.q_sqrtm_lower).T,
        ).T
        l_ext = p[:, None] * l_ext_p
        return Normal(mean=m_ext, cov_sqrtm_lower=l_ext)

    def revert_markov_kernel(self, linearisation_pt, cache, l0, output_scale_sqrtm):
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
        m_bw = p * m_bw_p
        l_bw = p[:, None] * l_bw_p
        g_bw = p[:, None] * g_bw_p * p_inv[None, :]

        backward_noise = Normal(mean=m_bw, cov_sqrtm_lower=l_bw)
        bw_model = Conditional(g_bw, noise=backward_noise)
        extrapolated = Normal(mean=m_ext, cov_sqrtm_lower=l_ext)
        return extrapolated, bw_model

    def init_conditional(self, rv_proto):
        op = self._init_backward_transition()
        noi = self._init_backward_noise(rv_proto=rv_proto)
        return Conditional(op, noise=noi)

    def _init_backward_transition(self):
        k = self.num_derivatives + 1
        return jnp.eye(k)

    @staticmethod
    def _init_backward_noise(rv_proto):
        mean = jnp.zeros_like(rv_proto.mean)
        cov_sqrtm_lower = jnp.zeros_like(rv_proto.cov_sqrtm_lower)
        return Normal(mean, cov_sqrtm_lower)

    def init_output_scale_sqrtm(self):
        return 1.0
