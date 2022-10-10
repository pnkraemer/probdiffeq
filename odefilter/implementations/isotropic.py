"""State-space models with isotropic covariance structure."""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from odefilter.implementations import _ibm, sqrtm


class IsotropicNormal(eqx.Module):
    """Random variable with a normal distribution."""

    mean: Float[Array, "n d"]
    cov_sqrtm_lower: Float[Array, "n n"]


class IsotropicImplementation(eqx.Module):
    """Handle isotropic covariances."""

    a: Any
    q_sqrtm_lower: Any

    @classmethod
    def from_num_derivatives(cls, *, num_derivatives):
        """Create a strategy from hyperparameters."""
        a, q_sqrtm = _ibm.system_matrices_1d(num_derivatives=num_derivatives)
        return cls(a=a, q_sqrtm_lower=q_sqrtm)

    @property
    def num_derivatives(self):
        """Number of derivatives in the state-space model."""  # noqa: D401
        return self.a.shape[0] - 1

    def init_corrected(self, *, taylor_coefficients):
        """Initialise the "corrected" RV by stacking Taylor coefficients."""
        m0_corrected = jnp.vstack(taylor_coefficients)
        c_sqrtm0_corrected = jnp.zeros_like(self.q_sqrtm_lower)
        return IsotropicNormal(mean=m0_corrected, cov_sqrtm_lower=c_sqrtm0_corrected)

    @staticmethod
    def init_error_estimate():  # noqa: D102
        return jnp.empty(())

    def init_backward_transition(self):  # noqa: D102
        return jnp.eye(*self.a.shape)

    def init_backward_noise(self, *, rv_proto):  # noqa: D102
        shape_m = rv_proto.mean.shape
        shape_l = rv_proto.cov_sqrtm_lower.shape
        return IsotropicNormal(
            mean=jnp.zeros(shape_m), cov_sqrtm_lower=jnp.zeros(shape_l)
        )

    def assemble_preconditioner(self, *, dt):  # noqa: D102
        return _ibm.preconditioner_diagonal(dt=dt, num_derivatives=self.num_derivatives)

    def extrapolate_mean(self, m0, /, *, p, p_inv):  # noqa: D102
        m0_p = p_inv[:, None] * m0
        m_ext_p = self.a @ m0_p
        m_ext = p[:, None] * m_ext_p
        return m_ext, m_ext_p, m0_p

    def estimate_error(self, *, linear_fn, m_obs, p):  # noqa: D102
        l_obs_raw = linear_fn(p[:, None] * self.q_sqrtm_lower)
        c_obs_raw = jnp.dot(l_obs_raw, l_obs_raw)
        res_white = m_obs / jnp.sqrt(c_obs_raw)
        diffusion_sqrtm = jnp.sqrt(jnp.dot(res_white, res_white) / res_white.size)
        error_estimate = diffusion_sqrtm * jnp.sqrt(c_obs_raw)
        return diffusion_sqrtm, error_estimate

    def complete_extrapolation(  # noqa: D102
        self, *, m_ext, l0, p_inv, p, diffusion_sqrtm
    ):
        l_ext_p = sqrtm.sum_of_sqrtm_factors(
            R1=(self.a @ (p_inv[:, None] * l0)).T,
            R2=(diffusion_sqrtm * self.q_sqrtm_lower).T,
        ).T
        l_ext = p[:, None] * l_ext_p
        return IsotropicNormal(m_ext, l_ext)

    def revert_markov_kernel(  # noqa: D102
        self, *, m_ext, l0, p, p_inv, diffusion_sqrtm, m0_p, m_ext_p
    ):
        l0_p = p_inv[:, None] * l0
        r_ext_p, (r_bw_p, g_bw_p) = sqrtm.revert_gauss_markov_correlation(
            R_X_F=(self.a @ l0_p).T,
            R_X=l0_p.T,
            R_YX=(diffusion_sqrtm * self.q_sqrtm_lower).T,
        )
        l_ext_p, l_bw_p = r_ext_p.T, r_bw_p.T
        m_bw_p = m0_p - g_bw_p @ m_ext_p

        # Un-apply the pre-conditioner
        l_ext = p[:, None] * l_ext_p
        m_bw, l_bw = p[:, None] * m_bw_p, p[:, None] * l_bw_p
        g_bw = p[:, None] * g_bw_p * p_inv[None, :]
        backward_op = g_bw
        backward_noise = IsotropicNormal(mean=m_bw, cov_sqrtm_lower=l_bw)
        extrapolated = IsotropicNormal(mean=m_ext, cov_sqrtm_lower=l_ext)
        return extrapolated, (backward_noise, backward_op)

    @staticmethod
    def final_correction(*, extrapolated, linear_fn, m_obs):  # noqa: D102
        m_ext, l_ext = extrapolated.mean, extrapolated.cov_sqrtm_lower
        l_obs = linear_fn(l_ext)  # shape (n,)
        c_obs = jnp.dot(l_obs, l_obs)
        g = (l_ext @ l_obs.T) / c_obs  # shape (n,)
        m_cor = m_ext - g[:, None] * m_obs[None, :]
        l_cor = l_ext - g[:, None] * l_obs[None, :]
        corrected = IsotropicNormal(mean=m_cor, cov_sqrtm_lower=l_cor)
        return corrected

    @staticmethod
    def extract_u(*, rv):  # noqa: D102
        return rv.mean[0]

    @staticmethod
    def condense_backward_models(*, bw_init, bw_state):  # noqa: D102
        A = bw_init.transition
        (b, B_sqrtm) = bw_init.noise.mean, bw_init.noise.cov_sqrtm_lower

        C = bw_state.transition
        (d, D_sqrtm) = bw_state.noise.mean, bw_state.noise.cov_sqrtm_lower

        g = A @ C
        xi = A @ d + b
        Xi = sqrtm.sum_of_sqrtm_factors(R1=(A @ D_sqrtm).T, R2=B_sqrtm.T).T

        noise = IsotropicNormal(mean=xi, cov_sqrtm_lower=Xi)
        return noise, g

    @staticmethod
    def marginalise_backwards(*, init, backward_model):
        """Compute marginals of a markov sequence."""

        def body_fun(carry, x):
            linop, noise = x.transition, x.noise
            out = IsotropicImplementation.marginalise_model_isotropic(
                init=carry, linop=linop, noise=noise
            )
            return out, out

        _, rvs = jax.lax.scan(f=body_fun, init=init, xs=backward_model, reverse=False)
        return rvs

    @staticmethod
    def marginalise_model_isotropic(*, init, linop, noise):
        """Marginalise the output of a linear model."""
        # Apply transition
        m_new = jnp.dot(linop, init.mean) + noise.mean
        l_new = sqrtm.sum_of_sqrtm_factors(
            R1=jnp.dot(linop, init.cov_sqrtm_lower).T, R2=noise.cov_sqrtm_lower.T
        ).T

        return IsotropicNormal(mean=m_new, cov_sqrtm_lower=l_new)
