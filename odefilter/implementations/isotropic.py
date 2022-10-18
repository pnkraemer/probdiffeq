"""State-space models with isotropic covariance structure."""

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.tree_util

from odefilter import _control_flow
from odefilter.implementations import _ibm, _interface, _sqrtm


class IsotropicNormal(NamedTuple):
    """Random variable with a normal distribution."""

    mean: Any  # (n, d) shape
    cov_sqrtm_lower: Any  # (n,n) shape


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class IsotropicImplementation(_interface.Implementation):
    """Handle isotropic covariances."""

    a: Any
    q_sqrtm_lower: Any

    def tree_flatten(self):
        children = self.a, self.q_sqrtm_lower
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        a, q_sqrtm_lower = children
        return cls(a=a, q_sqrtm_lower=q_sqrtm_lower)

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
        return jnp.zeros(())  # the initialisation is error-free

    def init_backward_transition(self):  # noqa: D102
        return jnp.eye(*self.a.shape)

    def init_backward_noise(self, *, rv_proto):  # noqa: D102
        return IsotropicNormal(
            mean=jnp.zeros_like(rv_proto.mean),
            cov_sqrtm_lower=jnp.zeros_like(rv_proto.cov_sqrtm_lower),
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

        # jnp.sqrt(l_obs.T @ l_obs) without forming the square
        l_obs = jnp.reshape(_sqrtm.sqrtm_to_cholesky(R=l_obs_raw[:, None]), ())
        res_white = (m_obs / l_obs) / jnp.sqrt(m_obs.size)

        # jnp.sqrt(\|res_white\|^2/d) without forming the square
        diffusion_sqrtm = jnp.reshape(
            _sqrtm.sqrtm_to_cholesky(R=res_white[:, None]), ()
        )

        error_estimate = diffusion_sqrtm * l_obs
        return diffusion_sqrtm, error_estimate

    def complete_extrapolation(  # noqa: D102
        self, *, m_ext, l0, p_inv, p, diffusion_sqrtm
    ):
        l0_p = p_inv[:, None] * l0
        l_ext_p = _sqrtm.sum_of_sqrtm_factors(
            R1=(self.a @ l0_p).T,
            R2=(diffusion_sqrtm * self.q_sqrtm_lower).T,
        ).T
        l_ext = p[:, None] * l_ext_p
        return IsotropicNormal(m_ext, l_ext)

    def revert_markov_kernel(  # noqa: D102
        self, *, m_ext, l0, p, p_inv, diffusion_sqrtm, m0_p, m_ext_p
    ):
        l0_p = p_inv[:, None] * l0
        r_ext_p, (r_bw_p, g_bw_p) = _sqrtm.revert_gauss_markov_correlation(
            R_X_F=(self.a @ l0_p).T,
            R_X=l0_p.T,
            R_YX=(diffusion_sqrtm * self.q_sqrtm_lower).T,
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

        backward_op = g_bw
        backward_noise = IsotropicNormal(mean=m_bw, cov_sqrtm_lower=l_bw)
        extrapolated = IsotropicNormal(mean=m_ext, cov_sqrtm_lower=l_ext)
        return extrapolated, (backward_noise, backward_op)

    @staticmethod
    def final_correction(*, extrapolated, linear_fn, m_obs):  # noqa: D102
        m_ext, l_ext = extrapolated.mean, extrapolated.cov_sqrtm_lower
        l_obs = linear_fn(l_ext)  # shape (n,)

        l_obs_scalar = jnp.reshape(_sqrtm.sqrtm_to_cholesky(R=l_obs[:, None]), ())
        c_obs = l_obs_scalar**2

        observed = IsotropicNormal(mean=m_obs, cov_sqrtm_lower=l_obs_scalar)

        g = (l_ext @ l_obs.T) / c_obs  # shape (n,)
        m_cor = m_ext - g[:, None] * m_obs[None, :]
        l_cor = l_ext - g[:, None] * l_obs[None, :]
        corrected = IsotropicNormal(mean=m_cor, cov_sqrtm_lower=l_cor)
        return observed, (corrected, g)

    @staticmethod
    def extract_sol(*, rv):  # noqa: D102
        m = rv.mean[..., 0, :]
        return m

    @staticmethod
    def condense_backward_models(*, bw_init, bw_state):  # noqa: D102

        A = bw_init.transition
        (b, B_sqrtm) = bw_init.noise.mean, bw_init.noise.cov_sqrtm_lower

        C = bw_state.transition
        (d, D_sqrtm) = (bw_state.noise.mean, bw_state.noise.cov_sqrtm_lower)

        g = A @ C
        xi = A @ d + b
        Xi = _sqrtm.sum_of_sqrtm_factors(R1=(A @ D_sqrtm).T, R2=B_sqrtm.T).T

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

        # Initial condition does not matter
        bw_models = jax.tree_util.tree_map(lambda x: x[1:, ...], backward_model)
        _, rvs = _control_flow.scan_with_init(
            f=body_fun, init=init, xs=bw_models, reverse=True
        )
        return rvs

    @staticmethod
    def marginalise_model_isotropic(*, init, linop, noise):
        """Marginalise the output of a linear model."""
        # Pull into preconditioned space
        m0_p = init.mean
        l0_p = init.cov_sqrtm_lower

        # Apply transition
        m_new_p = linop @ m0_p + noise.mean
        l_new_p = _sqrtm.sum_of_sqrtm_factors(
            R1=(linop @ l0_p).T, R2=noise.cov_sqrtm_lower.T
        ).T

        # Push back into non-preconditioned space
        m_new = m_new_p
        l_new = l_new_p

        return IsotropicNormal(mean=m_new, cov_sqrtm_lower=l_new)

    def init_preconditioner(self):  # noqa: D102
        empty = jnp.inf * jnp.ones((self.a.shape[0],))
        return empty, empty

    def evidence_sqrtm(self, *, observed):
        m_obs, l_obs = observed.mean, observed.cov_sqrtm_lower
        res_white = m_obs / l_obs
        evidence_sqrtm = jnp.sqrt(jnp.dot(res_white, res_white.T) / res_white.size)
        return evidence_sqrtm

    @staticmethod
    def scale_covariance(*, rv, scale_sqrtm):
        if jnp.ndim(scale_sqrtm) == 0:
            return IsotropicNormal(
                mean=rv.mean, cov_sqrtm_lower=scale_sqrtm * rv.cov_sqrtm_lower
            )
        return IsotropicNormal(
            mean=rv.mean,
            cov_sqrtm_lower=scale_sqrtm[:, None, None] * rv.cov_sqrtm_lower,
        )
