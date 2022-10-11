"""State-space models with dense covariance structure   ."""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, Float

from odefilter.implementations import _ibm, sqrtm


class MultivariateNormal(eqx.Module):
    """Random variable with a normal distribution."""

    mean: Float[Array, " k"]
    cov_sqrtm_lower: Float[Array, "k k"]


class DenseImplementation(eqx.Module):
    """Handle dense covariances."""

    a: Any
    q_sqrtm_lower: Any

    num_derivatives: int
    ode_dimension: int

    @classmethod
    def from_num_derivatives(cls, *, num_derivatives, ode_dimension):
        """Create a strategy from hyperparameters."""
        a, q_sqrtm = _ibm.system_matrices_1d(num_derivatives=num_derivatives)
        eye_d = jnp.eye(ode_dimension)
        return cls(
            a=jnp.kron(eye_d, a),
            q_sqrtm_lower=jnp.kron(eye_d, q_sqrtm),
            num_derivatives=num_derivatives,
            ode_dimension=ode_dimension,
        )

    def init_corrected(self, *, taylor_coefficients):
        """Initialise the "corrected" RV by stacking Taylor coefficients."""
        m0_matrix = jnp.vstack(taylor_coefficients)
        m0_corrected = jnp.reshape(m0_matrix, (-1,), order="F")
        c_sqrtm0_corrected = jnp.zeros_like(self.q_sqrtm_lower)
        return MultivariateNormal(mean=m0_corrected, cov_sqrtm_lower=c_sqrtm0_corrected)

    def init_error_estimate(self):  # noqa: D102
        return jnp.nan * jnp.ones((self.ode_dimension,))

    def init_backward_transition(self):  # noqa: D102
        raise NotImplementedError

    def assemble_preconditioner(self, *, dt):  # noqa: D102
        p, p_inv = _ibm.preconditioner_diagonal(
            dt=dt, num_derivatives=self.num_derivatives
        )
        p = jnp.tile(p, self.ode_dimension)
        p_inv = jnp.tile(p_inv, self.ode_dimension)
        return p, p_inv

    def extrapolate_mean(self, m0, /, *, p, p_inv):  # noqa: D102
        m0_p = p_inv * m0
        m_ext_p = self.a @ m0_p
        m_ext = p * m_ext_p
        return m_ext, m_ext_p, m0_p

    def estimate_error(self, *, linear_fn, m_obs, p):  # noqa: D102
        l_obs_nonsquare = jax.vmap(linear_fn, in_axes=1, out_axes=1)(
            p[:, None] * self.q_sqrtm_lower
        )
        l_obs_raw = sqrtm.sqrtm_to_cholesky(R=l_obs_nonsquare.T).T
        res_white = jsp.linalg.solve_triangular(l_obs_raw.T, m_obs, lower=False)
        diffusion_sqrtm = jnp.sqrt(jnp.dot(res_white, res_white.T) / res_white.size)
        error_estimate = diffusion_sqrtm * jnp.sqrt(
            jnp.einsum("nj,nj->n", l_obs_raw, l_obs_raw)
        )
        return diffusion_sqrtm, error_estimate

    def complete_extrapolation(  # noqa: D102
        self, *, m_ext, l0, p_inv, p, diffusion_sqrtm
    ):
        l_ext_p = sqrtm.sum_of_sqrtm_factors(
            R1=(self.a @ (p_inv[:, None] * l0)).T,
            R2=(diffusion_sqrtm * self.q_sqrtm_lower).T,
        ).T
        l_ext = p[:, None] * l_ext_p
        return MultivariateNormal(mean=m_ext, cov_sqrtm_lower=l_ext)

    def revert_markov_kernel(  # noqa: D102
        self, *, m_ext, l0, p, p_inv, diffusion_sqrtm, m0_p, m_ext_p
    ):
        raise NotImplementedError

    def final_correction(self, *, extrapolated, linear_fn, m_obs):  # noqa: D102
        m_ext, l_ext = extrapolated.mean, extrapolated.cov_sqrtm_lower
        l_obs_nonsquare = jax.vmap(linear_fn, in_axes=1, out_axes=1)(l_ext)

        l_obs = sqrtm.sqrtm_to_cholesky(R=l_obs_nonsquare.T).T
        crosscov = l_ext @ l_obs_nonsquare.T
        gain = jsp.linalg.cho_solve((l_obs, True), crosscov.T).T

        m_cor = m_ext - gain @ m_obs
        l_cor = l_ext - gain @ l_obs_nonsquare
        corrected = MultivariateNormal(mean=m_cor, cov_sqrtm_lower=l_cor)
        return corrected

    def extract_u(self, *, rv):  # noqa: D102
        return rv.mean.reshape((-1, self.ode_dimension), order="F")[0]
