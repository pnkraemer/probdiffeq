"""State-space models with dense covariance structure   ."""

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.tree_util import register_pytree_node_class

from odefilter.implementations import _ibm, _interface, _sqrtm


class MultivariateNormal(NamedTuple):
    """Random variable with a normal distribution."""

    mean: Any  # (k,) shape
    cov_sqrtm_lower: Any  # (k,k) shape


@register_pytree_node_class
@dataclass(frozen=True)
class DenseImplementation(_interface.Implementation):
    """Handle dense covariances."""

    a: Any
    q_sqrtm_lower: Any

    num_derivatives: int
    ode_dimension: int

    def tree_flatten(self):
        children = self.a, self.q_sqrtm_lower
        aux = self.num_derivatives, self.ode_dimension
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        a, q_sqrtm_lower = children
        n, d = aux
        return cls(a=a, q_sqrtm_lower=q_sqrtm_lower, num_derivatives=n, ode_dimension=d)

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
        if taylor_coefficients[0].shape[0] != self.ode_dimension:
            msg = "The solver's ODE dimension does not match the initial condition."
            raise ValueError(msg)

        m0_matrix = jnp.vstack(taylor_coefficients)
        m0_corrected = jnp.reshape(m0_matrix, (-1,), order="F")
        c_sqrtm0_corrected = jnp.zeros_like(self.q_sqrtm_lower)
        return MultivariateNormal(mean=m0_corrected, cov_sqrtm_lower=c_sqrtm0_corrected)

    def init_error_estimate(self):  # noqa: D102
        return jnp.zeros((self.ode_dimension,))  # the initialisation is error-free

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
        linear_fn_vmap = jax.vmap(linear_fn, in_axes=1, out_axes=1)
        l_obs_nonsquare = linear_fn_vmap(p[:, None] * self.q_sqrtm_lower)
        l_obs_raw = _sqrtm.sqrtm_to_cholesky(R=l_obs_nonsquare.T).T

        # todo: make this use evidence()
        res_white = jsp.linalg.solve_triangular(l_obs_raw.T, m_obs, lower=False)
        diffusion_sqrtm = jnp.sqrt(jnp.dot(res_white, res_white.T) / res_white.size)

        error_estimate = diffusion_sqrtm * jnp.sqrt(
            jnp.einsum("nj,nj->n", l_obs_raw, l_obs_raw)
        )
        return diffusion_sqrtm, error_estimate

    def complete_extrapolation(  # noqa: D102
        self, *, m_ext, l0, p_inv, p, diffusion_sqrtm
    ):
        l_ext_p = _sqrtm.sum_of_sqrtm_factors(
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

        l_obs = _sqrtm.sqrtm_to_cholesky(R=l_obs_nonsquare.T).T
        observed = MultivariateNormal(mean=m_obs, cov_sqrtm_lower=l_obs)

        crosscov = l_ext @ l_obs_nonsquare.T
        gain = jsp.linalg.cho_solve((l_obs, True), crosscov.T).T

        m_cor = m_ext - gain @ m_obs
        l_cor = l_ext - gain @ l_obs_nonsquare
        corrected = MultivariateNormal(mean=m_cor, cov_sqrtm_lower=l_cor)
        return observed, (corrected, gain)

    @staticmethod
    def evidence_sqrtm(*, observed):
        m_obs, l_obs = observed.mean, observed.cov_sqrtm_lower
        res_white = jsp.linalg.solve_triangular(l_obs.T, m_obs, lower=False)
        evidence_sqrtm = jnp.sqrt(jnp.dot(res_white, res_white.T) / res_white.size)
        return evidence_sqrtm

    def extract_sol(self, *, rv):  # noqa: D102
        def proj(x):
            return jnp.reshape(x, (-1, self.ode_dimension), order="F")[0]

        m = proj(rv.mean)
        return m

    def init_preconditioner(self):  # noqa: D102
        empty = jnp.inf * jnp.ones((self.num_derivatives * self.ode_dimension,))
        return empty, empty

    def init_backward_transition(self):  # noqa: D102
        k = self.num_derivatives * self.ode_dimension
        return jnp.eye(k)

    def init_backward_noise(self, rv_proto):  # noqa: D102
        return MultivariateNormal(
            mean=jnp.zeros_like(rv_proto.mean),
            cov_sqrtm_lower=jnp.zeros_like(rv_proto.cov_sqrtm_lower),
        )

    @staticmethod
    def scale_covariance(*, rv, scale_sqrtm):
        if jnp.ndim(scale_sqrtm) == 0:
            return MultivariateNormal(
                mean=rv.mean, cov_sqrtm_lower=scale_sqrtm * rv.cov_sqrtm_lower
            )
        return MultivariateNormal(
            mean=rv.mean,
            cov_sqrtm_lower=scale_sqrtm[:, None, None] * rv.cov_sqrtm_lower,
        )
