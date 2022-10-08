"""ODE filter backends."""
from typing import Any, Callable, Generic, List, Tuple, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import tree_util
from jaxtyping import Array, Float

from odefilter import sqrtm
from odefilter.prob import ibm, rv

# Filter/Smoother decides the type of the state and which extrapolate_cov function is called

# Isotropic/<nothing> decides the inputs/outputs of each function
# and governs most of the actual implementation
# Dynamic/<nothing> decides the order of extrapolation

"""
_Common
# implements step_fn() with extrapolate_and_linearise(),
# estimate_error(), and final_correct()
    -> _FilterCommon  # implements init_fn() with init_corrected(), returns FilteringSolution()
        -> Filter  # implements
        -> DynamicFilter

    -> _SmootherCommon  # implements init_fn() with init_corrected(), returns SmoothingSolution()
        -> Smoother  # implements extrapolate_and_linearise
        -> DynamicSmoother

Dynamic  # implements extrapolate_and_linearise() with extrapolate_mean(), linearise(), estimate_error(), and extrapolate_cov()/extrapolate_cov_and_compute_gains()
NotDynamic  # implements extrapolate_and_linearise() with extrapolate_mean() and extrapolate_cov()

class DynamicFilter
    def init_fn(tcoeffs):
    self.implementation.init_corrected(tcoeffs)

class DynamicSmoother
class Filter
class Smoother



class IsotropicImplementation:
    init_corrected
    assemble_preconditioner
    extrapolate_mean
    extrapolate_cov
    extrapolate_cov_and_compute_gains
    estimate_error
    final_correction

"""


NormalLike = TypeVar("RVLike", rv.Normal, rv.IsotropicNormal)
"""A type-variable to alias appropriate Normal-like random variables."""


class FilteringSolution(Generic[NormalLike], eqx.Module):
    """Filtering solution.

    By construction right-including, i.e. it defines the solution
    on the interval $(t_0, t_1]$.
    """

    corrected: NormalLike
    extrapolated: NormalLike


class SmoothingSolution(Generic[NormalLike], eqx.Module):
    """Filtering solution.

    By construction right-including, i.e. it defines the solution
    on the interval $(t_0, t_1]$.
    """

    filtering_solution: FilteringSolution[NormalLike]

    backward_transition: Any
    backward_noise: NormalLike


class DynamicFilter(eqx.Module):
    implementation: Any
    information: Any

    def init_fn(self, *, taylor_coefficients):
        corrected = self.implementation.init_corrected(
            taylor_coefficients=taylor_coefficients
        )
        error_estimate = self.implementation.init_error_estimate()

        extrapolated = tree_util.tree_map(jnp.empty_like, corrected)

        solution = FilteringSolution(extrapolated=extrapolated, corrected=corrected)
        return solution, error_estimate

    def step_fn(self, *, state, vector_field, dt):

        p, p_inv = self.implementation.assemble_preconditioner(dt=dt)

        # Extrapolate the mean
        m_ext, m_ext_p, m0_p = self.implementation.extrapolate_mean(
            state.corrected.mean, p=p, p_inv=p_inv
        )

        # Linearise the differential equation.
        m_obs, linear_fn = self.information(f=vector_field, x=m_ext)

        diffusion_sqrtm, error_estimate = self.implementation.estimate_error(
            linear_fn=linear_fn, m_obs=m_obs, p_inv=p_inv
        )
        error_estimate *= dt
        extrapolated = self.implementation.complete_extrapolation(
            m_ext=m_ext,
            l0=state.corrected.cov_sqrtm_lower,
            p=p,
            p_inv=p_inv,
            diffusion_sqrtm=diffusion_sqrtm,
        )

        # Final observation
        corrected, u = self.implementation.final_correction(
            extrapolated=extrapolated, linear_fn=linear_fn, m_obs=m_obs
        )
        state_new = FilteringSolution(extrapolated=extrapolated, corrected=corrected)
        return state_new, error_estimate, u


class DynamicSmoother(eqx.Module):
    implementation: Any
    information: Any

    def init_fn(self, *, taylor_coefficients):
        corrected = self.implementation.init_corrected(
            taylor_coefficients=taylor_coefficients
        )
        error_estimate = self.implementation.init_error_estimate()
        backward_transition = self.implementation.init_backward_transition()

        empty = tree_util.tree_map(jnp.empty_like, corrected)
        filtering_solution = FilteringSolution(extrapolated=empty, corrected=corrected)

        solution = SmoothingSolution(
            filtering_solution=filtering_solution,
            backward_transition=backward_transition,
            backward_noise=empty,
        )
        return solution, error_estimate

    def step_fn(self, *, state, vector_field, dt):

        p, p_inv = self.implementation.assemble_preconditioner(dt=dt)

        # Extrapolate the mean
        m_ext, m_ext_p, m0_p = self.implementation.extrapolate_mean(
            state.filtering_solution.corrected.mean, p=p, p_inv=p_inv
        )

        # Linearise the differential equation.
        m_obs, linear_fn = self.information(f=vector_field, x=m_ext)

        diffusion_sqrtm, error_estimate = self.implementation.estimate_error(
            linear_fn=linear_fn, m_obs=m_obs, p_inv=p_inv
        )
        error_estimate *= dt

        x = self.implementation.revert_markov_kernel(
            m_ext=m_ext,
            l0=state.filtering_solution.corrected.cov_sqrtm_lower,
            p=p,
            p_inv=p_inv,
            diffusion_sqrtm=diffusion_sqrtm,
            m0_p=m0_p,
            m_ext_p=m_ext_p,
        )
        extrapolated, (backward_noise, backward_op) = x

        # Final observation
        corrected, u = self.implementation.final_correction(
            extrapolated=extrapolated, linear_fn=linear_fn, m_obs=m_obs
        )
        filtering_solution = FilteringSolution(
            extrapolated=extrapolated, corrected=corrected
        )
        smoothing_solution = SmoothingSolution(
            filtering_solution=filtering_solution,
            backward_noise=backward_noise,
            backward_transition=backward_op,
        )
        return smoothing_solution, error_estimate, u


class IsotropicImplementation(eqx.Module):

    a: Any
    q_sqrtm_lower: Any

    @classmethod
    def from_num_derivatives(cls, *, num_derivatives):
        """Create a backend from hyperparameters."""
        a, q_sqrtm = ibm.system_matrices_1d(num_derivatives=num_derivatives)
        return cls(a=a, q_sqrtm_lower=q_sqrtm)

    @property
    def num_derivatives(self):
        """Number of derivatives in the state-space model."""  # noqa: D401
        return self.a.shape[0] - 1

    def init_corrected(self, *, taylor_coefficients):
        """Initialise the "corrected" RV by stacking Taylor coefficients."""
        m0_corrected = jnp.vstack(taylor_coefficients)

        # todo: remove? not necessary anymore?
        if m0_corrected.ndim == 1:
            m0_corrected = m0_corrected[:, None]

        c_sqrtm0_corrected = jnp.zeros_like(self.q_sqrtm_lower)
        return rv.IsotropicNormal(mean=m0_corrected, cov_sqrtm_lower=c_sqrtm0_corrected)

    @staticmethod
    def init_error_estimate():
        return jnp.empty(())

    def init_backward_transition(self):
        return jnp.empty_like(self.a)

    def assemble_preconditioner(self, *, dt):
        return ibm.preconditioner_diagonal(dt=dt, num_derivatives=self.num_derivatives)

    def extrapolate_mean(self, m0, /, *, p, p_inv):
        m0_p = p_inv[:, None] * m0
        m_ext_p = self.a @ m0_p
        m_ext = p[:, None] * m_ext_p
        return m_ext, m_ext_p, m0_p

    def estimate_error(self, *, linear_fn, m_obs, p_inv):
        l_obs_raw = linear_fn(p_inv[:, None] * self.q_sqrtm_lower)
        c_obs_raw = jnp.dot(l_obs_raw, l_obs_raw)
        res_white = m_obs / jnp.sqrt(c_obs_raw)
        diffusion_sqrtm = jnp.sqrt(jnp.dot(res_white, res_white) / res_white.size)
        error_estimate = diffusion_sqrtm * jnp.sqrt(c_obs_raw)
        return diffusion_sqrtm, error_estimate

    def complete_extrapolation(self, *, m_ext, l0, p_inv, p, diffusion_sqrtm):
        l_ext_p = sqrtm.sum_of_sqrtm_factors(
            R1=(self.a @ (p_inv[:, None] * l0)).T,
            R2=(diffusion_sqrtm * self.q_sqrtm_lower).T,
        ).T
        l_ext = p[:, None] * l_ext_p
        return rv.IsotropicNormal(m_ext, l_ext)

    def revert_markov_kernel(
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
        backward_noise = rv.IsotropicNormal(m_bw, l_bw)
        extrapolated = rv.IsotropicNormal(mean=m_ext, cov_sqrtm_lower=l_ext)
        return extrapolated, (backward_noise, backward_op)

    @staticmethod
    def final_correction(*, extrapolated, linear_fn, m_obs):
        m_ext, l_ext = extrapolated.mean, extrapolated.cov_sqrtm_lower
        l_obs = linear_fn(l_ext)  # shape (n,)
        c_obs = jnp.dot(l_obs, l_obs)
        g = (l_ext @ l_obs.T) / c_obs  # shape (n,)
        m_cor = m_ext - g[:, None] * m_obs[None, :]
        l_cor = l_ext - g[:, None] * l_obs[None, :]
        corrected = rv.IsotropicNormal(mean=m_cor, cov_sqrtm_lower=l_cor)
        return corrected, (corrected.mean[0])


class DenseImplementation(eqx.Module):

    a: Any
    q_sqrtm_lower: Any

    num_derivatives: int
    ode_dimension: int

    @classmethod
    def from_num_derivatives(cls, *, num_derivatives, ode_dimension):
        """Create a backend from hyperparameters."""
        a, q_sqrtm = ibm.system_matrices_1d(num_derivatives=num_derivatives)
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
        return rv.MultivariateNormal(
            mean=m0_corrected, cov_sqrtm_lower=c_sqrtm0_corrected
        )

    def init_error_estimate(self):
        return jnp.empty((self.ode_dimension,))

    def init_backward_transition(self):
        raise NotImplementedError

    def assemble_preconditioner(self, *, dt):
        p, p_inv = ibm.preconditioner_diagonal(
            dt=dt, num_derivatives=self.num_derivatives
        )
        p = jnp.tile(p, self.ode_dimension)
        p_inv = jnp.tile(p_inv, self.ode_dimension)
        return p, p_inv

    def extrapolate_mean(self, m0, /, *, p, p_inv):
        m0_p = p_inv * m0
        m_ext_p = self.a @ m0_p
        m_ext = p * m_ext_p
        return m_ext, m_ext_p, m0_p

    def estimate_error(self, *, linear_fn, m_obs, p_inv):
        l_obs_nonsquare = jax.vmap(linear_fn, in_axes=1, out_axes=1)(
            p_inv[:, None] * self.q_sqrtm_lower
        )
        l_obs_raw = sqrtm.sqrtm_to_cholesky(R=l_obs_nonsquare.T).T
        res_white = jsp.linalg.solve_triangular(l_obs_raw.T, m_obs, lower=False)
        diffusion_sqrtm = jnp.sqrt(jnp.dot(res_white, res_white.T) / res_white.size)
        error_estimate = diffusion_sqrtm * jnp.sqrt(
            jnp.einsum("nj,nj->n", l_obs_raw, l_obs_raw)
        )
        return diffusion_sqrtm, error_estimate

    def complete_extrapolation(self, *, m_ext, l0, p_inv, p, diffusion_sqrtm):
        l_ext_p = sqrtm.sum_of_sqrtm_factors(
            R1=(self.a @ (p_inv[:, None] * l0)).T,
            R2=(diffusion_sqrtm * self.q_sqrtm_lower).T,
        ).T
        l_ext = p[:, None] * l_ext_p
        return rv.MultivariateNormal(mean=m_ext, cov_sqrtm_lower=l_ext)

    def revert_markov_kernel(
        self, *, m_ext, l0, p, p_inv, diffusion_sqrtm, m0_p, m_ext_p
    ):
        raise NotImplementedError

    def final_correction(self, *, extrapolated, linear_fn, m_obs):
        m_ext, l_ext = extrapolated.mean, extrapolated.cov_sqrtm_lower
        l_obs_nonsquare = jax.vmap(linear_fn, in_axes=1, out_axes=1)(l_ext)

        l_obs = sqrtm.sqrtm_to_cholesky(R=l_obs_nonsquare.T).T
        crosscov = l_ext @ l_obs_nonsquare.T
        gain = jsp.linalg.cho_solve((l_obs, True), crosscov.T).T

        m_cor = m_ext - gain @ m_obs
        l_cor = l_ext - gain @ l_obs
        corrected = rv.MultivariateNormal(mean=m_cor, cov_sqrtm_lower=l_cor)
        u = m_cor.reshape((-1, self.ode_dimension), order="F")[0]
        return corrected, u
