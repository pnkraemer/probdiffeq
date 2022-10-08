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


class _IsotropicCommon(eqx.Module):

    a: Any
    q_sqrtm_lower: Any

    information: Any

    @classmethod
    def from_num_derivatives(cls, *, num_derivatives, information):
        """Create a backend from hyperparameters."""
        a, q_sqrtm = ibm.system_matrices_1d(num_derivatives=num_derivatives)
        return cls(a=a, q_sqrtm_lower=q_sqrtm, information=information)

    @property
    def num_derivatives(self):
        """Number of derivatives in the state-space model."""  # noqa: D401
        return self.a.shape[0] - 1

    def _init_corrected(self, *, taylor_coefficients):
        """Initialise the "corrected" RV by stacking Taylor coefficients."""
        m0_corrected = jnp.vstack(taylor_coefficients)

        # todo: remove? not necessary anymore?
        if m0_corrected.ndim == 1:
            m0_corrected = m0_corrected[:, None]

        c_sqrtm0_corrected = jnp.zeros_like(self.q_sqrtm_lower)
        return rv.IsotropicNormal(mean=m0_corrected, cov_sqrtm_lower=c_sqrtm0_corrected)

    def _estimate_error(self, *, linear_fn, m_obs, p_inv):
        l_obs_raw = linear_fn(p_inv[:, None] * self.q_sqrtm_lower)
        c_obs_raw = jnp.dot(l_obs_raw, l_obs_raw)
        res_white = m_obs / jnp.sqrt(c_obs_raw)
        diffusion_sqrtm = jnp.sqrt(jnp.dot(res_white, res_white) / res_white.size)
        error_estimate = diffusion_sqrtm * jnp.sqrt(c_obs_raw)
        return diffusion_sqrtm, error_estimate

    @staticmethod
    def _final_correction(*, m_ext, l_ext, linear_fn, m_obs):
        l_obs = linear_fn(l_ext)  # shape (n,)
        c_obs = jnp.dot(l_obs, l_obs)
        g = (l_ext @ l_obs.T) / c_obs  # shape (n,)
        m_cor = m_ext - g[:, None] * m_obs[None, :]
        l_cor = l_ext - g[:, None] * l_obs[None, :]
        corrected = rv.IsotropicNormal(mean=m_cor, cov_sqrtm_lower=l_cor)
        return corrected


class DynamicIsotropicFilter(_IsotropicCommon):
    """Terminal-value simulation with an isotropic covariance \
     structure and dynamic (think: time-varying) calibration."""

    def init_fn(
        self, *, taylor_coefficients: List[Float[Array, " d"]]
    ) -> FilteringSolution[rv.IsotropicNormal]:
        """Initialise."""
        # Infer the "corrected" random variable from the Taylor coefficients.
        # (There is no actual correction, because we have perfect information.)
        corrected = self._init_corrected(taylor_coefficients=taylor_coefficients)

        # Invent an "extrapolated" random variable.
        # It is required to have type- and shape-stability in the loop.
        # Its values are irrelevant.
        extrapolated = tree_util.tree_map(jnp.empty_like, corrected)

        return FilteringSolution(
            extrapolated=extrapolated, corrected=corrected
        ), jnp.empty(())

    def step_fn(
        self,
        *,
        state: FilteringSolution[rv.IsotropicNormal],
        vector_field: Callable[..., Float[Array, " d"]],
        dt: float,
    ) -> Tuple[
        FilteringSolution[rv.IsotropicNormal], Float[Array, " d"], Float[Array, " d"]
    ]:
        """Step."""
        # A note on variable naming:
        #  "m" and "l" refer to "mean" and "lower square root matrix".
        #  "c" is the covariance
        #  "0" is the initial state
        #  "_ext" is the extrapolation
        #  "_obs" is the observation
        #  "_obs_raw" is the observation assuming an error-free previous state
        #  "_cor" is the correction
        #  "p" and "p_inv" are preconditioners
        #  "_p" indicates that a variable lives in "preconditioned space"
        #

        # Compute preconditioner
        p, p_inv = ibm.preconditioner_diagonal(
            dt=dt, num_derivatives=self.num_derivatives
        )

        # Extrapolate the mean and linearise the differential equation.
        (m0, l0) = state.corrected.mean, state.corrected.cov_sqrtm_lower
        m_ext = p[:, None] * (self.a @ (p_inv[:, None] * m0))
        m_obs, linear_fn = self.information(f=vector_field, x=m_ext)

        # Observe the error-free state and calibrate some parameters
        diffusion_sqrtm, error_estimate = self._estimate_error(
            linear_fn=linear_fn, m_obs=m_obs, p_inv=p_inv
        )
        error_estimate *= dt

        # Full extrapolation
        l_ext_p = sqrtm.sum_of_sqrtm_factors(
            R1=(self.a @ (p_inv[:, None] * l0)).T,
            R2=(diffusion_sqrtm * self.q_sqrtm_lower).T,
        ).T
        l_ext = p[:, None] * l_ext_p
        extrapolated = rv.IsotropicNormal(mean=m_ext, cov_sqrtm_lower=l_ext)

        # Final observation
        corrected = self._final_correction(
            m_ext=m_ext, l_ext=l_ext, linear_fn=linear_fn, m_obs=m_obs
        )
        state_new = FilteringSolution(extrapolated=extrapolated, corrected=corrected)
        return state_new, error_estimate, (corrected.mean[0])


class DynamicIsotropicSmoother(_IsotropicCommon):
    """Simulation with an isotropic covariance \
     structure and dynamic (think: time-varying) calibration."""

    def init_fn(
        self, *, taylor_coefficients: List[Float[Array, " d"]]
    ) -> SmoothingSolution[rv.IsotropicNormal]:
        """Initialise."""
        corrected = self._init_corrected(taylor_coefficients=taylor_coefficients)

        empty = tree_util.tree_map(jnp.empty_like, corrected)
        filtering_solution = FilteringSolution(extrapolated=empty, corrected=corrected)
        return SmoothingSolution(
            filtering_solution=filtering_solution,
            backward_transition=jnp.empty_like(self.a),
            backward_noise=empty,
        ), jnp.empty(())

    def step_fn(
        self,
        *,
        state: SmoothingSolution[rv.IsotropicNormal],
        vector_field: Callable[..., Float[Array, " d"]],
        dt: float,
    ) -> Tuple[
        SmoothingSolution[rv.IsotropicNormal], Float[Array, " d"], Float[Array, " d"]
    ]:
        """Step."""
        # A note on variable naming:
        #  "m" and "l" refer to "mean" and "lower square root matrix".
        #  "c" is the covariance
        #  "0" is the initial state
        #  "_ext" is the extrapolation
        #  "_obs" is the observation
        #  "_obs_raw" is the observation assuming an error-free previous state
        #  "_cor" is the correction
        #  "p" and "p_inv" are preconditioners
        #  "_p" indicates that a variable lives in "preconditioned space"

        # Compute preconditioner
        p, p_inv = ibm.preconditioner_diagonal(
            dt=dt, num_derivatives=self.num_derivatives
        )
        # Extract previous correction
        m0 = state.filtering_solution.corrected.mean
        l0 = state.filtering_solution.corrected.cov_sqrtm_lower

        # Extrapolate the mean and linearise the differential equation.
        m0_p = p_inv[:, None] * m0
        m_ext_p = self.a @ m0_p
        m_ext = p[:, None] * m_ext_p
        m_obs, linear_fn = self.information(f=vector_field, x=m_ext)

        # Observe the error-free state and calibrate some parameters
        diffusion_sqrtm, error_estimate = self._estimate_error(
            linear_fn=linear_fn, m_obs=m_obs, p_inv=p_inv
        )
        error_estimate *= dt

        # Full extrapolation
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

        # Final correction
        corrected = self._final_correction(
            m_ext=m_ext, l_ext=l_ext, linear_fn=linear_fn, m_obs=m_obs
        )
        filtering_solution = FilteringSolution(
            extrapolated=extrapolated, corrected=corrected
        )
        state_new = SmoothingSolution(
            filtering_solution=filtering_solution,
            backward_transition=backward_op,
            backward_noise=backward_noise,
        )
        return state_new, error_estimate, corrected.mean[0]


class _Common(eqx.Module):

    a: Any
    q_sqrtm_lower: Any

    information: Any
    num_derivatives: int
    ode_dimension: int

    @classmethod
    def from_num_derivatives(cls, *, num_derivatives, ode_dimension, information):
        """Create a backend from hyperparameters."""
        a, q_sqrtm = ibm.system_matrices_1d(num_derivatives=num_derivatives)
        eye_d = jnp.eye(ode_dimension)

        return cls(
            a=jnp.kron(eye_d, a),
            q_sqrtm_lower=jnp.kron(eye_d, q_sqrtm),
            information=information,
            num_derivatives=num_derivatives,
            ode_dimension=ode_dimension,
        )

    def _init_corrected(self, *, taylor_coefficients):
        """Initialise the "corrected" RV by stacking Taylor coefficients."""
        m0_matrix = jnp.vstack(taylor_coefficients)
        m0_corrected = jnp.reshape(m0_matrix, (-1,), order="F")
        c_sqrtm0_corrected = jnp.zeros_like(self.q_sqrtm_lower)
        return rv.MultivariateNormal(
            mean=m0_corrected, cov_sqrtm_lower=c_sqrtm0_corrected
        )

    def _estimate_error(self, *, linear_fn, m_obs, p_inv):
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

    @staticmethod
    def _final_correction(*, m_ext, l_ext, linear_fn, m_obs):
        l_obs_nonsquare = jax.vmap(linear_fn, in_axes=1, out_axes=1)(l_ext)

        l_obs = sqrtm.sqrtm_to_cholesky(R=l_obs_nonsquare.T).T
        crosscov = l_ext @ l_obs_nonsquare.T
        gain = jsp.linalg.cho_solve((l_obs, True), crosscov.T).T

        m_cor = m_ext - gain @ m_obs
        l_cor = l_ext - gain @ l_obs
        corrected = rv.MultivariateNormal(mean=m_cor, cov_sqrtm_lower=l_cor)
        return corrected


class DynamicFilter(_Common):
    """Terminal-value simulation with an isotropic covariance \
     structure and dynamic (think: time-varying) calibration."""

    def init_fn(
        self, *, taylor_coefficients: List[Float[Array, " d"]]
    ) -> FilteringSolution[rv.MultivariateNormal]:
        """Initialise."""
        # Infer the "corrected" random variable from the Taylor coefficients.
        # (There is no actual correction, because we have perfect information.)
        corrected = self._init_corrected(taylor_coefficients=taylor_coefficients)

        # Invent an "extrapolated" random variable.
        # It is required to have type- and shape-stability in the loop.
        # Its values are irrelevant.
        extrapolated = tree_util.tree_map(jnp.empty_like, corrected)

        filtering_solution = FilteringSolution(
            extrapolated=extrapolated, corrected=corrected
        )
        error_estimate = jnp.empty_like(taylor_coefficients[0])
        return filtering_solution, error_estimate

    def step_fn(
        self,
        *,
        state: FilteringSolution[rv.MultivariateNormal],
        vector_field: Callable[..., Float[Array, " d"]],
        dt: float,
    ) -> Tuple[
        FilteringSolution[rv.MultivariateNormal], Float[Array, " d"], Float[Array, " d"]
    ]:
        """Step."""
        # A note on variable naming:
        #  "m" and "l" refer to "mean" and "lower square root matrix".
        #  "c" is the covariance
        #  "0" is the initial state
        #  "_ext" is the extrapolation
        #  "_obs" is the observation
        #  "_obs_raw" is the observation assuming an error-free previous state
        #  "_cor" is the correction
        #  "p" and "p_inv" are preconditioners
        #  "_p" indicates that a variable lives in "preconditioned space"
        #

        # Compute preconditioner
        p, p_inv = ibm.preconditioner_diagonal(
            dt=dt, num_derivatives=self.num_derivatives
        )
        p = jnp.tile(p, self.ode_dimension)
        p_inv = jnp.tile(p_inv, self.ode_dimension)

        # Extrapolate the mean and linearise the differential equation.
        (m0, l0) = state.corrected.mean, state.corrected.cov_sqrtm_lower
        m_ext = p * (self.a @ (p_inv * m0))
        m_obs, linear_fn = self.information(f=vector_field, x=m_ext)

        # Observe the error-free state and calibrate some parameters
        diffusion_sqrtm, error_estimate = self._estimate_error(
            linear_fn=linear_fn, m_obs=m_obs, p_inv=p_inv
        )
        error_estimate *= dt

        # Full extrapolation
        l_ext_p = sqrtm.sum_of_sqrtm_factors(
            R1=(self.a @ (p_inv[:, None] * l0)).T,
            R2=(diffusion_sqrtm * self.q_sqrtm_lower).T,
        ).T
        l_ext = p[:, None] * l_ext_p
        extrapolated = rv.MultivariateNormal(mean=m_ext, cov_sqrtm_lower=l_ext)

        # Final observation
        corrected = self._final_correction(
            m_ext=m_ext, l_ext=l_ext, linear_fn=linear_fn, m_obs=m_obs
        )
        state_new = FilteringSolution(extrapolated=extrapolated, corrected=corrected)

        u = jnp.reshape(corrected.mean, (-1, self.ode_dimension), order="F")[0, :]

        return state_new, error_estimate, u
