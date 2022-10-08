"""ODE filter backends."""
from typing import Any, Callable, Generic, List, Tuple, TypeVar

import equinox as eqx
import jax.numpy as jnp
from jax import tree_util
from jaxtyping import Array, Float

from odefilter import sqrtm
from odefilter.prob import ibm, rv

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

        return FilteringSolution(extrapolated=extrapolated, corrected=corrected)

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
        return state_new, error_estimate, jnp.squeeze(corrected.mean[0])


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
        )

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
        r_ext_p, (r_bw_p, g_bw_p) = sqrtm.revert_gaussian_markov_kernel(
            h_matmul_c_sqrtm_upper=(self.a @ l0_p).T,
            c_sqrtm_upper=l0_p.T,
            r_sqrtm_upper=(diffusion_sqrtm * self.q_sqrtm_lower).T,
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
        return state_new, error_estimate, jnp.squeeze(corrected.mean[0])
