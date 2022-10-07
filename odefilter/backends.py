"""ODE filter backends."""
from typing import Any, Callable, Generic, List, Tuple, TypeVar

import equinox as eqx
import jax.numpy as jnp
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


class DynamicIsotropicFilter(eqx.Module):
    """EK0 for terminal-value simulation with an isotropic covariance \
     structure and dynamic (time-varying) calibration."""

    a: Any
    q_sqrtm_upper: Any

    information: Any

    @classmethod
    def from_num_derivatives(cls, *, num_derivatives, information):
        """Create a backend from hyperparameters."""
        a, q_sqrtm = ibm.system_matrices_1d(num_derivatives=num_derivatives)
        return cls(a=a, q_sqrtm_upper=q_sqrtm.T, information=information)

    @property
    def q_sqrtm_lower(self):
        """Lower square root matrix."""
        return self.q_sqrtm_upper.T

    @property
    def num_derivatives(self):
        """Number of derivatives in the state-space model."""  # noqa: D401
        return self.a.shape[0] - 1

    def init_fn(
        self, *, taylor_coefficients: List[Float[Array, " d"]]
    ) -> FilteringSolution[rv.IsotropicNormal]:
        """Initialise."""
        # Infer the "corrected" random variable from the Taylor coefficients.
        # (There is no actual correction, because we have perfect information.)
        m0_corrected = jnp.stack(taylor_coefficients)
        if m0_corrected.ndim == 1:
            m0_corrected = m0_corrected[:, None]
        c_sqrtm0_corrected = jnp.zeros_like(self.q_sqrtm_upper)
        corrected = rv.IsotropicNormal(
            mean=m0_corrected, cov_sqrtm_upper=c_sqrtm0_corrected
        )

        # Invent an "extrapolated" random variable.
        # It is required to have type- and shape-stability in the loop.
        # We make it standard-normal because of a lack of better ideas.
        # Its values are irrelevant.
        m0_extrapolated = jnp.zeros_like(m0_corrected)
        c_sqrtm0_extrapolated = jnp.eye(*c_sqrtm0_corrected.shape)
        extrapolated = rv.IsotropicNormal(
            mean=m0_extrapolated, cov_sqrtm_upper=c_sqrtm0_extrapolated
        )
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
        # Compute preconditioner
        p, p_inv = ibm.preconditioner_diagonal(
            dt=dt, num_derivatives=self.num_derivatives
        )
        # Extract previous correction
        (m0, c_sqrtm0) = state.corrected.mean, state.corrected.cov_sqrtm_upper

        # Extrapolate the mean and linearise the differential equation.
        m_extrapolated = p[:, None] * (self.a @ (p_inv[:, None] * m0))
        bias, linear_fn = self.information(f=vector_field, x=m_extrapolated)

        # Observe the error-free state and calibrate some parameters
        s_sqrtm_lower = linear_fn(p_inv[:, None] * self.q_sqrtm_lower)
        s = jnp.dot(s_sqrtm_lower, s_sqrtm_lower)
        residual_white = bias / jnp.sqrt(s)
        diffusion_sqrtm = jnp.sqrt(
            jnp.dot(residual_white, residual_white) / residual_white.size
        )
        error_estimate = dt * diffusion_sqrtm * jnp.sqrt(s)

        # Full extrapolation
        c_sqrtm_extrapolated_p = sqrtm.sum_of_sqrtm_factors(
            R1=(self.a @ (p_inv[:, None] * c_sqrtm0)).T,
            R2=diffusion_sqrtm * self.q_sqrtm_lower,
        ).T
        c_sqrtm_extrapolated = p[:, None] * c_sqrtm_extrapolated_p

        rv_extrapolated = rv.IsotropicNormal(
            mean=m_extrapolated, cov_sqrtm_upper=c_sqrtm_extrapolated
        )

        # Final observation
        s_sqrtm = linear_fn(rv_extrapolated.cov_sqrtm_upper.T)  # shape (n,)
        s = jnp.dot(s_sqrtm, s_sqrtm)
        rv_observed = rv.IsotropicNormal(mean=bias, cov_sqrtm_upper=jnp.sqrt(s))
        g = (rv_extrapolated.cov_sqrtm_upper.T @ s_sqrtm.T) / s  # shape (n,)

        # Final correction
        m_cor = rv_extrapolated.mean - g[:, None] * rv_observed.mean[None, :]
        c_sqrtm_cor = rv_extrapolated.cov_sqrtm_upper.T - g[:, None] * s_sqrtm[None, :]
        rv_corrected = rv.IsotropicNormal(mean=m_cor, cov_sqrtm_upper=c_sqrtm_cor.T)

        state_new = FilteringSolution(
            extrapolated=rv_extrapolated, corrected=rv_corrected
        )
        return state_new, error_estimate, jnp.squeeze(rv_corrected.mean[0])
