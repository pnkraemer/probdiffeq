"""Implementations for scalar initial value problems."""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from probdiffeq import _sqrt_util
from probdiffeq.statespace import _corr
from probdiffeq.statespace.scalar import variables


def taylor_order_zero(*args, **kwargs):
    return _TaylorZerothOrder(*args, **kwargs)


def correct_affine_qoi_noisy(rv, affine, *, stdev):
    # Read inputs
    A, b = affine

    # Apply observation model to covariance
    cov_sqrtm = rv.cov_sqrtm_lower
    cov_sqrtm_obs_nonsquare = jnp.dot(A, cov_sqrtm[0, ...])

    # Revert the conditional covariances
    cov_sqrtm_obs_upper, (
        cov_sqrtm_cor_upper,
        gain,
    ) = _sqrt_util.revert_conditional(
        R_X_F=cov_sqrtm_obs_nonsquare[None, :].T,
        R_X=rv.cov_sqrtm_lower.T,
        R_YX=jnp.ones((1, 1)) * stdev,
    )
    cov_sqrtm_obs = cov_sqrtm_obs_upper.T
    cov_sqrtm_cor = cov_sqrtm_cor_upper.T
    gain = gain[:, 0]  # "squeeze"; output shape is (), not (1,)

    # Gather the observed variable
    mean_obs = jnp.dot(A, rv.mean[0, ...]) + b
    observed = variables.NormalQOI(mean=mean_obs, cov_sqrtm_lower=cov_sqrtm_obs)

    # Gather the corrected variable
    mean_cor = rv.mean - gain * mean_obs
    corrected = variables.NormalHiddenState(
        mean=mean_cor, cov_sqrtm_lower=cov_sqrtm_cor
    )
    return observed, (corrected, gain)


def correct_affine_ode_2nd(rv, affine):
    # Read inputs
    A, b = affine

    # Apply observation model to covariance
    cov_sqrtm = rv.cov_sqrtm_lower
    cov_sqrtm_obs_nonsquare = cov_sqrtm[2, ...] - jnp.dot(A, cov_sqrtm[0, ...])

    # Revert the conditional covariances
    cov_sqrtm_obs_upper, (
        cov_sqrtm_cor_upper,
        gain,
    ) = _sqrt_util.revert_conditional_noisefree(
        R_X_F=cov_sqrtm_obs_nonsquare[None, :].T, R_X=rv.cov_sqrtm_lower.T
    )
    cov_sqrtm_obs = cov_sqrtm_obs_upper.T
    cov_sqrtm_cor = cov_sqrtm_cor_upper.T
    gain = gain[:, 0]  # "squeeze"; output shape is (), not (1,)

    # Gather the observed variable
    mean_obs = rv.mean[2, ...] - jnp.dot(A, rv.mean[0, ...]) - b
    observed = variables.NormalQOI(mean=mean_obs, cov_sqrtm_lower=cov_sqrtm_obs)

    # Gather the corrected variable
    mean_cor = rv.mean - gain * mean_obs
    corrected = variables.NormalHiddenState(
        mean=mean_cor, cov_sqrtm_lower=cov_sqrtm_cor
    )
    return observed, (corrected, gain)


@jax.tree_util.register_pytree_node_class
class _TaylorZerothOrder(_corr.Correction):
    def __repr__(self):
        return f"<TS0 with ode_order={self.ode_order}>"

    def init(self, ssv, /):
        m_like = jnp.zeros(())
        chol_like = jnp.zeros(())
        obs_like = variables.NormalQOI(m_like, chol_like)
        return ssv, obs_like

    def estimate_error(self, ssv: variables.SSV, corr, /, vector_field, t, p):
        m0, m1 = self.select_derivatives(ssv.hidden_state)
        fx = vector_field(*m0, t=t, p=p)
        cache, observed = self.marginalise_observation(fx, m1, ssv.hidden_state)
        mahalanobis_norm = observed.mahalanobis_norm(jnp.zeros(()))
        output_scale = mahalanobis_norm / jnp.sqrt(m1.size)
        error_estimate_unscaled = observed.marginal_stds()
        error_estimate = output_scale * error_estimate_unscaled
        return error_estimate, observed, cache

    def marginalise_observation(self, fx, m1, x):
        b = m1 - fx
        cov_sqrtm_lower = x.cov_sqrtm_lower[self.ode_order, :]
        l_obs_raw = _sqrt_util.triu_via_qr(cov_sqrtm_lower[:, None])
        l_obs = jnp.reshape(l_obs_raw, ())
        observed = variables.NormalQOI(b, l_obs)
        cache = (b,)
        return cache, observed

    def select_derivatives(self, x):
        m0, m1 = x.mean[: self.ode_order], x.mean[self.ode_order]
        return m0, m1

    def complete(self, ssv: variables.SSV, corr, /, vector_field, t, p):
        (b,) = corr
        m_ext, l_ext = (ssv.hidden_state.mean, ssv.hidden_state.cov_sqrtm_lower)

        l_obs_nonsquare = l_ext[self.ode_order, :]
        r_obs_mat, (r_cor, gain_mat) = _sqrt_util.revert_conditional_noisefree(
            R_X_F=l_obs_nonsquare[:, None], R_X=l_ext.T
        )
        r_obs = jnp.reshape(r_obs_mat, ())
        gain = jnp.reshape(gain_mat, (-1,))
        m_cor = m_ext - gain * b
        observed = variables.NormalQOI(mean=b, cov_sqrtm_lower=r_obs.T)

        rv_cor = variables.NormalHiddenState(mean=m_cor, cov_sqrtm_lower=r_cor.T)
        corrected = variables.SSV(m_cor[0], rv_cor)
        return corrected, observed

    def extract(self, ssv, corr):
        return ssv


@jax.tree_util.register_pytree_node_class
class StatisticalFirstOrder(_corr.Correction):
    def __init__(self, ode_order, cubature_rule):
        if ode_order > 1:
            raise ValueError

        super().__init__(ode_order=ode_order)
        self.cubature_rule = cubature_rule

    def __repr__(self):
        return f"<SLR0 with ode_order={self.ode_order}>"

    def tree_flatten(self):
        # todo: should this call super().tree_flatten()?
        children = (self.cubature_rule,)
        aux = (self.ode_order,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (cubature_rule,) = children
        (ode_order,) = aux
        return cls(ode_order=ode_order, cubature_rule=cubature_rule)

    def init(self, ssv, /):
        m_like = jnp.zeros(())
        chol_like = jnp.zeros(())
        obs_like = variables.NormalQOI(m_like, chol_like)
        return ssv, obs_like

    def extract(self, ssv, corr):
        return ssv

    def begin(self, ssv, corr, /, vector_field, t, p):
        raise NotImplementedError

    def calibrate(
        self,
        fx_mean: ArrayLike,
        fx_centered_normed: ArrayLike,
        extrapolated: variables.NormalHiddenState,
    ):
        fx_mean = jnp.asarray(fx_mean)
        fx_centered_normed = jnp.asarray(fx_centered_normed)

        # Extract shapes
        (S,) = fx_centered_normed.shape
        (n,) = extrapolated.mean.shape

        # Marginal mean
        m_marg = extrapolated.mean[1] - fx_mean

        # Marginal covariance
        R1 = jnp.reshape(extrapolated.cov_sqrtm_lower[1, :], (n, 1))
        R2 = jnp.reshape(fx_centered_normed, (S, 1))
        std_marg_mat = _sqrt_util.sum_of_sqrtm_factors(R_stack=(R1, R2))
        std_marg = jnp.reshape(std_marg_mat, ())

        # Extract error estimate and output scale from marginals
        marginals = variables.NormalQOI(m_marg, std_marg)
        mahalanobis_norm = marginals.mahalanobis_norm(jnp.zeros(()))
        output_scale = mahalanobis_norm / jnp.sqrt(m_marg.size)

        error_estimate_unscaled = marginals.marginal_stds()
        error_estimate = error_estimate_unscaled * output_scale
        return error_estimate, output_scale

    def complete(self, ssv, corr, /, vector_field, t, p):
        raise NotImplementedError

    def linearize(self, rv, vmap_f):
        # Create sigma points
        pts, _, pts_centered_normed = self.transform_sigma_points(rv.hidden_state)

        # Evaluate the vector-field
        fx = vmap_f(pts)
        fx_mean, _, fx_centered_normed = self.center(fx)

        # Complete linearization
        return self.linearization_matrices(
            fx_centered_normed, fx_mean, pts_centered_normed, rv
        )

    def transform_sigma_points(self, rv: variables.NormalHiddenState):
        # Extract square-root of covariance (-> std-dev.)
        L0_nonsq = rv.cov_sqrtm_lower[0, :]
        r_marg1_x_mat = _sqrt_util.triu_via_qr(L0_nonsq[:, None])
        r_marg1_x = jnp.reshape(r_marg1_x_mat, ())

        # Multiply and shift the unit-points
        m_marg1_x = rv.mean[0]
        sigma_points_centered = self.cubature_rule.points * r_marg1_x[None]
        sigma_points = m_marg1_x[None] + sigma_points_centered

        # Scale the shifted points with square-root weights
        _w = self.cubature_rule.weights_sqrtm
        sigma_points_centered_normed = sigma_points_centered * _w
        return sigma_points, sigma_points_centered, sigma_points_centered_normed

    def center(self, fx):
        fx_mean = self.cubature_rule.weights_sqrtm**2 @ fx
        fx_centered = fx - fx_mean[None]
        fx_centered_normed = fx_centered * self.cubature_rule.weights_sqrtm
        return fx_mean, fx_centered, fx_centered_normed

    def linearization_matrices(
        self, fx_centered_normed, fx_mean, pts_centered_normed, rv
    ):
        # Revert the transition to get H and Omega
        # This is a pure sqrt-implementation of
        # Eq. (9) in https://arxiv.org/abs/2102.00514
        # It seems to be different to Section VI.B in
        # https://arxiv.org/pdf/2207.00426.pdf,
        # because the implementation below avoids sqrt-down-dates
        # pts_centered_normed = pts_centered * self.cubature_rule.weights_sqrtm[:, None]
        _, (std_noi_mat, linop_mat) = _sqrt_util.revert_conditional_noisefree(
            R_X_F=pts_centered_normed[:, None], R_X=fx_centered_normed[:, None]
        )
        std_noi = jnp.reshape(std_noi_mat, ())
        linop = jnp.reshape(linop_mat, ())

        # Catch up the transition-mean and return the result
        m_noi = fx_mean - linop * rv.mean[0]
        return linop, variables.NormalQOI(m_noi, std_noi)

    def complete_post_linearize(self, linop, rv, noise):
        # Compute the cubature-correction
        L0, L1 = (rv.cov_sqrtm_lower[0, :], rv.cov_sqrtm_lower[1, :])
        HL = L1 - linop * L0
        std_marg_mat, (r_bw, gain_mat) = _sqrt_util.revert_conditional(
            R_X=rv.cov_sqrtm_lower.T,
            R_X_F=HL[:, None],
            R_YX=noise.cov_sqrtm_lower[None, None],
        )

        # Reshape the matrices into appropriate scalar-valued versions
        (n,) = rv.mean.shape
        std_marg = jnp.reshape(std_marg_mat, ())
        gain = jnp.reshape(gain_mat, (n,))

        # Catch up the marginals
        x0, x1 = rv.mean[0], rv.mean[1]
        m_marg = x1 - (linop * x0 + noise.mean)
        obs = variables.NormalQOI(m_marg, std_marg)

        # Catch up the backward noise and return result
        m_bw = rv.mean - gain * m_marg
        rv_cor = variables.NormalHiddenState(m_bw, r_bw.T)
        cor = variables.SSV(m_bw[0], rv_cor)
        return cor, obs
