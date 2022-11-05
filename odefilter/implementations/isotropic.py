"""State-space models with isotropic covariance structure."""

import dataclasses
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from odefilter import _control_flow
from odefilter.implementations import _correction, _extrapolation, _ibm_util, _sqrtm


class _IsoNormal(NamedTuple):
    mean: Any  # (n, d) shape
    cov_sqrtm_lower: Any  # (n,n) shape


@jax.tree_util.register_pytree_node_class
class IsoTS0:
    def __init__(self, *, correction: "IsoTaylorZerothOrder", extrapolation: "IsoIBM"):
        self.correction = correction
        self.extrapolation = extrapolation

    def tree_flatten(self):
        children = (self.correction, self.extrapolation)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        correction, extrapolation = children
        return cls(correction=correction, extrapolation=extrapolation)

    @classmethod
    def from_params(cls, **kwargs):
        correction = IsoTaylorZerothOrder()
        extrapolation = IsoIBM.from_params(**kwargs)
        return cls(correction=correction, extrapolation=extrapolation)


@jax.tree_util.register_pytree_node_class
class IsoTaylorZerothOrder(_correction.Correction):
    def begin_correction(self, x: _IsoNormal, /, *, vector_field, t, p):
        m = x.mean
        m0, m1 = m[: self.ode_order, ...], m[self.ode_order, ...]
        bias = m1 - vector_field(*m0, t=t, p=p)
        cov_sqrtm_lower = x.cov_sqrtm_lower[self.ode_order, ...]

        l_obs = jnp.reshape(
            _sqrtm.sqrtm_to_upper_triangular(R=cov_sqrtm_lower[:, None]), ()
        )
        res_white = (bias / l_obs) / jnp.sqrt(bias.size)

        # jnp.sqrt(\|res_white\|^2/d) without forming the square
        output_scale_sqrtm = jnp.reshape(
            _sqrtm.sqrtm_to_upper_triangular(R=res_white[:, None]), ()
        )

        error_estimate = l_obs
        return output_scale_sqrtm * error_estimate, output_scale_sqrtm, (bias,)

    def complete_correction(self, *, extrapolated, cache):
        (bias,) = cache

        m_ext, l_ext = extrapolated.mean, extrapolated.cov_sqrtm_lower
        l_obs = l_ext[self.ode_order, ...]

        l_obs_scalar = jnp.reshape(
            _sqrtm.sqrtm_to_upper_triangular(R=l_obs[:, None]), ()
        )
        c_obs = l_obs_scalar**2

        observed = _IsoNormal(mean=bias, cov_sqrtm_lower=l_obs_scalar)

        g = (l_ext @ l_obs.T) / c_obs  # shape (n,)
        m_cor = m_ext - g[:, None] * bias[None, :]
        l_cor = l_ext - g[:, None] * l_obs[None, :]
        corrected = _IsoNormal(mean=m_cor, cov_sqrtm_lower=l_cor)
        return observed, (corrected, g)

    def evidence_sqrtm(self, *, observed):
        obs_pt, l_obs = observed.mean, observed.cov_sqrtm_lower
        res_white = obs_pt / l_obs
        evidence_sqrtm = jnp.sqrt(jnp.dot(res_white, res_white.T) / res_white.size)
        return evidence_sqrtm

    def correct_sol_observation(self, *, rv, u, observation_std):
        hc = rv.cov_sqrtm_lower[0, ...].reshape((1, -1))
        m_obs = rv.mean[0, ...]

        r_yx = observation_std * jnp.ones((1, 1))
        r_obs, (r_cor, gain) = _sqrtm.revert_conditional(
            R_X_F=hc.T, R_X=rv.cov_sqrtm_lower.T, R_YX=r_yx
        )
        m_cor = rv.mean - gain * (m_obs - u)[None, :]

        return _IsoNormal(m_obs, r_obs.T), (_IsoNormal(m_cor, r_cor.T), gain)

    def negative_marginal_log_likelihood(self, observed, u):
        m_obs, l_obs = observed.mean, observed.cov_sqrtm_lower

        res_white = (m_obs - u) / jnp.reshape(l_obs, ())

        x1 = jnp.dot(res_white, res_white.T)
        x2 = jnp.reshape(l_obs, ()) ** 2
        x3 = res_white.size * jnp.log(jnp.pi * 2)
        return 0.5 * (x1 + x2 + x3)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class IsoIBM(_extrapolation.Extrapolation):

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
    def from_params(cls, *, num_derivatives=4):
        a, q_sqrtm = _ibm_util.system_matrices_1d(num_derivatives=num_derivatives)
        return cls(a=a, q_sqrtm_lower=q_sqrtm)

    @property
    def num_derivatives(self):
        return self.a.shape[0] - 1

    def init_corrected(self, *, taylor_coefficients):
        m0_corrected = jnp.vstack(taylor_coefficients)
        c_sqrtm0_corrected = jnp.zeros_like(self.q_sqrtm_lower)
        return _IsoNormal(mean=m0_corrected, cov_sqrtm_lower=c_sqrtm0_corrected)

    def init_error_estimate(self):
        return jnp.zeros(())  # the initialisation is error-free

    def init_backward_transition(self):
        return jnp.eye(*self.a.shape)

    def init_backward_noise(self, *, rv_proto):
        return _IsoNormal(
            mean=jnp.zeros_like(rv_proto.mean),
            cov_sqrtm_lower=jnp.zeros_like(rv_proto.cov_sqrtm_lower),
        )

    def init_output_scale_sqrtm(self):
        return 1.0

    def begin_extrapolation(self, m0, /, *, dt):
        p, p_inv = self._assemble_preconditioner(dt=dt)
        m0_p = p_inv[:, None] * m0
        m_ext_p = self.a @ m0_p
        m_ext = p[:, None] * m_ext_p
        q_sqrtm = p[:, None] * self.q_sqrtm_lower
        return _IsoNormal(m_ext, q_sqrtm), (m_ext_p, m0_p, p, p_inv)

    def _assemble_preconditioner(self, *, dt):
        return _ibm_util.preconditioner_diagonal(
            dt=dt, num_derivatives=self.num_derivatives
        )

    def complete_extrapolation(
        self, *, linearisation_pt, l0, cache, output_scale_sqrtm
    ):
        _, _, p, p_inv = cache
        m_ext = linearisation_pt.mean

        l0_p = p_inv[:, None] * l0
        l_ext_p = _sqrtm.sum_of_sqrtm_factors(
            R1=(self.a @ l0_p).T,
            R2=(output_scale_sqrtm * self.q_sqrtm_lower).T,
        ).T
        l_ext = p[:, None] * l_ext_p
        return _IsoNormal(m_ext, l_ext)

    def revert_markov_kernel(self, *, linearisation_pt, l0, cache, output_scale_sqrtm):
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
        m_bw = p[:, None] * m_bw_p
        l_bw = p[:, None] * l_bw_p
        g_bw = p[:, None] * g_bw_p * p_inv[None, :]

        backward_op = g_bw
        backward_noise = _IsoNormal(mean=m_bw, cov_sqrtm_lower=l_bw)
        extrapolated = _IsoNormal(mean=m_ext, cov_sqrtm_lower=l_ext)
        return extrapolated, (backward_noise, backward_op)

    def extract_sol(self, *, rv):
        m = rv.mean[..., 0, :]
        return m

    def condense_backward_models(
        self, *, transition_init, noise_init, transition_state, noise_state
    ):

        A = transition_init
        (b, B_sqrtm) = noise_init.mean, noise_init.cov_sqrtm_lower

        C = transition_state
        (d, D_sqrtm) = (noise_state.mean, noise_state.cov_sqrtm_lower)

        g = A @ C
        xi = A @ d + b
        Xi = _sqrtm.sum_of_sqrtm_factors(R1=(A @ D_sqrtm).T, R2=B_sqrtm.T).T

        noise = _IsoNormal(mean=xi, cov_sqrtm_lower=Xi)
        return noise, g

    def marginalise_backwards(self, *, init, linop, noise):
        """Compute marginals of a markov sequence."""

        def body_fun(carry, x):
            op, noi = x
            out = self.marginalise_model(init=carry, linop=op, noise=noi)
            return out, out

        # Initial condition does not matter
        bw_models = jax.tree_util.tree_map(lambda x: x[1:, ...], (linop, noise))
        _, rvs = _control_flow.scan_with_init(
            f=body_fun, init=init, xs=bw_models, reverse=True
        )
        return rvs

    def marginalise_model(self, *, init, linop, noise):
        """Marginalise the output of a linear model."""
        # todo: add preconditioner?

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

        return _IsoNormal(mean=m_new, cov_sqrtm_lower=l_new)

    def sample_backwards(self, init, linop, noise, base_samples):
        def body_fun(carry, x):
            op, noi = x
            out = op @ carry + noi
            return out, out

        linop_sample, noise_ = jax.tree_util.tree_map(
            lambda x: x[1:, ...], (linop, noise)
        )
        noise_sample = self._transform_samples(noise_, base_samples[..., :-1, :, :])
        init_sample = self._transform_samples(init, base_samples[..., -1, :, :])

        # todo: should we use an associative scan here?
        _, samples = _control_flow.scan_with_init(
            f=body_fun, init=init_sample, xs=(linop_sample, noise_sample), reverse=True
        )
        return samples

    # automatically batched because of numpy's broadcasting rules?
    def _transform_samples(self, rvs, base):
        return rvs.mean + rvs.cov_sqrtm_lower @ base

    def extract_mean_from_marginals(self, mean):
        return mean[..., 0, :]

    def scale_covariance(self, *, rv, scale_sqrtm):
        if jnp.ndim(scale_sqrtm) == 0:
            return _IsoNormal(
                mean=rv.mean, cov_sqrtm_lower=scale_sqrtm * rv.cov_sqrtm_lower
            )
        return _IsoNormal(
            mean=rv.mean,
            cov_sqrtm_lower=scale_sqrtm[:, None, None] * rv.cov_sqrtm_lower,
        )
