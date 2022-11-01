"""Batch-style extrapolations."""
from dataclasses import dataclass
from typing import Any, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_pytree_node_class

from odefilter import _control_flow
from odefilter.implementations import _correction, _extrapolation, _ibm_util, _sqrtm

# todo: reconsider naming!


class _BatchNormal(NamedTuple):
    """Random variable with a normal distribution."""

    mean: Any  # (d, k) shape
    cov_sqrtm_lower: Any  # (d, k, k) shape


_CType = Tuple[Array]  # Cache type


@register_pytree_node_class
class TaylorZerothOrder(_correction.Correction[_BatchNormal, _CType]):
    """TaylorZerothOrder-linearise an ODE assuming a linearisation-point with\
     isotropic Kronecker structure."""

    def begin_correction(
        self, x: _BatchNormal, /, *, vector_field, t, p
    ) -> Tuple[Array, float, _CType]:
        m = x.mean

        # m has shape (d, n)
        bias = m[..., self.ode_order] - vector_field(
            *(m[..., : self.ode_order]).T, t=t, p=p
        )
        l_obs_nonsquare = self._cov_sqrtm_lower(
            cache=(), cov_sqrtm_lower=x.cov_sqrtm_lower
        )

        l_obs_nonsquare_1 = l_obs_nonsquare[..., None]  # (d, k, 1)

        # (d, 1, 1)
        l_obs_raw = jax.vmap(_sqrtm.sqrtm_to_upper_triangular)(R=l_obs_nonsquare_1)
        l_obs = l_obs_raw[..., 0, 0]  # (d,)

        output_scale_sqrtm = self.evidence_sqrtm(
            observed=_BatchNormal(mean=bias, cov_sqrtm_lower=l_obs)
        )  # (d,)

        error_estimate = l_obs  # (d,)
        return output_scale_sqrtm * error_estimate, output_scale_sqrtm, (bias,)

    def complete_correction(self, *, extrapolated: _BatchNormal, cache: _CType):
        (bias,) = cache

        # (d, k), (d, k, k)
        m_ext, l_ext = extrapolated.mean, extrapolated.cov_sqrtm_lower
        l_obs_nonsquare = self._cov_sqrtm_lower(
            cache=(), cov_sqrtm_lower=l_ext
        )  # (d, k)

        # (d, 1, 1)
        l_obs = _sqrtm.sqrtm_to_upper_triangular(R=l_obs_nonsquare[..., None])
        l_obs_scalar = l_obs[..., 0, 0]  # (d,)

        # (d,), (d,)
        observed = _BatchNormal(mean=bias, cov_sqrtm_lower=l_obs_scalar)

        # (d, k)
        crosscov = (l_ext @ l_obs_nonsquare[..., None])[..., 0]

        gain = crosscov / (l_obs_scalar[..., None]) ** 2  # (d, k)

        m_cor = m_ext - (gain * bias[..., None])  # (d, k)
        l_cor = l_ext - gain[..., None] * l_obs_nonsquare[..., None, :]  # (d, k, k)

        corrected = _BatchNormal(mean=m_cor, cov_sqrtm_lower=l_cor)
        return observed, (corrected, gain)

    def _cov_sqrtm_lower(self, *, cache, cov_sqrtm_lower):
        return cov_sqrtm_lower[:, self.ode_order, ...]

    def evidence_sqrtm(self, *, observed: _BatchNormal) -> float:
        obs_pt, l_obs = observed.mean, observed.cov_sqrtm_lower  # (d,), (d,)

        res_white = obs_pt / l_obs  # (d, )
        evidence_sqrtm = res_white / jnp.sqrt(res_white.size)

        return evidence_sqrtm


@register_pytree_node_class
@dataclass(frozen=True)
class BatchIBM(_extrapolation.Extrapolation):
    """Handle block-diagonal covariances."""

    a: Any
    q_sqrtm_lower: Any

    @property
    def num_derivatives(self):
        return self.a.shape[1] - 1

    @property
    def ode_dimension(self):
        return self.a.shape[0]

    def tree_flatten(self):
        children = self.a, self.q_sqrtm_lower
        return children, ()

    @classmethod
    def tree_unflatten(cls, _aux, children):
        a, q_sqrtm_lower = children
        return cls(a=a, q_sqrtm_lower=q_sqrtm_lower)

    @classmethod
    def from_params(cls, *, ode_dimension, num_derivatives=4):
        """Create a strategy from hyperparameters."""
        a, q_sqrtm = _ibm_util.system_matrices_1d(num_derivatives=num_derivatives)
        a = jnp.stack([a] * ode_dimension)
        q_sqrtm = jnp.stack([q_sqrtm] * ode_dimension)
        return cls(a=a, q_sqrtm_lower=q_sqrtm)

    def _assemble_preconditioner(self, *, dt):  # noqa: D102
        p, p_inv = _ibm_util.preconditioner_diagonal(
            dt=dt, num_derivatives=self.num_derivatives
        )
        p = jnp.stack([p] * self.ode_dimension)
        p_inv = jnp.stack([p_inv] * self.ode_dimension)
        return p, p_inv

    def complete_extrapolation(
        self, *, linearisation_pt, cache, l0, output_scale_sqrtm
    ):
        *_, p, p_inv = cache
        m_ext = linearisation_pt.mean
        r_ext_p = jax.vmap(_sqrtm.sum_of_sqrtm_factors)(
            R1=_transpose(self.a @ (p_inv[..., None] * l0)),
            R2=_transpose(output_scale_sqrtm[:, None, None] * self.q_sqrtm_lower),
        )
        l_ext_p = _transpose(r_ext_p)
        l_ext = p[..., None] * l_ext_p
        return _BatchNormal(mean=m_ext, cov_sqrtm_lower=l_ext)

    def condense_backward_models(
        self, *, transition_init, noise_init, transition_state, noise_state
    ):  # noqa: D102

        A = transition_init  # (d, k, k)
        # (d, k), (d, k, k)
        (b, B_sqrtm) = (noise_init.mean, noise_init.cov_sqrtm_lower)

        C = transition_state  # (d, k, k)
        # (d, k), (d, k, k)
        (d, D_sqrtm) = (noise_state.mean, noise_state.cov_sqrtm_lower)

        g = A @ C
        xi = (A @ d[..., None])[..., 0] + b
        Xi_r = jax.vmap(_sqrtm.sum_of_sqrtm_factors)(
            R1=_transpose(A @ D_sqrtm), R2=_transpose(B_sqrtm)
        )
        Xi = _transpose(Xi_r)

        noise = _BatchNormal(mean=xi, cov_sqrtm_lower=Xi)
        return noise, g

    def extract_mean_from_marginals(self, mean):
        return mean[..., 0]

    def extract_sol(self, *, rv):  # noqa: D102
        return self.extract_mean_from_marginals(mean=rv.mean)

    def begin_extrapolation(self, m0, /, *, dt):

        p, p_inv = self._assemble_preconditioner(dt=dt)
        m0_p = p_inv * m0  # (d, k)
        m_ext_p = (self.a @ m0_p[..., None])[..., 0]  # (d, k)
        m_ext = p * m_ext_p

        q_ext = p[..., None] * self.q_sqrtm_lower
        return _BatchNormal(m_ext, q_ext), (m_ext_p, m0_p, p, p_inv)

    # todo: make into init_backward_model?
    def init_backward_noise(self, *, rv_proto):  # noqa: D102
        return _BatchNormal(
            mean=jnp.zeros_like(rv_proto.mean),
            cov_sqrtm_lower=jnp.zeros_like(rv_proto.cov_sqrtm_lower),
        )

    def init_backward_transition(self):  # noqa: D102
        return jnp.stack([jnp.eye(self.num_derivatives + 1)] * self.ode_dimension)

    def init_corrected(self, *, taylor_coefficients):
        """Initialise the "corrected" RV by stacking Taylor coefficients."""
        m0_matrix = jnp.vstack(taylor_coefficients).T
        c_sqrtm0_corrected = jnp.zeros_like(self.q_sqrtm_lower)
        return _BatchNormal(mean=m0_matrix, cov_sqrtm_lower=c_sqrtm0_corrected)

    # todo: move to correction?
    def init_error_estimate(self):  # noqa: D102
        return jnp.zeros((self.ode_dimension,))  # the initialisation is error-free

    # todo: move to correction?
    def init_output_scale_sqrtm(self):
        return jnp.ones((self.ode_dimension,))

    def marginalise_backwards(self, *, init, linop, noise):
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
        # todo: add preconditioner?

        # Pull into preconditioned space
        m0_p = init.mean
        l0_p = init.cov_sqrtm_lower

        # Apply transition
        m_new_p = (linop @ m0_p[..., None])[..., 0] + noise.mean
        r_new_p = jax.vmap(_sqrtm.sum_of_sqrtm_factors)(
            R1=_transpose(linop @ l0_p), R2=_transpose(noise.cov_sqrtm_lower)
        )
        l_new_p = _transpose(r_new_p)

        # Push back into non-preconditioned space
        m_new = m_new_p
        l_new = l_new_p

        return _BatchNormal(mean=m_new, cov_sqrtm_lower=l_new)

    def revert_markov_kernel(  # noqa: D102
        self, *, linearisation_pt, l0, output_scale_sqrtm, cache
    ):
        m_ext_p, m0_p, p, p_inv = cache
        m_ext = linearisation_pt.mean

        # (d, k, 1) * (d, k, k) = (d, k, k)
        l0_p = p_inv[..., None] * l0

        r_ext_p, (r_bw_p, g_bw_p) = jax.vmap(_sqrtm.revert_conditional)(
            R_X_F=_transpose(self.a @ l0_p),
            R_X=_transpose(l0_p),
            # transpose((d, 1, 1) * (d, k, k)) = tranpose((d,k,k)) = (d, k, k)
            R_YX=_transpose(output_scale_sqrtm[..., None, None] * self.q_sqrtm_lower),
        )
        l_ext_p, l_bw_p = _transpose(r_ext_p), _transpose(r_bw_p)
        m_bw_p = m0_p - (g_bw_p @ m_ext_p[..., None])[..., 0]

        # Un-apply the pre-conditioner.
        # The backward models remains preconditioned, because
        # we do backward passes in preconditioner-space.
        l_ext = p[..., None] * l_ext_p
        m_bw = p * m_bw_p
        l_bw = p[..., None] * l_bw_p
        g_bw = p[..., None] * g_bw_p * p_inv[:, None, :]

        backward_op = g_bw
        backward_noise = _BatchNormal(mean=m_bw, cov_sqrtm_lower=l_bw)
        extrapolated = _BatchNormal(mean=m_ext, cov_sqrtm_lower=l_ext)
        return extrapolated, (backward_noise, backward_op)

    def sample_backwards(self, init, linop, noise, base_samples):
        def body_fun(carry, x):
            op, noi = x
            out = (op @ carry[..., None])[..., 0] + noi
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
        # (d,k) + ((d,k,k) @ (d,k,1))[..., 0] = (d,k)
        return rvs.mean + (rvs.cov_sqrtm_lower @ base[..., None])[..., 0]

    def scale_covariance(self, *, rv, scale_sqrtm):
        # Endpoint: (d, 1, 1) * (d, k, k) -> (d, k, k)
        if jnp.ndim(scale_sqrtm) == 1:
            return _BatchNormal(
                mean=rv.mean,
                cov_sqrtm_lower=scale_sqrtm[:, None, None] * rv.cov_sqrtm_lower,
            )

        # Time series: (N, d, 1, 1) * (N, d, k, k) -> (N, d, k, k)
        return _BatchNormal(
            mean=rv.mean,
            cov_sqrtm_lower=scale_sqrtm[..., None, None] * rv.cov_sqrtm_lower,
        )


def _transpose(x):
    return jnp.swapaxes(x, -1, -2)
