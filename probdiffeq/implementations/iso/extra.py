"""Extrapolations."""

from typing import Any, Tuple

import jax
import jax.numpy as jnp

from probdiffeq.implementations import _collections, _ibm_util, _sqrtm
from probdiffeq.implementations.iso import _vars


@jax.tree_util.register_pytree_node_class
class IsoConditional(_collections.AbstractConditional):
    def __init__(self, transition, noise):
        self.transition = transition
        self.noise = noise

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(transition={self.transition}, noise={self.noise})"

    def tree_flatten(self):
        children = self.transition, self.noise
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        transition, noise = children
        return cls(transition=transition, noise=noise)

    def __call__(self, x, /):
        m = self.transition @ x + self.noise.mean
        return _vars.IsoStateSpaceVar(_vars.IsoNormal(m, self.noise.cov_sqrtm_lower))

    def scale_covariance(self, scale_sqrtm):
        noise = self.noise.scale_covariance(scale_sqrtm=scale_sqrtm)
        return IsoConditional(transition=self.transition, noise=noise)

    def merge_with_incoming_conditional(self, incoming, /):
        A = self.transition
        (b, B_sqrtm) = self.noise.mean, self.noise.cov_sqrtm_lower

        C = incoming.transition
        (d, D_sqrtm) = (incoming.noise.mean, incoming.noise.cov_sqrtm_lower)

        g = A @ C
        xi = A @ d + b
        Xi = _sqrtm.sum_of_sqrtm_factors(R_stack=((A @ D_sqrtm).T, B_sqrtm.T)).T

        noise = _vars.IsoNormal(mean=xi, cov_sqrtm_lower=Xi)
        bw_model = IsoConditional(g, noise=noise)
        return bw_model

    def marginalise(self, rv: _vars.IsoStateSpaceVar, /) -> _vars.IsoStateSpaceVar:
        """Marginalise the output of a linear model."""
        # Read
        m0 = rv.hidden_state.mean
        l0 = rv.hidden_state.cov_sqrtm_lower

        # Apply transition
        m_new = self.transition @ m0 + self.noise.mean
        l_new = _sqrtm.sum_of_sqrtm_factors(
            R_stack=((self.transition @ l0).T, self.noise.cov_sqrtm_lower.T)
        ).T

        return _vars.IsoStateSpaceVar(
            _vars.IsoNormal(mean=m_new, cov_sqrtm_lower=l_new)
        )


@jax.tree_util.register_pytree_node_class
class IsoIBM(_collections.AbstractExtrapolation):
    def __init__(self, a, q_sqrtm_lower):
        self.a = a
        self.q_sqrtm_lower = q_sqrtm_lower

    def __repr__(self):
        name = self.__class__.__name__
        return f"{name}(a={self.a}, q_sqrtm_lower={self.q_sqrtm_lower})"

    def tree_flatten(self):
        children = self.a, self.q_sqrtm_lower
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        a, q_sqrtm_lower = children
        return cls(a=a, q_sqrtm_lower=q_sqrtm_lower)

    @classmethod
    def from_params(cls, num_derivatives):
        a, q_sqrtm = _ibm_util.system_matrices_1d(num_derivatives=num_derivatives)
        return cls(a=a, q_sqrtm_lower=q_sqrtm)

    @property
    def num_derivatives(self):
        return self.a.shape[0] - 1

    def init_corrected(self, taylor_coefficients):
        m0_corrected = jnp.vstack(taylor_coefficients)
        c_sqrtm0_corrected = jnp.zeros_like(self.q_sqrtm_lower)
        rv = _vars.IsoNormal(mean=m0_corrected, cov_sqrtm_lower=c_sqrtm0_corrected)
        return _vars.IsoStateSpaceVar(rv)

    def init_ssv(self, ode_shape):
        assert len(ode_shape) == 1
        (d,) = ode_shape
        m0 = jnp.zeros((self.num_derivatives + 1, d))
        c0 = jnp.eye(self.num_derivatives + 1)
        return _vars.IsoStateSpaceVar(_vars.IsoNormal(m0, c0))

    def init_error_estimate(self):
        return jnp.zeros(())  # the initialisation is error-free

    def init_output_scale_sqrtm(self):
        return 1.0

    def begin_extrapolation(
        self, p0: _vars.IsoStateSpaceVar, /, dt
    ) -> Tuple[_vars.IsoStateSpaceVar, Any]:
        p, p_inv = self._assemble_preconditioner(dt=dt)
        m0_p = p_inv[:, None] * p0.hidden_state.mean
        m_ext_p = self.a @ m0_p
        m_ext = p[:, None] * m_ext_p
        q_sqrtm = p[:, None] * self.q_sqrtm_lower

        ext = _vars.IsoNormal(m_ext, q_sqrtm)
        return _vars.IsoStateSpaceVar(ext), (m_ext_p, m0_p, p, p_inv)

    def _assemble_preconditioner(self, dt):
        return _ibm_util.preconditioner_diagonal(
            dt=dt, num_derivatives=self.num_derivatives
        )

    def complete_extrapolation(self, linearisation_pt, p0, cache, output_scale_sqrtm):
        _, _, p, p_inv = cache
        m_ext = linearisation_pt.hidden_state.mean

        l0_p = p_inv[:, None] * p0.hidden_state.cov_sqrtm_lower
        l_ext_p = _sqrtm.sum_of_sqrtm_factors(
            R_stack=((self.a @ l0_p).T, (output_scale_sqrtm * self.q_sqrtm_lower).T)
        ).T
        l_ext = p[:, None] * l_ext_p
        return _vars.IsoStateSpaceVar(_vars.IsoNormal(m_ext, l_ext))

    def revert_markov_kernel(self, linearisation_pt, p0, cache, output_scale_sqrtm):
        m_ext_p, m0_p, p, p_inv = cache
        m_ext = linearisation_pt.hidden_state.mean

        l0_p = p_inv[:, None] * p0.hidden_state.cov_sqrtm_lower
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

        backward_noise = _vars.IsoNormal(mean=m_bw, cov_sqrtm_lower=l_bw)
        bw_model = IsoConditional(g_bw, noise=backward_noise)
        extrapolated = _vars.IsoNormal(mean=m_ext, cov_sqrtm_lower=l_ext)
        return _vars.IsoStateSpaceVar(extrapolated), bw_model

    # todo: should this be a classmethod in IsoConditional?
    def init_conditional(self, ssv_proto):
        op = self._init_backward_transition()
        noi = self._init_backward_noise(rv_proto=ssv_proto.hidden_state)
        return IsoConditional(op, noise=noi)

    def _init_backward_transition(self):
        return jnp.eye(*self.a.shape)

    def _init_backward_noise(self, rv_proto):
        return _vars.IsoNormal(
            mean=jnp.zeros_like(rv_proto.mean),
            cov_sqrtm_lower=jnp.zeros_like(rv_proto.cov_sqrtm_lower),
        )
