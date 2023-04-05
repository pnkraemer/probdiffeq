"""Implementations for scalar initial value problems."""

import jax
import jax.numpy as jnp

from probdiffeq import _sqrt_util
from probdiffeq.implementations import _collections, _ibm_util
from probdiffeq.implementations.scalar import _conds, _vars


@jax.tree_util.register_pytree_node_class
class IBM(_collections.AbstractExtrapolation):
    def __init__(self, a, q_sqrtm_lower, preconditioner_scales, preconditioner_powers):
        self.a = a
        self.q_sqrtm_lower = q_sqrtm_lower

        self.preconditioner_scales = preconditioner_scales
        self.preconditioner_powers = preconditioner_powers

    def __repr__(self):
        name = self.__class__.__name__
        args1 = f"a={self.a}, q={self.q_sqrtm_lower}"
        args2 = f"preconditioner_scales={self.preconditioner_scales}"
        args3 = f"preconditioner_powers={self.preconditioner_powers}"
        return f"{name}({args1}, {args2}, {args3})"

    def tree_flatten(self):
        children = (
            self.a,
            self.q_sqrtm_lower,
            self.preconditioner_scales,
            self.preconditioner_powers,
        )
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        a, q_sqrtm_lower, scales, powers = children
        return cls(
            a=a,
            q_sqrtm_lower=q_sqrtm_lower,
            preconditioner_scales=scales,
            preconditioner_powers=powers,
        )

    @classmethod
    def from_params(cls, num_derivatives):
        a, q_sqrtm = _ibm_util.system_matrices_1d(num_derivatives=num_derivatives)
        _tmp = _ibm_util.preconditioner_prepare(num_derivatives=num_derivatives)
        scales, powers = _tmp
        return cls(
            a=a,
            q_sqrtm_lower=q_sqrtm,
            preconditioner_scales=scales,
            preconditioner_powers=powers,
        )

    @property
    def num_derivatives(self):
        return self.a.shape[0] - 1

    def init_hidden_state(self, taylor_coefficients):
        if len(taylor_coefficients) != self.num_derivatives + 1:
            msg1 = "The number of Taylor coefficients does not match "
            msg2 = "the number of derivatives in the implementation."
            raise ValueError(msg1 + msg2)

        m0_matrix = jnp.stack(taylor_coefficients)
        m0_corrected = jnp.reshape(m0_matrix, (-1,), order="F")
        c_sqrtm0_corrected = jnp.zeros_like(self.q_sqrtm_lower)

        rv = _vars.NormalHiddenState(
            mean=m0_corrected, cov_sqrtm_lower=c_sqrtm0_corrected
        )
        return _vars.StateSpaceVar(rv)

    def init_error_estimate(self):
        return jnp.zeros(())

    def begin_extrapolation(self, p0, /, dt):
        p, p_inv = self._assemble_preconditioner(dt=dt)
        m0_p = p_inv * p0.hidden_state.mean
        m_ext_p = self.a @ m0_p
        m_ext = p * m_ext_p
        q_sqrtm = p[:, None] * self.q_sqrtm_lower
        extrapolated = _vars.NormalHiddenState(m_ext, q_sqrtm)
        return _vars.StateSpaceVar(extrapolated), (m_ext_p, m0_p, p, p_inv)

    def _assemble_preconditioner(self, dt):
        return _ibm_util.preconditioner_diagonal(
            dt=dt, scales=self.preconditioner_scales, powers=self.preconditioner_powers
        )

    def complete_extrapolation(self, linearisation_pt, p0, cache, output_scale_sqrtm):
        _, _, p, p_inv = cache
        m_ext = linearisation_pt.hidden_state.mean
        l_ext_p = _sqrt_util.sum_of_sqrtm_factors(
            R_stack=(
                (self.a @ (p_inv[:, None] * p0.hidden_state.cov_sqrtm_lower)).T,
                (output_scale_sqrtm * self.q_sqrtm_lower).T,
            )
        ).T
        l_ext = p[:, None] * l_ext_p
        return _vars.StateSpaceVar(
            _vars.NormalHiddenState(mean=m_ext, cov_sqrtm_lower=l_ext)
        )

    def revert_markov_kernel(self, linearisation_pt, p0, cache, output_scale_sqrtm):
        m_ext_p, m0_p, p, p_inv = cache
        m_ext = linearisation_pt.hidden_state.mean

        l0_p = p_inv[:, None] * p0.hidden_state.cov_sqrtm_lower
        r_ext_p, (r_bw_p, g_bw_p) = _sqrt_util.revert_conditional(
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
        m_bw = p * m_bw_p
        l_bw = p[:, None] * l_bw_p
        g_bw = p[:, None] * g_bw_p * p_inv[None, :]

        backward_noise = _vars.NormalHiddenState(mean=m_bw, cov_sqrtm_lower=l_bw)
        bw_model = _conds.ConditionalHiddenState(g_bw, noise=backward_noise)
        extrapolated = _vars.NormalHiddenState(mean=m_ext, cov_sqrtm_lower=l_ext)
        return _vars.StateSpaceVar(extrapolated), bw_model

    def init_conditional(self, ssv_proto):
        op = self._init_backward_transition()
        noi = self._init_backward_noise(rv_proto=ssv_proto.hidden_state)
        return _conds.ConditionalHiddenState(op, noise=noi)

    def _init_backward_transition(self):
        k = self.num_derivatives + 1
        return jnp.eye(k)

    @staticmethod
    def _init_backward_noise(rv_proto):
        mean = jnp.zeros_like(rv_proto.mean)
        cov_sqrtm_lower = jnp.zeros_like(rv_proto.cov_sqrtm_lower)
        return _vars.NormalHiddenState(mean, cov_sqrtm_lower)

    def init_output_scale_sqrtm(self):
        return 1.0
