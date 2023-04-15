"""Implementations for scalar initial value problems."""

import jax
import jax.numpy as jnp

from probdiffeq import _sqrt_util
from probdiffeq.statespace import _collections, _ibm_util
from probdiffeq.statespace.scalar import _conds, _vars


def ibm_scalar(num_derivatives):
    a, q_sqrtm = _ibm_util.system_matrices_1d(num_derivatives=num_derivatives)
    _tmp = _ibm_util.preconditioner_prepare(num_derivatives=num_derivatives)
    scales, powers = _tmp
    return _IBM(
        a=a,
        q_sqrtm_lower=q_sqrtm,
        preconditioner_scales=scales,
        preconditioner_powers=powers,
    )


@jax.tree_util.register_pytree_node_class
class _IBM(_collections.AbstractExtrapolation):
    def __repr__(self):
        args2 = f"num_derivatives={self.num_derivatives}"
        return f"<IBM with {args2}>"

    @property
    def num_derivatives(self):
        return self.a.shape[0] - 1

    def solution_from_tcoeffs_without_reversal(self, taylor_coefficients, /):
        m0_corrected, c_sqrtm0_corrected = self._stack_tcoeffs(taylor_coefficients)
        return _vars.NormalHiddenState(
            mean=m0_corrected, cov_sqrtm_lower=c_sqrtm0_corrected
        )

    def solution_from_tcoeffs_with_reversal(self, taylor_coefficients, /):
        m0_corrected, c_sqrtm0_corrected = self._stack_tcoeffs(taylor_coefficients)
        rv = _vars.NormalHiddenState(
            mean=m0_corrected, cov_sqrtm_lower=c_sqrtm0_corrected
        )
        cond = self._init_conditional(rv_proto=rv)
        return rv, cond

    def _stack_tcoeffs(self, taylor_coefficients):
        if len(taylor_coefficients) != self.num_derivatives + 1:
            msg1 = "The number of Taylor coefficients does not match "
            msg2 = "the number of derivatives in the implementation."
            raise ValueError(msg1 + msg2)
        m0_matrix = jnp.stack(taylor_coefficients)
        m0_corrected = jnp.reshape(m0_matrix, (-1,), order="F")
        c_sqrtm0_corrected = jnp.zeros_like(self.q_sqrtm_lower)
        return m0_corrected, c_sqrtm0_corrected

    def init_with_reversal(self, rv, cond, /):
        return _vars.SSV(
            rv,
            backward_model=cond,
            hidden_shape=rv.mean.shape,
            observed_state=None,
            output_scale_dynamic=None,
            error_estimate=None,
            cache_extra=None,
            cache_corr=None,
        )

    def init_with_reversal_and_reset(self, rv, _cond, /):
        cond = self._init_conditional(rv_proto=rv)
        return _vars.SSV(
            rv,
            backward_model=cond,
            hidden_shape=rv.mean.shape,
            observed_state=None,
            output_scale_dynamic=None,
            error_estimate=None,
            cache_extra=None,
            cache_corr=None,
        )

    def _init_conditional(self, rv_proto):
        op = jnp.eye(self.num_derivatives + 1)
        mean = jnp.zeros_like(rv_proto.mean)
        cov_sqrtm_lower = jnp.zeros_like(rv_proto.cov_sqrtm_lower)
        noi = _vars.NormalHiddenState(mean, cov_sqrtm_lower)
        return _conds.ConditionalHiddenState(op, noise=noi)

    def init_without_reversal(self, rv, /):
        # Prepare caches

        return _vars.SSV(
            rv,
            hidden_shape=rv.mean.shape,
            backward_model=None,
            observed_state=None,
            output_scale_dynamic=None,
            error_estimate=None,
            cache_extra=None,
            cache_corr=None,
        )

    def extract_without_reversal(self, s, /):
        return s.hidden_state

    def extract_with_reversal(self, s, /):
        return s.hidden_state, s.backward_model

    def begin(self, s0, /, dt):
        p, p_inv = self._assemble_preconditioner(dt=dt)
        m0_p = p_inv * s0.hidden_state.mean
        m_ext_p = self.a @ m0_p
        m_ext = p * m_ext_p
        q_sqrtm = p[:, None] * self.q_sqrtm_lower
        extrapolated = _vars.NormalHiddenState(m_ext, q_sqrtm)
        return _vars.SSV(
            hidden_state=extrapolated,
            cache_extra=(m_ext_p, m0_p, p, p_inv),
            hidden_shape=s0.hidden_shape,
            backward_model=s0.backward_model,  # irrelevant
            output_scale_dynamic=None,
            cache_corr=None,
            # todo: The below should not be necessary
            #  but currently, it is: because of pytree-shape stability in interpolation
            observed_state=jax.tree_util.tree_map(jnp.zeros_like, s0.observed_state),
            error_estimate=jax.tree_util.tree_map(jnp.zeros_like, s0.error_estimate),
        )

    def _assemble_preconditioner(self, dt):
        return _ibm_util.preconditioner_diagonal(
            dt=dt, scales=self.preconditioner_scales, powers=self.preconditioner_powers
        )

    def complete_without_reversal(self, state, /, state_previous, output_scale):
        _, _, p, p_inv = state.cache_extra
        m_ext = state.hidden_state.mean
        l0 = state_previous.hidden_state.cov_sqrtm_lower
        l0_p = p_inv[:, None] * l0
        l_ext_p = _sqrt_util.sum_of_sqrtm_factors(
            R_stack=(
                (self.a @ l0_p).T,
                (output_scale * self.q_sqrtm_lower).T,
            )
        ).T
        l_ext = p[:, None] * l_ext_p

        rv = _vars.NormalHiddenState(mean=m_ext, cov_sqrtm_lower=l_ext)
        return _vars.SSV(
            rv,
            hidden_shape=state.hidden_shape,
            observed_state=state.observed_state,
            error_estimate=state.error_estimate,
            cache_corr=state.cache_corr,
            backward_model=None,
            output_scale_dynamic=None,
            cache_extra=None,
        )

    def complete_with_reversal(self, state, /, state_previous, output_scale):
        m_ext_p, m0_p, p, p_inv = state.cache_extra
        m_ext = state.hidden_state.mean
        l0_p = p_inv[:, None] * state_previous.hidden_state.cov_sqrtm_lower

        r_ext_p, (r_bw_p, g_bw_p) = _sqrt_util.revert_conditional(
            R_X_F=(self.a @ l0_p).T,
            R_X=l0_p.T,
            R_YX=(output_scale * self.q_sqrtm_lower).T,
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
        rv = _vars.NormalHiddenState(mean=m_ext, cov_sqrtm_lower=l_ext)
        return _vars.SSV(
            rv,
            backward_model=bw_model,
            hidden_shape=state.hidden_shape,
            error_estimate=state.error_estimate,
            cache_corr=state.cache_corr,
            observed_state=state.observed_state,  # usually None?
            output_scale_dynamic=None,
            cache_extra=None,
        )

    def promote_output_scale(self, output_scale):
        return output_scale

    def replace_backward_model(self, s, /, backward_model):
        return _vars.SSV(
            hidden_state=s.hidden_state,
            hidden_shape=s.hidden_shape,
            output_scale_dynamic=s.output_scale_dynamic,
            error_estimate=s.error_estimate,
            observed_state=s.observed_state,
            cache_extra=s.cache_extra,
            cache_corr=s.cache_corr,
            backward_model=backward_model,  # new
        )

    def duplicate_with_unit_backward_model(self, s, /):
        unit_bw_model = self._init_conditional(rv_proto=s.hidden_state)
        return self.replace_backward_model(s, unit_bw_model)
