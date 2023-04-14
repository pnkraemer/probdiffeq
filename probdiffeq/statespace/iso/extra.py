"""Extrapolations."""

import jax
import jax.numpy as jnp

from probdiffeq import _sqrt_util
from probdiffeq.statespace import _collections, _ibm_util
from probdiffeq.statespace.iso import _conds, _vars


def ibm_iso(num_derivatives):
    a, q_sqrtm = _ibm_util.system_matrices_1d(num_derivatives=num_derivatives)
    _tmp = _ibm_util.preconditioner_prepare(num_derivatives=num_derivatives)
    scales, powers = _tmp
    return _IsoIBM(
        a=a,
        q_sqrtm_lower=q_sqrtm,
        preconditioner_scales=scales,
        preconditioner_powers=powers,
    )


@jax.tree_util.register_pytree_node_class
class _IsoIBM(_collections.AbstractExtrapolation):
    def __repr__(self):
        args2 = f"num_derivatives={self.num_derivatives}"
        return f"<Isotropic IBM with {args2}>"

    @property
    def num_derivatives(self):
        return self.a.shape[0] - 1

    def solution_from_tcoeffs_without_reversal(self, taylor_coefficients, /):
        m0_corrected, c_sqrtm0_corrected = self._stack_tcoeffs(taylor_coefficients)
        return _vars.IsoNormalHiddenState(
            mean=m0_corrected, cov_sqrtm_lower=c_sqrtm0_corrected
        )

    def _stack_tcoeffs(self, taylor_coefficients):
        if len(taylor_coefficients) != self.num_derivatives + 1:
            msg1 = "The number of Taylor coefficients does not match "
            msg2 = "the number of derivatives in the implementation."
            raise ValueError(msg1 + msg2)
        m0_corrected = jnp.stack(taylor_coefficients)
        c_sqrtm0_corrected = jnp.zeros_like(self.q_sqrtm_lower)
        return m0_corrected, c_sqrtm0_corrected

    def solution_from_tcoeffs_with_reversal(self, taylor_coefficients, /):
        m0_corrected, c_sqrtm0_corrected = self._stack_tcoeffs(taylor_coefficients)
        rv = _vars.IsoNormalHiddenState(
            mean=m0_corrected, cov_sqrtm_lower=c_sqrtm0_corrected
        )
        cond = self._init_conditional(rv_proto=rv)
        return rv, cond

    def _init_conditional(self, rv_proto):
        op = jnp.eye(*self.a.shape)
        noi = _vars.IsoNormalHiddenState(
            mean=jnp.zeros_like(rv_proto.mean),
            cov_sqrtm_lower=jnp.zeros_like(rv_proto.cov_sqrtm_lower),
        )
        return _conds.IsoConditionalHiddenState(op, noise=noi)

    def extract_with_reversal(self, s, /):
        return s.hidden_state, s.backward_model

    def extract_without_reversal(self, s, /):
        return s.hidden_state

    def init_without_reversal(self, rv, /):
        observed = _vars.IsoNormalQOI(
            mean=jnp.zeros_like(rv.mean[..., 0, :]),
            cov_sqrtm_lower=jnp.zeros_like(rv.cov_sqrtm_lower[..., 0, 0]),
        )

        error_estimate = jnp.empty(())
        output_scale_dynamic = jnp.empty(())

        # Prepare caches
        m_like = jnp.empty(rv.mean.shape)
        p_like = m_like[..., 0]
        cache_extra = (m_like, m_like, p_like, p_like)
        return _vars.IsoStateSpaceVar(
            rv,
            # A bunch of caches that are filled at some point:
            observed_state=observed,
            output_scale_dynamic=output_scale_dynamic,
            error_estimate=error_estimate,
            cache_extra=cache_extra,
            cache_corr=None,
            backward_model=None,
        )

    def init_with_reversal(self, rv, conds, /):
        observed = _vars.IsoNormalQOI(
            mean=jnp.zeros_like(rv.mean[..., 0, :]),
            cov_sqrtm_lower=jnp.zeros_like(rv.cov_sqrtm_lower[..., 0, 0]),
        )

        error_estimate = jnp.empty(())
        output_scale_dynamic = jnp.empty(())

        # Prepare caches
        m_like = jnp.empty(rv.mean.shape)
        p_like = m_like[..., 0]
        cache_extra = (m_like, m_like, p_like, p_like)
        return _vars.IsoStateSpaceVar(
            rv,
            backward_model=conds,
            # A bunch of caches that are filled at some point:
            observed_state=observed,
            output_scale_dynamic=output_scale_dynamic,
            error_estimate=error_estimate,
            cache_extra=cache_extra,
            cache_corr=None,
        )

    # todo: why does this method have the same name as the above?
    def _init_ssv(self, ode_shape):
        assert len(ode_shape) == 1
        (d,) = ode_shape
        m0 = jnp.zeros((self.num_derivatives + 1, d))
        c0 = jnp.eye(self.num_derivatives + 1)
        rv = _vars.IsoNormalHiddenState(m0, c0)
        return _vars.IsoStateSpaceVar(rv, cache=None)

    def promote_output_scale(self, output_scale):
        return output_scale

    def begin(self, s0: _vars.IsoStateSpaceVar, /, dt) -> _vars.IsoStateSpaceVar:
        p, p_inv = self._assemble_preconditioner(dt=dt)
        m0_p = p_inv[:, None] * s0.hidden_state.mean
        m_ext_p = self.a @ m0_p
        m_ext = p[:, None] * m_ext_p
        q_sqrtm = p[:, None] * self.q_sqrtm_lower

        ext = _vars.IsoNormalHiddenState(m_ext, q_sqrtm)
        return _vars.IsoStateSpaceVar(
            ext,  # NEW!!
            observed_state=s0.observed_state,  # irrelevant
            output_scale_dynamic=s0.output_scale_dynamic,  # irrelevant
            error_estimate=s0.error_estimate,  # irrelevant
            cache_extra=(m_ext_p, m0_p, p, p_inv),  # NEW!!
            cache_corr=s0.cache_corr,  # irrelevant
            backward_model=s0.backward_model,  # None or will-be-overwritten-later
        )

    def _assemble_preconditioner(self, dt):
        return _ibm_util.preconditioner_diagonal(
            dt=dt, scales=self.preconditioner_scales, powers=self.preconditioner_powers
        )

    def complete_without_reversal(
        self,
        state: _vars.IsoStateSpaceVar,
        /,
        state_previous: _vars.IsoStateSpaceVar,
        output_scale: float,
    ) -> _vars.IsoStateSpaceVar:
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
        rv = _vars.IsoNormalHiddenState(m_ext, l_ext)
        # Use output_begin as an error estimate?
        return _vars.IsoStateSpaceVar(
            rv,  # NEW !!
            observed_state=state.observed_state,
            output_scale_dynamic=state.output_scale_dynamic,
            error_estimate=state.error_estimate,
            cache_corr=state.cache_corr,
            cache_extra=state.cache_extra,  # irrelevant
            backward_model=None,
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
        m_bw = p[:, None] * m_bw_p
        l_bw = p[:, None] * l_bw_p
        g_bw = p[:, None] * g_bw_p * p_inv[None, :]

        backward_noise = _vars.IsoNormalHiddenState(mean=m_bw, cov_sqrtm_lower=l_bw)
        bw_model = _conds.IsoConditionalHiddenState(g_bw, noise=backward_noise)
        extrapolated = _vars.IsoNormalHiddenState(mean=m_ext, cov_sqrtm_lower=l_ext)
        return _vars.IsoStateSpaceVar(
            extrapolated,
            observed_state=state.observed_state,
            error_estimate=state.error_estimate,
            output_scale_dynamic=state.output_scale_dynamic,
            cache_extra=state.cache_extra,
            cache_corr=state.cache_corr,
            backward_model=bw_model,
        )

    def replace_backward_model(self, s, /, backward_model):
        return _vars.IsoStateSpaceVar(
            s.hidden_state,
            observed_state=s.observed_state,
            error_estimate=s.error_estimate,
            output_scale_dynamic=s.output_scale_dynamic,
            cache_extra=s.cache_extra,
            cache_corr=s.cache_corr,
            backward_model=backward_model,  # new
        )
