"""Extrapolations."""

from typing import Tuple

import jax
import jax.numpy as jnp

from probdiffeq import _sqrt_util
from probdiffeq.statespace import _extra, _ibm_util
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
class _IsoIBM(_extra.Extrapolation):
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

    def extract_with_reversal(self, s, e, /):
        return s.hidden_state, e.backward_model

    def extract_without_reversal(self, s, /):
        return s.hidden_state

    def init_without_reversal(self, rv, /):
        extra = _extra.State(backward_model=None, cache=None)
        ssv = _vars.IsoSSV(rv)
        return ssv, extra

    def init_with_reversal(self, rv, conds, /):
        ssv = _vars.IsoSSV(rv)
        extra = _extra.State(backward_model=conds, cache=None)
        return ssv, extra

    def init_with_reversal_and_reset(self, rv, _conds, /):
        cond = self._init_conditional(rv_proto=rv)
        ssv = _vars.IsoSSV(rv)
        extra = _extra.State(backward_model=cond, cache=None)
        return ssv, extra

    def _init_conditional(self, rv_proto):
        op = jnp.eye(*self.a.shape)
        noi = _vars.IsoNormalHiddenState(
            mean=jnp.zeros_like(rv_proto.mean),
            cov_sqrtm_lower=jnp.zeros_like(rv_proto.cov_sqrtm_lower),
        )
        return _conds.IsoConditionalHiddenState(op, noise=noi)

    # todo: why does this method have the same name as the above?
    def _init_ssv(self, ode_shape):
        assert len(ode_shape) == 1
        (d,) = ode_shape
        m0 = jnp.zeros((self.num_derivatives + 1, d))
        c0 = jnp.eye(self.num_derivatives + 1)
        rv = _vars.IsoNormalHiddenState(m0, c0)
        return _vars.IsoSSV(rv, cache=None)

    def promote_output_scale(self, output_scale):
        return output_scale

    # todo: split into begin_with and begin_without
    def begin(
        self, s0: _vars.IsoSSV, ex: _extra.State, /, dt
    ) -> Tuple[_vars.IsoSSV, _extra.State]:
        p, p_inv = self._assemble_preconditioner(dt=dt)
        m0_p = p_inv[:, None] * s0.hidden_state.mean
        m_ext_p = self.a @ m0_p
        m_ext = p[:, None] * m_ext_p
        q_sqrtm = p[:, None] * self.q_sqrtm_lower

        rv = _vars.IsoNormalHiddenState(m_ext, q_sqrtm)
        ssv = _vars.IsoSSV(rv)

        l0 = s0.hidden_state.cov_sqrtm_lower
        # todo: once we have different begin() methods
        #  (and once fixed-point smoothing does its own complete())
        #  backward_model should be None here.
        ext = _extra.State(
            cache=(m_ext_p, m0_p, p, p_inv, l0), backward_model=ex.backward_model
        )
        return ssv, ext
        # return _vars.IsoSSV(
        #     ext,
        #     cache_extra=,
        #     hidden_shape=s0.hidden_shape,
        #     output_scale_dynamic=None,
        #     cache_corr=None,
        #     backward_model=None,
        #     # todo: The below should not be necessary
        #     #  but currently, it is: because of pytree-shape stability in interpolation
        #     observed_state=jax.tree_util.tree_map(jnp.zeros_like, s0.observed_state),
        #     error_estimate=jax.tree_util.tree_map(jnp.zeros_like, s0.error_estimate),
        # )

    def _assemble_preconditioner(self, dt):
        return _ibm_util.preconditioner_diagonal(
            dt=dt, scales=self.preconditioner_scales, powers=self.preconditioner_powers
        )

    def complete_without_reversal(
        self,
        state: _vars.IsoSSV,
        extra: _extra.State,
        /,
        output_scale: float,
    ) -> _vars.IsoSSV:
        m_ext = state.hidden_state.mean
        _, _, p, p_inv, l0 = extra.cache

        l0_p = p_inv[:, None] * l0
        l_ext_p = _sqrt_util.sum_of_sqrtm_factors(
            R_stack=(
                (self.a @ l0_p).T,
                (output_scale * self.q_sqrtm_lower).T,
            )
        ).T
        l_ext = p[:, None] * l_ext_p
        rv = _vars.IsoNormalHiddenState(m_ext, l_ext)
        ssv = _vars.IsoSSV(rv)
        extra = _extra.State(backward_model=None, cache=None)
        return ssv, extra
        # return _vars.IsoSSV(
        #     rv,
        #     hidden_shape=state.hidden_shape,
        #     error_estimate=state.error_estimate,
        #     cache_corr=state.cache_corr,
        #     observed_state=state.observed_state,  # usually None?
        #     output_scale_dynamic=None,
        #     cache_extra=None,
        #     backward_model=None,
        # )

    def complete_with_reversal(self, state, extra, /, output_scale):
        m_ext_p, m0_p, p, p_inv, l0 = extra.cache
        m_ext = state.hidden_state.mean

        l0_p = p_inv[:, None] * l0
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

        # Gather outputs
        backward_noise = _vars.IsoNormalHiddenState(mean=m_bw, cov_sqrtm_lower=l_bw)
        bw_model = _conds.IsoConditionalHiddenState(g_bw, noise=backward_noise)
        extrapolated = _vars.IsoNormalHiddenState(mean=m_ext, cov_sqrtm_lower=l_ext)

        # Return results
        ssv = _vars.IsoSSV(extrapolated)
        extra = _extra.State(backward_model=bw_model, cache=None)
        return ssv, extra

    def duplicate_with_unit_backward_model(self, e, /):
        unit_bw_model = self._init_conditional(rv_proto=e.backward_model.noise)
        return self.replace_backward_model(e, unit_bw_model)

    def replace_backward_model(self, e, /, backward_model):
        extra = _extra.State(backward_model=backward_model, cache=e.cache)
        return extra

    #
    #
    #     return _vars.IsoSSV(
    #         s.hidden_state,
    #         backward_model=backward_model,  # new
    #         hidden_shape=s.hidden_shape,
    #         observed_state=s.observed_state,
    #         error_estimate=s.error_estimate,
    #         output_scale_dynamic=s.output_scale_dynamic,
    #         cache_extra=s.cache_extra,
    #         cache_corr=s.cache_corr,
    #     )
