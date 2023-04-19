"""Extrapolations."""

from typing import Any, Tuple

import jax
import jax.numpy as jnp

from probdiffeq import _collections, _sqrt_util
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
class _IsoIBM(_extra.Extrapolation[_vars.IsoSSV, Any]):
    def __repr__(self):
        args2 = f"num_derivatives={self.num_derivatives}"
        return f"<Isotropic IBM with {args2}>"

    @property
    def num_derivatives(self):
        return self.a.shape[0] - 1

    def filter_solution_from_tcoeffs(self, taylor_coefficients, /):
        if len(taylor_coefficients) != self.num_derivatives + 1:
            msg1 = "The number of Taylor coefficients does not match "
            msg2 = "the number of derivatives in the implementation."
            raise ValueError(msg1 + msg2)
        m0_corrected = jnp.stack(taylor_coefficients)
        c_sqrtm0_corrected = jnp.zeros_like(self.q_sqrtm_lower)
        rv = _vars.IsoNormalHiddenState(
            mean=m0_corrected, cov_sqrtm_lower=c_sqrtm0_corrected
        )
        return rv

    def smoother_solution_from_tcoeffs(self, taylor_coefficients, /):
        if len(taylor_coefficients) != self.num_derivatives + 1:
            msg1 = "The number of Taylor coefficients does not match "
            msg2 = "the number of derivatives in the implementation."
            raise ValueError(msg1 + msg2)
        m0_corrected = jnp.stack(taylor_coefficients)
        c_sqrtm0_corrected = jnp.zeros_like(self.q_sqrtm_lower)
        rv = _vars.IsoNormalHiddenState(
            mean=m0_corrected, cov_sqrtm_lower=c_sqrtm0_corrected
        )
        cond = self.smoother_init_conditional(rv_proto=rv)
        return _collections.MarkovSequence(init=rv, backward_model=cond)

    def filter_init(self, rv, /):
        ssv = _vars.IsoSSV(rv)
        cache = None
        return ssv, cache

    def filter_extract(self, ssv, ex, /):
        return ssv.hidden_state

    def smoother_init(self, sol, /):
        ssv = _vars.IsoSSV(sol.init)
        cache = sol.backward_model
        return ssv, cache

    def smoother_extract(self, ssv, ex, /):
        return _collections.MarkovSequence(init=ssv.hidden_state, backward_model=ex)

    def standard_normal(self, ode_shape):
        # Used for Runge-Kutta initialisation.
        assert len(ode_shape) == 1
        (d,) = ode_shape
        m0 = jnp.zeros((self.num_derivatives + 1, d))
        c0 = jnp.eye(self.num_derivatives + 1)
        return _vars.IsoNormalHiddenState(m0, c0)

    # Unnecessary?
    def init_error_estimate(self):
        return jnp.zeros(())  # the initialisation is error-free

    def promote_output_scale(self, output_scale):
        return output_scale

    def extract_output_scale(self, output_scale):
        if output_scale.ndim > 0:
            return output_scale[-1]
        return output_scale

    def filter_begin(self, s0: _vars.IsoSSV, ex0, /, dt) -> Tuple[_vars.IsoSSV, Any]:
        p, p_inv = self._assemble_preconditioner(dt=dt)
        m0_p = p_inv[:, None] * s0.hidden_state.mean
        m_ext_p = self.a @ m0_p
        m_ext = p[:, None] * m_ext_p
        q_sqrtm = p[:, None] * self.q_sqrtm_lower
        l0 = s0.hidden_state.cov_sqrtm_lower

        ext = _vars.IsoNormalHiddenState(m_ext, q_sqrtm)
        ssv = _vars.IsoSSV(ext)
        cache = (m_ext_p, m0_p, p, p_inv, l0)
        return ssv, cache

    def smoother_begin(self, s0: _vars.IsoSSV, ex0, /, dt) -> Tuple[_vars.IsoSSV, Any]:
        p, p_inv = self._assemble_preconditioner(dt=dt)
        m0_p = p_inv[:, None] * s0.hidden_state.mean
        m_ext_p = self.a @ m0_p
        m_ext = p[:, None] * m_ext_p
        q_sqrtm = p[:, None] * self.q_sqrtm_lower
        l0 = s0.hidden_state.cov_sqrtm_lower

        ext = _vars.IsoNormalHiddenState(m_ext, q_sqrtm)
        ssv = _vars.IsoSSV(ext)
        cache = (ex0, m_ext_p, m0_p, p, p_inv, l0)
        return ssv, cache

    def _assemble_preconditioner(self, dt):
        return _ibm_util.preconditioner_diagonal(
            dt=dt, scales=self.preconditioner_scales, powers=self.preconditioner_powers
        )

    def filter_complete(self, st, ex, /, output_scale):
        _, _, p, p_inv, l0 = ex
        m_ext = st.hidden_state.mean

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
        return ssv, None

    def smoother_complete(self, ssv, extra, /, output_scale):
        _, m_ext_p, m0_p, p, p_inv, l0 = extra
        m_ext = ssv.hidden_state.mean

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

        backward_noise = _vars.IsoNormalHiddenState(mean=m_bw, cov_sqrtm_lower=l_bw)
        bw_model = _conds.IsoConditionalHiddenState(g_bw, noise=backward_noise)
        extrapolated = _vars.IsoNormalHiddenState(mean=m_ext, cov_sqrtm_lower=l_ext)
        return _vars.IsoSSV(extrapolated), bw_model

    # todo: should this be a classmethod in _conds.IsoConditional?
    def smoother_init_conditional(self, rv_proto):
        op = self._init_backward_transition()
        noi = self._init_backward_noise(rv_proto=rv_proto)
        return _conds.IsoConditionalHiddenState(op, noise=noi)

    def _init_backward_transition(self):
        return jnp.eye(*self.a.shape)

    def _init_backward_noise(self, rv_proto):
        return _vars.IsoNormalHiddenState(
            mean=jnp.zeros_like(rv_proto.mean),
            cov_sqrtm_lower=jnp.zeros_like(rv_proto.cov_sqrtm_lower),
        )
