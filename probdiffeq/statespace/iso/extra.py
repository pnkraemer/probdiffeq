"""Extrapolations."""

from typing import Any, Tuple

import jax
import jax.numpy as jnp

from probdiffeq import _markov, _sqrt_util
from probdiffeq.statespace import _extra, _ibm_util
from probdiffeq.statespace.iso import variables


def ibm_iso(num_derivatives):
    a, q_sqrtm = _ibm_util.system_matrices_1d(num_derivatives=num_derivatives)
    _tmp = _ibm_util.preconditioner_prepare(num_derivatives=num_derivatives)
    scales, powers = _tmp
    dynamic = (a, q_sqrtm, scales, powers)
    static = {}
    return _extra.ExtrapolationBundle(_IBMFi, _IBMSm, _IBMFp, *dynamic, **static)


class _IBMFi(_extra.Extrapolation[variables.IsoSSV, Any]):
    @property
    def num_derivatives(self):
        return self.a.shape[0] - 1

    def __repr__(self):
        args2 = f"num_derivatives={self.num_derivatives}"
        return f"<Isotropic IBM with {args2}>"

    # Content:

    def solution_from_tcoeffs(self, tcoeffs, /):
        m0, c_sqrtm0 = _stack_tcoeffs(tcoeffs, q_like=self.q_sqrtm_lower)
        rv = variables.IsoNormalHiddenState(mean=m0, cov_sqrtm_lower=c_sqrtm0)
        return rv

    def init(self, rv: variables.IsoNormalHiddenState, /):
        ssv = variables.IsoSSV(rv.mean[0, :], rv)
        cache = None
        return ssv, cache

    def begin(
        self, s0: variables.IsoSSV, _extra, /, dt
    ) -> Tuple[variables.IsoSSV, Any]:
        p, p_inv = self._assemble_preconditioner(dt=dt)
        m0_p = p_inv[:, None] * s0.hidden_state.mean
        m_ext_p = self.a @ m0_p
        m_ext = p[:, None] * m_ext_p
        q_sqrtm = p[:, None] * self.q_sqrtm_lower
        l0 = s0.hidden_state.cov_sqrtm_lower

        ext = variables.IsoNormalHiddenState(m_ext, q_sqrtm)
        ssv = variables.IsoSSV(m_ext[0, :], ext)
        cache = (p, p_inv, l0)
        return ssv, cache

    def complete(self, st, ex, /, output_scale):
        p, p_inv, l0 = ex
        m_ext = st.hidden_state.mean

        l0_p = p_inv[:, None] * l0
        l_ext_p = _sqrt_util.sum_of_sqrtm_factors(
            R_stack=(
                (self.a @ l0_p).T,
                (output_scale * self.q_sqrtm_lower).T,
            )
        ).T
        l_ext = p[:, None] * l_ext_p
        rv = variables.IsoNormalHiddenState(m_ext, l_ext)
        ssv = variables.IsoSSV(m_ext[0, :], rv)
        return ssv, None

    def extract(self, ssv, _extra, /):
        return ssv.hidden_state

    # Some helpers:

    def _assemble_preconditioner(self, dt):
        # todo: 'partial' scales and powers into the preconditioner?
        k = {"scales": self.preconditioner_scales, "powers": self.preconditioner_powers}
        return _ibm_util.preconditioner_diagonal(dt=dt, **k)

    def standard_normal(self, ode_shape):
        # Used for Runge-Kutta initialisation.
        assert len(ode_shape) == 1
        (d,) = ode_shape
        m0 = jnp.zeros((self.num_derivatives + 1, d))
        c0 = jnp.eye(self.num_derivatives + 1)
        return variables.IsoNormalHiddenState(m0, c0)

    def promote_output_scale(self, output_scale):
        return output_scale

    def extract_output_scale(self, output_scale):
        if output_scale.ndim > 0:
            return output_scale[-1]
        return output_scale


class _IBMSm(_extra.Extrapolation[variables.IsoSSV, Any]):
    def __repr__(self):
        args2 = f"num_derivatives={self.num_derivatives}"
        return f"<Isotropic IBM with {args2}>"

    @property
    def num_derivatives(self):
        return self.a.shape[0] - 1

    # Actual content:

    def solution_from_tcoeffs(self, taylor_coefficients, /):
        m0, c_sqrtm0 = _stack_tcoeffs(taylor_coefficients, q_like=self.q_sqrtm_lower)
        rv = variables.IsoNormalHiddenState(mean=m0, cov_sqrtm_lower=c_sqrtm0)
        cond = self.init_conditional(rv_proto=rv)
        return _markov.MarkovSequence(init=rv, backward_model=cond)

    def init(self, sol: _markov.MarkovSequence, /):
        ssv = variables.IsoSSV(sol.init.mean[0, :], sol.init)
        cache = sol.backward_model
        return ssv, cache

    def extract(self, ssv, ex, /):
        return _markov.MarkovSequence(init=ssv.hidden_state, backward_model=ex)

    def begin(self, s0: variables.IsoSSV, ex0, /, dt) -> Tuple[variables.IsoSSV, Any]:
        p, p_inv = self._assemble_preconditioner(dt=dt)
        m0_p = p_inv[:, None] * s0.hidden_state.mean
        m_ext_p = self.a @ m0_p
        m_ext = p[:, None] * m_ext_p
        q_sqrtm = p[:, None] * self.q_sqrtm_lower
        l0 = s0.hidden_state.cov_sqrtm_lower

        ext = variables.IsoNormalHiddenState(m_ext, q_sqrtm)
        ssv = variables.IsoSSV(m_ext[0, :], ext)
        cache = (ex0, m_ext_p, m0_p, p, p_inv, l0)
        return ssv, cache

    def complete(self, ssv, extra, /, output_scale):
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

        backward_noise = variables.IsoNormalHiddenState(mean=m_bw, cov_sqrtm_lower=l_bw)
        bw_model = variables.IsoConditionalHiddenState(g_bw, noise=backward_noise)
        extrapolated = variables.IsoNormalHiddenState(mean=m_ext, cov_sqrtm_lower=l_ext)
        return variables.IsoSSV(m_ext[0, :], extrapolated), bw_model

    # Some helpers:

    def _assemble_preconditioner(self, dt):
        return _ibm_util.preconditioner_diagonal(
            dt=dt, scales=self.preconditioner_scales, powers=self.preconditioner_powers
        )

    def standard_normal(self, ode_shape):
        # Used for Runge-Kutta initialisation.
        assert len(ode_shape) == 1
        (d,) = ode_shape
        m0 = jnp.zeros((self.num_derivatives + 1, d))
        c0 = jnp.eye(self.num_derivatives + 1)
        return variables.IsoNormalHiddenState(m0, c0)

    def promote_output_scale(self, output_scale):
        return output_scale

    def extract_output_scale(self, output_scale):
        if output_scale.ndim > 0:
            return output_scale[-1]
        return output_scale

    def init_conditional(self, rv_proto):
        op = jnp.eye(*self.a.shape)
        noi = jax.tree_util.tree_map(jnp.zeros_like, rv_proto)
        return variables.IsoConditionalHiddenState(op, noise=noi)


class _IBMFp(_extra.Extrapolation[variables.IsoSSV, Any]):
    def __repr__(self):
        args2 = f"num_derivatives={self.num_derivatives}"
        return f"<Isotropic IBM with {args2}>"

    @property
    def num_derivatives(self):
        return self.a.shape[0] - 1

    def solution_from_tcoeffs(self, taylor_coefficients, /):
        m0, c_sqrtm0 = _stack_tcoeffs(taylor_coefficients, q_like=self.q_sqrtm_lower)
        rv = variables.IsoNormalHiddenState(mean=m0, cov_sqrtm_lower=c_sqrtm0)
        cond = self.init_conditional(rv_proto=rv)
        return _markov.MarkovSequence(init=rv, backward_model=cond)

    def init(self, sol, /):
        ssv = variables.IsoSSV(sol.init.mean[0, :], sol.init)
        cache = self.init_conditional(rv_proto=sol.init)
        return ssv, cache

    def extract(self, ssv, ex, /):
        return _markov.MarkovSequence(init=ssv.hidden_state, backward_model=ex)

    def standard_normal(self, ode_shape):
        # Used for Runge-Kutta initialisation.
        assert len(ode_shape) == 1
        (d,) = ode_shape
        m0 = jnp.zeros((self.num_derivatives + 1, d))
        c0 = jnp.eye(self.num_derivatives + 1)
        return variables.IsoNormalHiddenState(m0, c0)

    def promote_output_scale(self, output_scale):
        return output_scale

    def extract_output_scale(self, output_scale):
        if output_scale.ndim > 0:
            return output_scale[-1]
        return output_scale

    def begin(self, s0: variables.IsoSSV, ex0, /, dt) -> Tuple[variables.IsoSSV, Any]:
        p, p_inv = self._assemble_preconditioner(dt=dt)
        m0_p = p_inv[:, None] * s0.hidden_state.mean
        m_ext_p = self.a @ m0_p
        m_ext = p[:, None] * m_ext_p
        q_sqrtm = p[:, None] * self.q_sqrtm_lower
        l0 = s0.hidden_state.cov_sqrtm_lower

        ext = variables.IsoNormalHiddenState(m_ext, q_sqrtm)
        ssv = variables.IsoSSV(m_ext[0, :], ext)
        cache = (ex0, m_ext_p, m0_p, p, p_inv, l0)
        return ssv, cache

    def _assemble_preconditioner(self, dt):
        return _ibm_util.preconditioner_diagonal(
            dt=dt, scales=self.preconditioner_scales, powers=self.preconditioner_powers
        )

    def complete(self, ssv, extra, /, output_scale):
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

        backward_noise = variables.IsoNormalHiddenState(mean=m_bw, cov_sqrtm_lower=l_bw)
        bw_model = variables.IsoConditionalHiddenState(g_bw, noise=backward_noise)
        extrapolated = variables.IsoNormalHiddenState(mean=m_ext, cov_sqrtm_lower=l_ext)
        return variables.IsoSSV(m_ext[0, :], extrapolated), bw_model

    def init_conditional(self, rv_proto):
        op = jnp.eye(*self.a.shape)
        noi = jax.tree_util.tree_map(jnp.zeros_like, rv_proto)
        return variables.IsoConditionalHiddenState(op, noise=noi)


def _stack_tcoeffs(taylor_coefficients, q_like):
    num_expected = q_like.shape[0]
    if len(taylor_coefficients) != num_expected:
        msg1 = "The number of Taylor coefficients does not match "
        msg2 = "the number of derivatives in the implementation."
        raise ValueError(msg1 + msg2)
    c_sqrtm0_corrected = jnp.zeros_like(q_like)
    m0_corrected = jnp.stack(taylor_coefficients)
    return m0_corrected, c_sqrtm0_corrected
