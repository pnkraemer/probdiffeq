"""Extrapolations."""

from typing import Any, Tuple

import jax.numpy as jnp

from probdiffeq import _markov, _sqrt_util
from probdiffeq.statespace import _extra, _ibm_util
from probdiffeq.statespace.iso import variables


def ibm_iso(num_derivatives):
    a, q_sqrtm = _ibm_util.system_matrices_1d(num_derivatives=num_derivatives)
    precon = _ibm_util.preconditioner_prepare(num_derivatives=num_derivatives)
    dynamic = (a, q_sqrtm, precon)
    static = {}
    return _extra.ExtrapolationBundle(_IBMFi, _IBMSm, _IBMFp, *dynamic, **static)


class _IBMFi(_extra.Extrapolation[variables.IsoSSV, Any]):
    @property
    def num_derivatives(self):
        return self.a.shape[0] - 1

    def __repr__(self):
        args2 = f"num_derivatives={self.num_derivatives}"
        return f"<Isotropic IBM with {args2}>"

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
        p, p_inv = self.preconditioner(dt=dt)
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


class _IBMSm(_extra.Extrapolation[variables.IsoSSV, Any]):
    def __repr__(self):
        args2 = f"num_derivatives={self.num_derivatives}"
        return f"<Isotropic IBM with {args2}>"

    @property
    def num_derivatives(self):
        return self.a.shape[0] - 1

    def solution_from_tcoeffs(self, taylor_coefficients, /):
        m0, c_sqrtm0 = _stack_tcoeffs(taylor_coefficients, q_like=self.q_sqrtm_lower)
        rv = variables.IsoNormalHiddenState(mean=m0, cov_sqrtm_lower=c_sqrtm0)
        cond = variables.identity_conditional(self.num_derivatives, m0.shape[1:])
        return _markov.MarkovSequence(init=rv, backward_model=cond)

    def init(self, sol: _markov.MarkovSequence, /):
        ssv = variables.IsoSSV(sol.init.mean[0, :], sol.init)
        cache = sol.backward_model
        return ssv, cache

    def extract(self, ssv, ex, /):
        return _markov.MarkovSequence(init=ssv.hidden_state, backward_model=ex)

    def begin(self, s0: variables.IsoSSV, ex0, /, dt) -> Tuple[variables.IsoSSV, Any]:
        p, p_inv = self.preconditioner(dt=dt)
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
        cond = variables.identity_conditional(self.num_derivatives, m0.shape[1:])
        return _markov.MarkovSequence(init=rv, backward_model=cond)

    def init(self, sol, /):
        ssv = variables.IsoSSV(sol.init.mean[0, :], sol.init)
        ode_shape = sol.init.mean.shape[1:]
        cond = variables.identity_conditional(self.num_derivatives, ode_shape)
        return ssv, cond

    def extract(self, ssv, ex, /):
        return _markov.MarkovSequence(init=ssv.hidden_state, backward_model=ex)

    def begin(self, s0: variables.IsoSSV, ex0, /, dt) -> Tuple[variables.IsoSSV, Any]:
        p, p_inv = self.preconditioner(dt=dt)
        m0_p = p_inv[:, None] * s0.hidden_state.mean
        m_ext_p = self.a @ m0_p
        m_ext = p[:, None] * m_ext_p
        q_sqrtm = p[:, None] * self.q_sqrtm_lower
        l0 = s0.hidden_state.cov_sqrtm_lower

        ext = variables.IsoNormalHiddenState(m_ext, q_sqrtm)
        ssv = variables.IsoSSV(m_ext[0, :], ext)
        cache = (m_ext_p, m0_p, p, p_inv, l0, ex0)
        return ssv, cache

    def complete(self, ssv, extra, /, output_scale):
        m_ext_p, m0_p, p, p_inv, l0, ex0 = extra
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

        # Merge backward-models
        backward_noise = variables.IsoNormalHiddenState(mean=m_bw, cov_sqrtm_lower=l_bw)
        bw_model = variables.IsoConditionalHiddenState(g_bw, noise=backward_noise)
        bw_model = variables.merge_conditionals(ex0, bw_model)

        extrapolated = variables.IsoNormalHiddenState(mean=m_ext, cov_sqrtm_lower=l_ext)
        return variables.IsoSSV(m_ext[0, :], extrapolated), bw_model

    def reset(self, ssv, _extra):
        ode_shape = ssv.hidden_state.mean.shape[1:]
        cond = variables.identity_conditional(self.num_derivatives, ode_shape)
        return ssv, cond


def _stack_tcoeffs(taylor_coefficients, q_like):
    num_expected = q_like.shape[0]
    if len(taylor_coefficients) != num_expected:
        msg1 = "The number of Taylor coefficients does not match "
        msg2 = "the number of derivatives in the implementation."
        raise ValueError(msg1 + msg2)
    c_sqrtm0_corrected = jnp.zeros_like(q_like)
    m0_corrected = jnp.stack(taylor_coefficients)
    return m0_corrected, c_sqrtm0_corrected
