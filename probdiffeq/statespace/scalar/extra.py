"""Implementations for scalar initial value problems."""
import jax
import jax.numpy as jnp

from probdiffeq import _markov, _sqrt_util
from probdiffeq.statespace import _extra, _ibm_util
from probdiffeq.statespace.scalar import variables


def ibm_scalar(num_derivatives):
    a, q_sqrtm = _ibm_util.system_matrices_1d(num_derivatives=num_derivatives)
    precon = _ibm_util.preconditioner_prepare(num_derivatives=num_derivatives)
    dynamic = (a, q_sqrtm, precon)
    static = {}
    return _extra.ExtrapolationBundle(_IBMFi, _IBMSm, _IBMFp, *dynamic, **static)


class _IBMFi(_extra.Extrapolation):
    def __repr__(self):
        args2 = f"num_derivatives={self.num_derivatives}"
        return f"<IBM with {args2}>"

    @property
    def num_derivatives(self):
        return self.a.shape[0] - 1

    def solution_from_tcoeffs(self, taylor_coefficients, /):
        m0_corrected, c_sqrtm0_corrected = self._stack_tcoeffs(taylor_coefficients)
        return variables.NormalHiddenState(
            mean=m0_corrected, cov_sqrtm_lower=c_sqrtm0_corrected
        )

    def _stack_tcoeffs(self, taylor_coefficients):
        if len(taylor_coefficients) != self.num_derivatives + 1:
            msg1 = "The number of Taylor coefficients does not match "
            msg2 = "the number of derivatives in the implementation."
            raise ValueError(msg1 + msg2)
        m0_matrix = jnp.stack(taylor_coefficients)
        m0_corrected = jnp.reshape(m0_matrix, (-1,), order="F")
        c_sqrtm0_corrected = jnp.zeros_like(self.q_sqrtm_lower)
        return m0_corrected, c_sqrtm0_corrected

    def init(self, sol: variables.NormalHiddenState, /):
        return variables.SSV(sol.mean[0], sol), None

    def extract(self, ssv, extra, /):
        return ssv.hidden_state

    def begin(self, ssv, extra, /, dt):
        p, p_inv = self.preconditioner(dt=dt)
        m0_p = p_inv * ssv.hidden_state.mean
        l0 = ssv.hidden_state.cov_sqrtm_lower
        m_ext_p = self.a @ m0_p
        m_ext = p * m_ext_p
        q_sqrtm = p[:, None] * self.q_sqrtm_lower
        extrapolated = variables.NormalHiddenState(m_ext, q_sqrtm)
        return variables.SSV(m_ext[0], extrapolated), (p, p_inv, l0)

    def complete(self, ssv, extra, /, output_scale):
        p, p_inv, l0 = extra
        m_ext = ssv.hidden_state.mean
        l_ext_p = _sqrt_util.sum_of_sqrtm_factors(
            R_stack=(
                (self.a @ (p_inv[:, None] * l0)).T,
                (output_scale * self.q_sqrtm_lower).T,
            )
        ).T
        l_ext = p[:, None] * l_ext_p

        rv = variables.NormalHiddenState(mean=m_ext, cov_sqrtm_lower=l_ext)
        ssv = variables.SSV(m_ext[0], rv)
        return ssv, None

    def promote_output_scale(self, output_scale):
        return output_scale

    def extract_output_scale(self, output_scale):
        if output_scale.ndim > 0:
            return output_scale[-1]
        return output_scale


class _IBMSm(_extra.Extrapolation):
    def __repr__(self):
        args2 = f"num_derivatives={self.num_derivatives}"
        return f"<IBM with {args2}>"

    @property
    def num_derivatives(self):
        return self.a.shape[0] - 1

    def solution_from_tcoeffs(self, taylor_coefficients, /):
        m0_corrected, c_sqrtm0_corrected = self._stack_tcoeffs(taylor_coefficients)
        rv = variables.NormalHiddenState(
            mean=m0_corrected, cov_sqrtm_lower=c_sqrtm0_corrected
        )
        cond = self._init_conditional(rv_proto=rv)
        return _markov.MarkovSequence(init=rv, backward_model=cond)

    def _stack_tcoeffs(self, taylor_coefficients):
        if len(taylor_coefficients) != self.num_derivatives + 1:
            msg1 = "The number of Taylor coefficients does not match "
            msg2 = "the number of derivatives in the implementation."
            raise ValueError(msg1 + msg2)
        m0_matrix = jnp.stack(taylor_coefficients)
        m0_corrected = jnp.reshape(m0_matrix, (-1,), order="F")
        c_sqrtm0_corrected = jnp.zeros_like(self.q_sqrtm_lower)
        return m0_corrected, c_sqrtm0_corrected

    def init(self, sol: _markov.MarkovSequence, /):
        ssv = variables.SSV(sol.init.mean[0], sol.init)
        extra = sol.backward_model
        return ssv, extra

    def extract(self, ssv, extra, /):
        return _markov.MarkovSequence(init=ssv.hidden_state, backward_model=extra)

    def begin(self, ssv, extra, /, dt):
        p, p_inv = self.preconditioner(dt=dt)
        m0_p = p_inv * ssv.hidden_state.mean
        l0 = ssv.hidden_state.cov_sqrtm_lower
        m_ext_p = self.a @ m0_p
        m_ext = p * m_ext_p
        q_sqrtm = p[:, None] * self.q_sqrtm_lower
        extrapolated = variables.NormalHiddenState(m_ext, q_sqrtm)
        ssv = variables.SSV(m_ext[0], extrapolated)
        cache = (extra, m_ext_p, m0_p, p, p_inv, l0)
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
        m_bw = p * m_bw_p
        l_bw = p[:, None] * l_bw_p
        g_bw = p[:, None] * g_bw_p * p_inv[None, :]

        backward_noise = variables.NormalHiddenState(mean=m_bw, cov_sqrtm_lower=l_bw)
        bw_model = variables.ConditionalHiddenState(g_bw, noise=backward_noise)
        extrapolated = variables.NormalHiddenState(mean=m_ext, cov_sqrtm_lower=l_ext)
        ssv = variables.SSV(m_ext[0], extrapolated)
        return ssv, bw_model

    def _init_conditional(self, rv_proto):
        op = self._init_backward_transition()
        noi = self._init_backward_noise(rv_proto=rv_proto)
        return variables.ConditionalHiddenState(op, noise=noi)

    def _init_backward_transition(self):
        k = self.num_derivatives + 1
        return jnp.eye(k)

    @staticmethod
    def _init_backward_noise(rv_proto):
        mean = jnp.zeros_like(rv_proto.mean)
        cov_sqrtm_lower = jnp.zeros_like(rv_proto.cov_sqrtm_lower)
        return variables.NormalHiddenState(mean, cov_sqrtm_lower)

    def promote_output_scale(self, output_scale):
        return output_scale

    def extract_output_scale(self, output_scale):
        if output_scale.ndim > 0:
            return output_scale[-1]
        return output_scale


class _IBMFp(_extra.Extrapolation):
    def __repr__(self):
        args2 = f"num_derivatives={self.num_derivatives}"
        return f"<IBM with {args2}>"

    @property
    def num_derivatives(self):
        return self.a.shape[0] - 1

    def solution_from_tcoeffs(self, taylor_coefficients, /):
        m0_corrected, c_sqrtm0_corrected = self._stack_tcoeffs(taylor_coefficients)
        rv = variables.NormalHiddenState(
            mean=m0_corrected, cov_sqrtm_lower=c_sqrtm0_corrected
        )
        cond = self._init_conditional(rv_proto=rv)
        return _markov.MarkovSequence(init=rv, backward_model=cond)

    def _stack_tcoeffs(self, taylor_coefficients):
        if len(taylor_coefficients) != self.num_derivatives + 1:
            msg1 = "The number of Taylor coefficients does not match "
            msg2 = "the number of derivatives in the implementation."
            raise ValueError(msg1 + msg2)
        m0_matrix = jnp.stack(taylor_coefficients)
        m0_corrected = jnp.reshape(m0_matrix, (-1,), order="F")
        c_sqrtm0_corrected = jnp.zeros_like(self.q_sqrtm_lower)
        return m0_corrected, c_sqrtm0_corrected

    def init(self, sol: _markov.MarkovSequence, /):
        ssv = variables.SSV(sol.init.mean[0], sol.init)
        extra = self._init_conditional(rv_proto=sol.init)
        return ssv, extra

    def extract(self, ssv, extra, /):
        return _markov.MarkovSequence(init=ssv.hidden_state, backward_model=extra)

    def begin(self, ssv, extra, /, dt):
        p, p_inv = self.preconditioner(dt=dt)
        m0_p = p_inv * ssv.hidden_state.mean
        l0 = ssv.hidden_state.cov_sqrtm_lower
        m_ext_p = self.a @ m0_p
        m_ext = p * m_ext_p
        q_sqrtm = p[:, None] * self.q_sqrtm_lower
        extrapolated = variables.NormalHiddenState(m_ext, q_sqrtm)
        ssv = variables.SSV(m_ext[0], extrapolated)
        cache = (m_ext_p, m0_p, p, p_inv, l0, extra)
        return ssv, cache

    def complete(self, ssv, extra, /, output_scale):
        m_ext_p, m0_p, p, p_inv, l0, bw0 = extra
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
        m_bw = p * m_bw_p
        l_bw = p[:, None] * l_bw_p
        g_bw = p[:, None] * g_bw_p * p_inv[None, :]

        backward_noise = variables.NormalHiddenState(mean=m_bw, cov_sqrtm_lower=l_bw)
        bw_model = variables.ConditionalHiddenState(g_bw, noise=backward_noise)
        bw_model = variables.merge_conditionals(bw0, bw_model)

        extrapolated = variables.NormalHiddenState(mean=m_ext, cov_sqrtm_lower=l_ext)
        ssv = variables.SSV(m_ext[0], extrapolated)
        return ssv, bw_model

    def _init_conditional(self, rv_proto):
        op = self._init_backward_transition()
        noi = self._init_backward_noise(rv_proto=rv_proto)
        return variables.ConditionalHiddenState(op, noise=noi)

    def _init_backward_transition(self):
        k = self.num_derivatives + 1
        return jnp.eye(k)

    @staticmethod
    def _init_backward_noise(rv_proto):
        mean = jnp.zeros_like(rv_proto.mean)
        cov_sqrtm_lower = jnp.zeros_like(rv_proto.cov_sqrtm_lower)
        return variables.NormalHiddenState(mean, cov_sqrtm_lower)

    def promote_output_scale(self, output_scale):
        return output_scale

    def extract_output_scale(self, output_scale):
        if output_scale.ndim > 0:
            return output_scale[-1]
        return output_scale

    def reset(self, ssv, extra, /):
        return ssv, self._init_conditional(extra.noise)


# Register scalar extrapolations as pytrees because we want to vmap them
# for block-diagonal models.
# todo: this feels very temporary...


def _flatten(fi):
    child = fi.a, fi.q_sqrtm_lower, fi.preconditioner
    aux = ()
    return child, aux


def _fi_unflatten(_aux, children):
    return _IBMFi(*children)


def _sm_unflatten(_aux, children):
    return _IBMSm(*children)


def _fp_unflatten(_aux, children):
    return _IBMFp(*children)


jax.tree_util.register_pytree_node(_IBMFi, _flatten, _fi_unflatten)
jax.tree_util.register_pytree_node(_IBMSm, _flatten, _sm_unflatten)
jax.tree_util.register_pytree_node(_IBMFp, _flatten, _fp_unflatten)
