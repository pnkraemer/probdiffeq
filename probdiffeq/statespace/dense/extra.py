"""Extrapolations."""

import jax
import jax.numpy as jnp

from probdiffeq import _markov, _sqrt_util
from probdiffeq.statespace import _extra, _ibm_util
from probdiffeq.statespace.dense import variables


def ibm_dense(ode_shape, num_derivatives):
    assert len(ode_shape) == 1
    (d,) = ode_shape

    a, q_sqrtm = _ibm_util.system_matrices_1d(num_derivatives=num_derivatives)
    precon_1d = _ibm_util.preconditioner_prepare(num_derivatives=num_derivatives)

    @jax.tree_util.Partial
    def preconditioner(*args, **kwargs):
        p1, p2 = precon_1d(*args, **kwargs)
        p1 = jnp.tile(p1, d)
        p2 = jnp.tile(p2, d)
        return p1, p2

    eye_d = jnp.eye(d)
    params = (jnp.kron(eye_d, a), jnp.kron(eye_d, q_sqrtm), preconditioner)

    factory = _DenseExtrapolationFactory(ode_shape, num_derivatives=num_derivatives)
    return factory, params


class _DenseExtrapolationFactory(_extra.ExtrapolationFactory):
    def __init__(self, ode_shape, *, num_derivatives):
        self.kwargs = {"num_derivatives": num_derivatives, "ode_shape": ode_shape}

    def filter(self, *params):
        return _IBMFi(*params, **self.kwargs)

    def smoother(self, *params):
        return _IBMSm(*params, **self.kwargs)

    def fixedpoint(self, *params):
        return _IBMFp(*params, **self.kwargs)


class _IBMFi(_extra.Extrapolation):
    def __init__(self, *args, num_derivatives, ode_shape):
        super().__init__(*args)

        self.num_derivatives = num_derivatives
        assert len(ode_shape) == 1
        self.ode_shape = ode_shape

    @property
    def target_shape(self):
        return (self.num_derivatives + 1,) + self.ode_shape

    def __repr__(self):
        args2 = f"num_derivatives={self.num_derivatives}"
        args3 = f"ode_shape={self.ode_shape}"
        return f"<Dense IBM with {args2}, {args3}>"

    def solution_from_tcoeffs(self, tcoeffs, /):
        m0, c_sqrtm0 = _stack_tcoeffs(tcoeffs, target_shape=self.target_shape)
        return variables.DenseNormal(
            mean=m0, cov_sqrtm_lower=c_sqrtm0, target_shape=self.target_shape
        )

    def init(self, sol: variables.DenseNormal, /):
        u = sol.mean.reshape(self.target_shape, order="F")[0, :]
        ssv = variables.DenseSSV(u, sol, target_shape=self.target_shape)
        extra = None
        return ssv, extra

    def extract(self, ssv, _extra, /):
        return ssv.hidden_state

    def begin(self, ssv, extra, /, dt):
        p, p_inv = self.preconditioner(dt=dt)
        m0_p = p_inv * ssv.hidden_state.mean
        m_ext_p = self.a @ m0_p
        m_ext = p * m_ext_p
        q_sqrtm = p[:, None] * self.q_sqrtm_lower

        l0 = ssv.hidden_state.cov_sqrtm_lower

        (d,) = self.ode_shape
        shape = (self.num_derivatives + 1, d)
        ext = variables.DenseNormal(m_ext, q_sqrtm, target_shape=shape)
        cache = (p, p_inv, l0)
        u = m_ext.reshape(self.target_shape, order="F")[0, :]
        ssv = variables.DenseSSV(u, ext, target_shape=shape)
        return ssv, cache

    def complete(self, ssv, extra, /, output_scale):
        p, p_inv, l0 = extra
        m_ext = ssv.hidden_state.mean
        l0_p = p_inv[:, None] * l0
        r_stack = ((self.a @ l0_p).T, (output_scale * self.q_sqrtm_lower).T)
        l_ext_p = _sqrt_util.sum_of_sqrtm_factors(R_stack=r_stack).T
        l_ext = p[:, None] * l_ext_p

        shape = ssv.target_shape
        rv = variables.DenseNormal(
            mean=m_ext, cov_sqrtm_lower=l_ext, target_shape=shape
        )
        u = m_ext.reshape(self.target_shape, order="F")[0, :]
        ssv = variables.DenseSSV(u, rv, target_shape=shape)
        return ssv, None


class _IBMSm(_extra.Extrapolation):
    def __init__(self, *args, num_derivatives, ode_shape):
        super().__init__(*args)

        self.num_derivatives = num_derivatives
        assert len(ode_shape) == 1
        self.ode_shape = ode_shape

    @property
    def target_shape(self):
        return (self.num_derivatives + 1,) + self.ode_shape

    def __repr__(self):
        args2 = f"num_derivatives={self.num_derivatives}"
        args3 = f"ode_shape={self.ode_shape}"
        return f"<Dense IBM with {args2}, {args3}>"

    def solution_from_tcoeffs(self, tcoeffs, /):
        m0, c_sqrtm0 = _stack_tcoeffs(tcoeffs, target_shape=self.target_shape)
        rv = variables.DenseNormal(
            mean=m0, cov_sqrtm_lower=c_sqrtm0, target_shape=self.target_shape
        )
        conds = variables.identity_conditional(*self.target_shape)
        return _markov.MarkovSeqRev(init=rv, conditional=conds)

    def init(self, sol: _markov.MarkovSeqRev, /):
        u = sol.init.mean.reshape(self.target_shape, order="F")[0, :]
        ssv = variables.DenseSSV(u, sol.init, target_shape=self.target_shape)
        extra = sol.conditional
        return ssv, extra

    def extract(self, ssv, extra, /) -> _markov.MarkovSeqRev:
        rv = ssv.hidden_state
        cond = extra
        return _markov.MarkovSeqRev(init=rv, conditional=cond)

    def begin(self, ssv, extra, /, dt):
        p, p_inv = self.preconditioner(dt=dt)
        m0_p = p_inv * ssv.hidden_state.mean
        m_ext_p = self.a @ m0_p
        m_ext = p * m_ext_p
        q_sqrtm = p[:, None] * self.q_sqrtm_lower

        l0 = ssv.hidden_state.cov_sqrtm_lower

        (d,) = self.ode_shape
        shape = (self.num_derivatives + 1, d)
        ext = variables.DenseNormal(m_ext, q_sqrtm, target_shape=shape)
        cache = (extra, m_ext_p, m0_p, p, p_inv, l0)
        u = m_ext.reshape(self.target_shape, order="F")[0, :]
        ssv = variables.DenseSSV(u, ext, target_shape=shape)
        return ssv, cache

    def complete(self, ssv, extra, /, output_scale):
        _, m_ext_p, m0_p, p, p_inv, l0 = extra
        # todo: move this to cache? it may be modified by correction models
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

        backward_noise = variables.DenseNormal(
            mean=m_bw, cov_sqrtm_lower=l_bw, target_shape=self.target_shape
        )
        bw_model = variables.DenseConditional(
            g_bw, noise=backward_noise, target_shape=self.target_shape
        )
        rv = variables.DenseNormal(
            mean=m_ext, cov_sqrtm_lower=l_ext, target_shape=self.target_shape
        )
        u = m_ext.reshape(self.target_shape, order="F")[0, :]
        ext = variables.DenseSSV(u, rv, target_shape=self.target_shape)
        return ext, bw_model


class _IBMFp(_extra.Extrapolation):
    def __init__(self, *args, num_derivatives, ode_shape):
        super().__init__(*args)

        self.num_derivatives = num_derivatives
        assert len(ode_shape) == 1
        self.ode_shape = ode_shape

    @property
    def target_shape(self):
        return (self.num_derivatives + 1,) + self.ode_shape

    def __repr__(self):
        args2 = f"num_derivatives={self.num_derivatives}"
        args3 = f"ode_shape={self.ode_shape}"
        return f"<Dense IBM with {args2}, {args3}>"

    def solution_from_tcoeffs(self, tcoeffs, /):
        m0, c_sqrtm0 = _stack_tcoeffs(tcoeffs, target_shape=self.target_shape)
        rv = variables.DenseNormal(
            mean=m0, cov_sqrtm_lower=c_sqrtm0, target_shape=self.target_shape
        )
        conds = variables.identity_conditional(*self.target_shape)
        return _markov.MarkovSeqRev(init=rv, conditional=conds)

    def init(self, sol: _markov.MarkovSeqRev, /):
        u = sol.init.mean.reshape(self.target_shape, order="F")[0, :]
        ssv = variables.DenseSSV(u, sol.init, target_shape=self.target_shape)
        extra = variables.identity_conditional(*self.target_shape)
        return ssv, extra

    def extract(self, ssv, extra, /) -> _markov.MarkovSeqRev:
        rv = ssv.hidden_state
        cond = extra
        return _markov.MarkovSeqRev(init=rv, conditional=cond)

    def begin(self, ssv, extra, /, dt):
        p, p_inv = self.preconditioner(dt=dt)
        m0_p = p_inv * ssv.hidden_state.mean
        m_ext_p = self.a @ m0_p
        m_ext = p * m_ext_p
        q_sqrtm = p[:, None] * self.q_sqrtm_lower

        l0 = ssv.hidden_state.cov_sqrtm_lower

        (d,) = self.ode_shape
        shape = (self.num_derivatives + 1, d)
        ext = variables.DenseNormal(m_ext, q_sqrtm, target_shape=shape)
        cache = (extra, m_ext_p, m0_p, p, p_inv, l0)
        u = m_ext.reshape(self.target_shape, order="F")[0, :]
        ssv = variables.DenseSSV(u, ext, target_shape=shape)
        return ssv, cache

    def complete(self, ssv, extra, /, output_scale):
        bw0, m_ext_p, m0_p, p, p_inv, l0 = extra
        # todo: move this to cache? it may be modified by correction models
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

        backward_noise = variables.DenseNormal(
            mean=m_bw, cov_sqrtm_lower=l_bw, target_shape=self.target_shape
        )
        bw_model = variables.DenseConditional(
            g_bw, noise=backward_noise, target_shape=self.target_shape
        )
        bw_model = variables.merge_conditionals(bw0, bw_model)
        rv = variables.DenseNormal(
            mean=m_ext, cov_sqrtm_lower=l_ext, target_shape=self.target_shape
        )
        u = m_ext.reshape(self.target_shape, order="F")[0, :]
        ext = variables.DenseSSV(u, rv, target_shape=self.target_shape)
        return ext, bw_model

    def reset(self, ssv, _extra, /):
        return ssv, variables.identity_conditional(*self.target_shape)


def _stack_tcoeffs(tcoeffs, /, *, target_shape):
    if len(tcoeffs) != target_shape[0]:
        msg1 = "The number of Taylor coefficients does not match "
        msg2 = "the number of derivatives in the implementation."
        raise ValueError(msg1 + msg2)

    if tcoeffs[0].shape != target_shape[1:]:
        msg = "The solver's ODE dimension does not match the initial condition."
        raise ValueError(msg)

    m0_matrix = jnp.stack(tcoeffs)
    m0_corrected = jnp.reshape(m0_matrix, (-1,), order="F")

    assert len(target_shape) == 2
    n, d = target_shape
    c_sqrtm0_corrected = jnp.zeros((n * d, n * d))

    return m0_corrected, c_sqrtm0_corrected
