"""Extrapolation behaviour for scalar state-space models."""
from typing import Tuple

import jax
import jax.numpy as jnp

from probdiffeq import _markov, _sqrt_util
from probdiffeq.statespace import _extra, _ibm_util
from probdiffeq.statespace.scalar import variables


def ibm_discretise_fwd(
    dts, /, *, num_derivatives, output_scale=1.0
) -> _markov.MarkovSeqPreconFwd:
    """Construct the discrete transition densities of an IBM prior.

    Initialises with a scaled standard normal distribution.
    """
    init = variables.standard_normal(num_derivatives + 1, output_scale=output_scale)

    transitions_vmap = jax.vmap(ibm_transitions_precon, in_axes=(0, None, None))
    cond, precon = transitions_vmap(dts, num_derivatives, output_scale)

    return _markov.MarkovSeqPreconFwd(
        init=init, conditional=cond, preconditioner=precon
    )


def ibm_transitions_precon(dt, /, num_derivatives, output_scale):
    """Compute the discrete transition densities for the IBM on a pre-specified grid."""
    a, q_sqrtm = _ibm_util.system_matrices_1d(
        num_derivatives=num_derivatives, output_scale=output_scale
    )
    q0 = jnp.zeros((num_derivatives + 1,))
    noise = variables.NormalHiddenState(q0, q_sqrtm)
    transitions = variables.ConditionalHiddenState(transition=a, noise=noise)

    precon_fun = _ibm_util.preconditioner_prepare(num_derivatives=num_derivatives)
    p, p_inv = precon_fun(dt)

    return transitions, (p, p_inv)


def extrapolate_precon_with_reversal(
    rv: variables.NormalHiddenState,
    conditional: variables.ConditionalHiddenState,
    preconditioner: Tuple[jax.Array, jax.Array],
):
    """Extrapolate and compute smoothing gains in a preconditioned model.

    Careful: the reverse-conditional is preconditioned.
    """
    # Read quantities
    a = conditional.transition
    q0, q_sqrtm = conditional.noise.mean, conditional.noise.cov_sqrtm_lower
    p, p_inv = preconditioner
    m0, l0 = rv.mean, rv.cov_sqrtm_lower

    # Apply preconditioner
    m0_p = p_inv * m0
    l0_p = p_inv[:, None] * l0

    # Extrapolate with reversal
    m_ext_p = a @ m0_p + q0
    r_ext_p, (r_rev_p, gain_p) = _sqrt_util.revert_conditional(
        R_X_F=(a @ l0_p).T,
        R_X=l0_p.T,
        R_YX=q_sqrtm.T,
    )
    l_ext_p = r_ext_p.T
    l_rev_p = r_rev_p.T

    # Catch up with the mean
    m_rev_p = m0_p - gain_p @ m_ext_p

    # Unapply preconditioner for the state variable
    # (the system matrices remain preconditioned)
    m_ext = p * m_ext_p
    l_ext = p[:, None] * l_ext_p

    # Gather and return variables
    marginal = variables.NormalHiddenState(mean=m_ext, cov_sqrtm_lower=l_ext)
    reversal_p = variables.NormalHiddenState(mean=m_rev_p, cov_sqrtm_lower=l_rev_p)
    conditional = variables.ConditionalHiddenState(transition=gain_p, noise=reversal_p)
    return marginal, conditional


def extrapolate_precon(
    rv: variables.NormalHiddenState,
    conditional: variables.ConditionalHiddenState,
    preconditioner: Tuple[jax.Array, jax.Array],
):
    # Read quantities
    a = conditional.transition
    q0, q_sqrtm = conditional.noise.mean, conditional.noise.cov_sqrtm_lower
    p, p_inv = preconditioner
    m0, l0 = rv.mean, rv.cov_sqrtm_lower

    # Apply preconditioner
    m0_p = p_inv * m0
    l0_p = p_inv[:, None] * l0

    # Extrapolate with reversal
    m_ext_p = a @ m0_p + q0
    r_ext_p = _sqrt_util.sum_of_sqrtm_factors(R_stack=((a @ l0_p).T, q_sqrtm.T))
    l_ext_p = r_ext_p.T

    # Unapply preconditioner for the state variable
    m_ext = p * m_ext_p
    l_ext = p[:, None] * l_ext_p

    # Gather and return variables
    marginal = variables.NormalHiddenState(mean=m_ext, cov_sqrtm_lower=l_ext)
    return marginal


def ibm_scalar_factory(num_derivatives):
    a, q_sqrtm = _ibm_util.system_matrices_1d(num_derivatives=num_derivatives)
    precon = _ibm_util.preconditioner_prepare(num_derivatives=num_derivatives)

    factory = _ScalarExtrapolationFactory()
    params = (a, q_sqrtm, precon)
    return factory, params


class _ScalarExtrapolationFactory(_extra.ExtrapolationFactory):
    def filter(self, *params):
        return _IBMFi(*params)

    def smoother(self, *params):
        return _IBMSm(*params)

    def fixedpoint(self, *params):
        return _IBMFp(*params)


class _IBMFi(_extra.Extrapolation):
    def __repr__(self):
        args2 = f"num_derivatives={self.num_derivatives}"
        return f"<IBM with {args2}>"

    @property
    def num_derivatives(self):
        return self.a.shape[0] - 1

    def solution_from_tcoeffs(self, tcoeffs, /):
        m0, c_sqrtm0 = _stack_tcoeffs(tcoeffs, num_derivatives=self.num_derivatives)
        return variables.NormalHiddenState(mean=m0, cov_sqrtm_lower=c_sqrtm0)

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


class _IBMSm(_extra.Extrapolation):
    def __repr__(self):
        args2 = f"num_derivatives={self.num_derivatives}"
        return f"<IBM with {args2}>"

    @property
    def num_derivatives(self):
        return self.a.shape[0] - 1

    def solution_from_tcoeffs(self, tcoeffs, /):
        m0, c_sqrtm0 = _stack_tcoeffs(tcoeffs, num_derivatives=self.num_derivatives)
        rv = variables.NormalHiddenState(mean=m0, cov_sqrtm_lower=c_sqrtm0)
        cond = variables.identity_conditional(ndim=len(tcoeffs))
        return _markov.MarkovSeqRev(init=rv, conditional=cond)

    def init(self, sol: _markov.MarkovSeqRev, /):
        ssv = variables.SSV(sol.init.mean[0], sol.init)
        extra = sol.conditional
        return ssv, extra

    def extract(self, ssv, extra, /):
        return _markov.MarkovSeqRev(init=ssv.hidden_state, conditional=extra)

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


class _IBMFp(_extra.Extrapolation):
    def __repr__(self):
        args2 = f"num_derivatives={self.num_derivatives}"
        return f"<IBM with {args2}>"

    @property
    def num_derivatives(self):
        return self.a.shape[0] - 1

    def solution_from_tcoeffs(self, tcoeffs, /):
        m0, c_sqrtm0 = _stack_tcoeffs(tcoeffs, num_derivatives=self.num_derivatives)
        rv = variables.NormalHiddenState(mean=m0, cov_sqrtm_lower=c_sqrtm0)
        cond = variables.identity_conditional(ndim=len(tcoeffs))
        return _markov.MarkovSeqRev(init=rv, conditional=cond)

    def init(self, sol: _markov.MarkovSeqRev, /):
        ssv = variables.SSV(sol.init.mean[0], sol.init)
        extra = variables.identity_conditional(ndim=sol.init.mean.shape[0])
        return ssv, extra

    def extract(self, ssv, extra, /):
        return _markov.MarkovSeqRev(init=ssv.hidden_state, conditional=extra)

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

    def reset(self, ssv, _extra, /):
        cond = variables.identity_conditional(ndim=ssv.hidden_state.mean.shape[0])
        return ssv, cond


def _stack_tcoeffs(tcoeffs, /, *, num_derivatives):
    if len(tcoeffs) != num_derivatives + 1:
        msg1 = "The number of Taylor coefficients does not match "
        msg2 = "the number of derivatives in the implementation."
        raise ValueError(msg1 + msg2)
    m0_matrix = jnp.stack(tcoeffs)
    m0_corrected = jnp.reshape(m0_matrix, (-1,), order="F")
    c_sqrtm0_corrected = jnp.zeros((num_derivatives + 1, num_derivatives + 1))
    return m0_corrected, c_sqrtm0_corrected


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
