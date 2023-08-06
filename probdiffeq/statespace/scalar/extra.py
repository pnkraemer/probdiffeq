"""Extrapolation behaviour for scalar state-space models."""
from typing import Tuple

import jax
import jax.numpy as jnp

from probdiffeq import _markov, _sqrt_util
from probdiffeq.backend import statespace
from probdiffeq.statespace import _ibm_util, extra, variables


def ibm_discretise_fwd(
    dts, /, *, num_derivatives, output_scale=1.0
) -> _markov.MarkovSeqPreconFwd:
    """Construct the discrete transition densities of an IBM prior.

    Initialises with a scaled standard normal distribution.
    """
    init = statespace.ssm_util.standard_normal(num_derivatives + 1, output_scale)
    discretise = statespace.ssm_util.ibm_transitions(num_derivatives, output_scale)
    cond, precon = jax.vmap(discretise)(dts)
    # transitions_vmap = jax.vmap(ibm_transitions_precon, in_axes=(0, None, None))
    # cond, precon = transitions_vmap(dts, num_derivatives, output_scale)

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
    rv,
    conditional,
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
    rv,
    conditional,
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


def ibm_factory(num_derivatives, output_scale=1.0):
    discretise = statespace.ssm_util.ibm_transitions(num_derivatives, output_scale)
    return _ScalarExtrapolationFactory(args=(discretise, num_derivatives))
    #
    # a, q_sqrtm = _ibm_util.system_matrices_1d(num_derivatives, output_scale)
    # precon = _ibm_util.preconditioner_prepare(num_derivatives=num_derivatives)
    #
    # return _ScalarExtrapolationFactory(args=(a, q_sqrtm, precon))


class _ScalarExtrapolationFactory(extra.ExtrapolationFactory):
    def __init__(self, args):
        self.args = args

    def string_repr(self):
        num_derivatives = self.filter().num_derivatives
        return f"<Scalar IBM with num_derivatives={num_derivatives}>"

    def filter(self):
        return _IBMFi(*self.args)

    def smoother(self):
        return _IBMSm(*self.args)

    def fixedpoint(self):
        return _IBMFp(*self.args)


jax.tree_util.register_pytree_node(
    nodetype=_ScalarExtrapolationFactory,
    flatten_func=lambda a: (a.args, ()),
    unflatten_func=lambda a, b: _ScalarExtrapolationFactory(b),
)


class _IBMFi(extra.Extrapolation):
    def __init__(self, discretise, num_derivatives):
        self.discretise = discretise
        self.num_derivatives = num_derivatives

    def __repr__(self):
        args2 = f"num_derivatives={self.num_derivatives}"
        return f"<IBM with {args2}>"

    def solution_from_tcoeffs(self, tcoeffs, /):
        return statespace.ssm_util.normal_from_tcoeffs(tcoeffs, self.num_derivatives)

    def init(self, sol, /):
        u = statespace.random.qoi(sol)

        return variables.SSV(sol.mean[0], sol), None

    def extract(self, ssv, extra, /):
        return ssv.hidden_state

    def begin(self, ssv, extra, /, dt):
        cond, (p, p_inv) = self.discretise(dt)

        rv = ssv.hidden_state
        rv_p = statespace.ssm_util.preconditioner_apply(rv, p_inv)

        m_ext_p = statespace.random.mean(rv_p)
        extrapolated_p = statespace.cond.conditional.apply(m_ext_p, cond)

        extrapolated = statespace.ssm_util.preconditioner_apply(extrapolated_p, p)
        qoi = statespace.random.qoi(extrapolated)
        ssv = variables.SSV(qoi, extrapolated)
        cache = (cond, (p, p_inv), rv_p)
        return ssv, cache

    def complete(self, ssv, extra, /, output_scale):
        cond, (p, p_inv), rv_p = extra

        # Extrapolate the Cholesky factor (re-extrapolate the mean for simplicity)
        A, noise = cond
        noise = statespace.random.rescale_cholesky(noise, output_scale)
        extrapolated_p = statespace.cond.conditional.marginalise(rv_p, (A, noise))
        extrapolated = statespace.ssm_util.preconditioner_apply(extrapolated_p, p)

        # Gather and return
        u = statespace.random.qoi(extrapolated)
        ssv = variables.SSV(u, extrapolated)
        return ssv, None


class _IBMSm(extra.Extrapolation):
    def __init__(self, discretise, num_derivatives):
        self.discretise = discretise
        self.num_derivatives = num_derivatives

    def __repr__(self):
        args2 = f"num_derivatives={self.num_derivatives}"
        return f"<IBM with {args2}>"

    def solution_from_tcoeffs(self, tcoeffs, /):
        rv = statespace.ssm_util.normal_from_tcoeffs(tcoeffs, self.num_derivatives)
        cond = statespace.ssm_util.identity_conditional(ndim=len(tcoeffs))
        return _markov.MarkovSeqRev(init=rv, conditional=cond)

    def init(self, sol: _markov.MarkovSeqRev, /):
        hidden_state = sol.init

        # it is always this function -- remove u from SSV (and remove SSV altogether?)
        u = statespace.random.qoi(hidden_state)
        ssv = variables.SSV(u, hidden_state)
        extra = sol.conditional
        return ssv, extra

    def extract(self, ssv, extra, /):
        return _markov.MarkovSeqRev(init=ssv.hidden_state, conditional=extra)

    def begin(self, ssv, extra, /, dt):
        cond, (p, p_inv) = self.discretise(dt)

        rv = ssv.hidden_state
        rv_p = statespace.ssm_util.preconditioner_apply(rv, p_inv)

        m_ext_p = statespace.random.mean(rv_p)
        extrapolated_p = statespace.cond.conditional.apply(m_ext_p, cond)

        extrapolated = statespace.ssm_util.preconditioner_apply(extrapolated_p, p)
        qoi = statespace.random.qoi(extrapolated)
        ssv = variables.SSV(qoi, extrapolated)
        cache = (cond, (p, p_inv), rv_p)
        return ssv, cache

    def complete(self, ssv, extra, /, output_scale):
        cond, (p, p_inv), rv_p = extra

        # Extrapolate the Cholesky factor (re-extrapolate the mean for simplicity)
        A, noise = cond
        noise = statespace.random.rescale_cholesky(noise, output_scale)
        extrapolated_p, cond_p = statespace.cond.conditional.revert(rv_p, (A, noise))
        extrapolated = statespace.ssm_util.preconditioner_apply(extrapolated_p, p)
        cond = statespace.ssm_util.preconditioner_apply_cond(cond_p, p, p_inv)

        # Gather and return
        u = statespace.random.qoi(extrapolated)
        ssv = variables.SSV(u, extrapolated)
        return ssv, cond


class _IBMFp(extra.Extrapolation):
    def __init__(self, discretise, num_derivatives):
        self.discretise = discretise
        self.num_derivatives = num_derivatives

    def __repr__(self):
        args2 = f"num_derivatives={self.num_derivatives}"
        return f"<IBM with {args2}>"

    def solution_from_tcoeffs(self, tcoeffs, /):
        rv = statespace.ssm_util.normal_from_tcoeffs(tcoeffs, self.num_derivatives)
        cond = statespace.ssm_util.identity_conditional(ndim=len(tcoeffs))
        return _markov.MarkovSeqRev(init=rv, conditional=cond)

    def init(self, sol: _markov.MarkovSeqRev, /):
        hidden_state = sol.init

        # it is always this function -- remove u from SSV (and remove SSV altogether?)
        u = statespace.random.qoi(hidden_state)
        ssv = variables.SSV(u, hidden_state)
        extra = sol.conditional
        return ssv, extra

    def extract(self, ssv, extra, /):
        return _markov.MarkovSeqRev(init=ssv.hidden_state, conditional=extra)

    def begin(self, ssv, extra, /, dt):
        cond, (p, p_inv) = self.discretise(dt)

        rv = ssv.hidden_state
        rv_p = statespace.ssm_util.preconditioner_apply(rv, p_inv)

        m_ext_p = statespace.random.mean(rv_p)
        extrapolated_p = statespace.cond.conditional.apply(m_ext_p, cond)

        extrapolated = statespace.ssm_util.preconditioner_apply(extrapolated_p, p)
        qoi = statespace.random.qoi(extrapolated)
        ssv = variables.SSV(qoi, extrapolated)
        cache = (cond, (p, p_inv), rv_p, extra)
        return ssv, cache

    def complete(self, ssv, extra, /, output_scale):
        cond, (p, p_inv), rv_p, bw0 = extra

        # Extrapolate the Cholesky factor (re-extrapolate the mean for simplicity)
        A, noise = cond
        noise = statespace.random.rescale_cholesky(noise, output_scale)
        extrapolated_p, cond_p = statespace.cond.conditional.revert(rv_p, (A, noise))
        extrapolated = statespace.ssm_util.preconditioner_apply(extrapolated_p, p)
        cond = statespace.ssm_util.preconditioner_apply_cond(cond_p, p, p_inv)

        # Merge conditionals
        cond = statespace.cond.conditional.merge(bw0, cond)

        # Gather and return
        u = statespace.random.qoi(extrapolated)
        ssv = variables.SSV(u, extrapolated)
        return ssv, cond

        extrapolated = variables.NormalHiddenState(mean=m_ext, cov_sqrtm_lower=l_ext)
        ssv = variables.SSV(m_ext[0], extrapolated)
        return ssv, bw_model

    def reset(self, ssv, _extra, /):
        cond = statespace.ssm_util.identity_conditional(ndim=self.num_derivatives + 1)
        return ssv, cond


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
