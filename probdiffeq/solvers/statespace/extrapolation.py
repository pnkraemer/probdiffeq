"""Extrapolation model interfaces."""

import abc
import functools

import jax

from probdiffeq.impl import impl
from probdiffeq.solvers import markov


class Extrapolation(abc.ABC):
    """Extrapolation model interface."""

    @abc.abstractmethod
    def solution_from_tcoeffs(self, taylor_coefficients, /):
        raise NotImplementedError

    @abc.abstractmethod
    def init(self, sol, /):
        raise NotImplementedError

    @abc.abstractmethod
    def begin(self, ssv, extra, /, dt):
        raise NotImplementedError

    @abc.abstractmethod
    def complete(self, ssv, extra, /, output_scale):
        raise NotImplementedError

    @abc.abstractmethod
    def extract(self, ssv, extra, /):
        raise NotImplementedError


# At the point of choosing the recipe
# (aka selecting the desired state-space model factorisation),
# it is too early to know whether we solve forward-in-time only (aka filtering)
# or require a dense, or fixed-point solution. Therefore, the recipes return
# extrapolation *factories* instead of extrapolation models.
class ExtrapolationFactory(abc.ABC):
    @abc.abstractmethod
    def string_repr(self):
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self) -> Extrapolation:
        raise NotImplementedError

    @abc.abstractmethod
    def dense(self) -> Extrapolation:
        raise NotImplementedError

    @abc.abstractmethod
    def save_at(self) -> Extrapolation:
        raise NotImplementedError


class PreconFilter(Extrapolation):
    def __init__(self, discretise, num_derivatives):
        self.discretise = discretise
        self.num_derivatives = num_derivatives

    def __repr__(self):
        args2 = f"num_derivatives={self.num_derivatives}"
        return f"<IBM with {args2}>"

    def solution_from_tcoeffs(self, tcoeffs, /):
        return impl.ssm_util.normal_from_tcoeffs(tcoeffs, self.num_derivatives)

    def init(self, sol, /):
        return sol, None

    def extract(self, hidden_state, _extra, /):
        return hidden_state

    def begin(self, rv, _extra, /, dt):
        cond, (p, p_inv) = self.discretise(dt)

        rv_p = impl.ssm_util.preconditioner_apply(rv, p_inv)

        m_ext_p = impl.random.mean(rv_p)
        extrapolated_p = impl.conditional.apply(m_ext_p, cond)

        extrapolated = impl.ssm_util.preconditioner_apply(extrapolated_p, p)
        cache = (cond, (p, p_inv), rv_p)
        return extrapolated, cache

    def complete(self, _ssv, extra, /, output_scale):
        cond, (p, p_inv), rv_p = extra

        # Extrapolate the Cholesky factor (re-extrapolate the mean for simplicity)
        A, noise = cond
        noise = impl.random.rescale_cholesky(noise, output_scale)
        extrapolated_p = impl.conditional.marginalise(rv_p, (A, noise))
        extrapolated = impl.ssm_util.preconditioner_apply(extrapolated_p, p)

        # Gather and return
        return extrapolated, None


class PreconSmoother(Extrapolation):
    def __init__(self, discretise, num_derivatives):
        self.discretise = discretise
        self.num_derivatives = num_derivatives

    def __repr__(self):
        args2 = f"num_derivatives={self.num_derivatives}"
        return f"<IBM with {args2}>"

    def solution_from_tcoeffs(self, tcoeffs, /):
        rv = impl.ssm_util.normal_from_tcoeffs(tcoeffs, self.num_derivatives)
        cond = impl.ssm_util.identity_conditional(len(tcoeffs))
        return markov.MarkovSeqRev(init=rv, conditional=cond)

    def init(self, sol: markov.MarkovSeqRev, /):
        return sol.init, sol.conditional

    def extract(self, hidden_state, extra, /):
        return markov.MarkovSeqRev(init=hidden_state, conditional=extra)

    def begin(self, rv, _extra, /, dt):
        cond, (p, p_inv) = self.discretise(dt)

        rv_p = impl.ssm_util.preconditioner_apply(rv, p_inv)

        m_p = impl.random.mean(rv_p)
        extrapolated_p = impl.conditional.apply(m_p, cond)

        extrapolated = impl.ssm_util.preconditioner_apply(extrapolated_p, p)
        cache = (cond, (p, p_inv), rv_p)
        return extrapolated, cache

    def complete(self, _ssv, extra, /, output_scale):
        cond, (p, p_inv), rv_p = extra

        # Extrapolate the Cholesky factor (re-extrapolate the mean for simplicity)
        A, noise = cond
        noise = impl.random.rescale_cholesky(noise, output_scale)
        extrapolated_p, cond_p = impl.conditional.revert(rv_p, (A, noise))
        extrapolated = impl.ssm_util.preconditioner_apply(extrapolated_p, p)
        cond = impl.ssm_util.preconditioner_apply_cond(cond_p, p, p_inv)

        # Gather and return
        return extrapolated, cond


class PreconFixedPoint(Extrapolation):
    def __init__(self, discretise, num_derivatives):
        self.discretise = discretise
        self.num_derivatives = num_derivatives

    def __repr__(self):
        args2 = f"num_derivatives={self.num_derivatives}"
        return f"<IBM with {args2}>"

    def solution_from_tcoeffs(self, tcoeffs, /):
        rv = impl.ssm_util.normal_from_tcoeffs(tcoeffs, self.num_derivatives)
        cond = impl.ssm_util.identity_conditional(len(tcoeffs))
        return markov.MarkovSeqRev(init=rv, conditional=cond)

    def init(self, sol: markov.MarkovSeqRev, /):
        return sol.init, sol.conditional

    def extract(self, hidden_state, extra, /):
        return markov.MarkovSeqRev(init=hidden_state, conditional=extra)

    def begin(self, rv, extra, /, dt):
        cond, (p, p_inv) = self.discretise(dt)

        rv_p = impl.ssm_util.preconditioner_apply(rv, p_inv)

        m_ext_p = impl.random.mean(rv_p)
        extrapolated_p = impl.conditional.apply(m_ext_p, cond)

        extrapolated = impl.ssm_util.preconditioner_apply(extrapolated_p, p)
        cache = (cond, (p, p_inv), rv_p, extra)
        return extrapolated, cache

    def complete(self, _rv, extra, /, output_scale):
        cond, (p, p_inv), rv_p, bw0 = extra

        # Extrapolate the Cholesky factor (re-extrapolate the mean for simplicity)
        A, noise = cond
        noise = impl.random.rescale_cholesky(noise, output_scale)
        extrapolated_p, cond_p = impl.conditional.revert(rv_p, (A, noise))
        extrapolated = impl.ssm_util.preconditioner_apply(extrapolated_p, p)
        cond = impl.ssm_util.preconditioner_apply_cond(cond_p, p, p_inv)

        # Merge conditionals
        cond = impl.conditional.merge(bw0, cond)

        # Gather and return
        return extrapolated, cond

    def reset(self, ssv, _extra, /):
        cond = impl.ssm_util.identity_conditional(self.num_derivatives + 1)
        return ssv, cond


# Register scalar extrapolations as pytrees because we want to vmap them
# for block-diagonal models.
# todo: this feels very temporary...


def _flatten(fi):
    child = fi.a, fi.q_sqrtm_lower, fi.preconditioner
    aux = ()
    return child, aux


def _unflatten(nodetype, _aux, children):
    return nodetype(*children)


for nodetype in [PreconFilter, PreconSmoother, PreconFixedPoint]:
    jax.tree_util.register_pytree_node(
        nodetype=nodetype,
        flatten_func=_flatten,
        unflatten_func=functools.partial(_unflatten, nodetype),
    )


class IBMExtrapolationFactory(ExtrapolationFactory):
    def __init__(self, args):
        self.args = args

    def string_repr(self):
        num_derivatives = self.forward().num_derivatives
        return f"<IBM with num_derivatives={num_derivatives}>"

    def forward(self):
        return PreconFilter(*self.args)

    def dense(self):
        return PreconSmoother(*self.args)

    def save_at(self):
        return PreconFixedPoint(*self.args)


jax.tree_util.register_pytree_node(
    nodetype=IBMExtrapolationFactory,
    flatten_func=lambda a: (a.args, ()),
    unflatten_func=lambda _a, b: IBMExtrapolationFactory(b),
)


def ibm_adaptive(num_derivatives) -> IBMExtrapolationFactory:
    discretise = impl.ssm_util.ibm_transitions(num_derivatives)
    return IBMExtrapolationFactory(args=(discretise, num_derivatives))


def ibm_discretise_fwd(dts, /, *, num_derivatives):
    discretise = impl.ssm_util.ibm_transitions(num_derivatives)
    return jax.vmap(discretise)(dts)


def unit_markov_sequence(num_derivatives):
    cond = impl.ssm_util.identity_conditional(num_derivatives + 1)
    init = impl.ssm_util.standard_normal(num_derivatives + 1, 1.0)
    return markov.MarkovSeqRev(init=init, conditional=cond)
