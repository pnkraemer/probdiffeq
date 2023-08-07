"""Extrapolation model interfaces."""

import abc
import functools

import jax

from probdiffeq import _markov
from probdiffeq.backend import statespace
from probdiffeq.statespace import variables


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
    def filter(self) -> Extrapolation:
        raise NotImplementedError

    @abc.abstractmethod
    def smoother(self) -> Extrapolation:
        raise NotImplementedError

    @abc.abstractmethod
    def fixedpoint(self) -> Extrapolation:
        raise NotImplementedError


class PreconFilter(Extrapolation):
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
        return variables.SSV(u, sol), None

    def extract(self, ssv, extra, /):
        return ssv.hidden_state

    def begin(self, ssv, extra, /, dt):
        cond, (p, p_inv) = self.discretise(dt)

        rv = ssv.hidden_state
        rv_p = statespace.ssm_util.preconditioner_apply(rv, p_inv)

        m_ext_p = statespace.random.mean(rv_p)
        extrapolated_p = statespace.conditional.apply(m_ext_p, cond)

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
        extrapolated_p = statespace.conditional.marginalise(rv_p, (A, noise))
        extrapolated = statespace.ssm_util.preconditioner_apply(extrapolated_p, p)

        # Gather and return
        u = statespace.random.qoi(extrapolated)
        ssv = variables.SSV(u, extrapolated)
        return ssv, None


class PreconSmoother(Extrapolation):
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

        m_p = statespace.random.mean(rv_p)
        extrapolated_p = statespace.conditional.apply(m_p, cond)

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
        extrapolated_p, cond_p = statespace.conditional.revert(rv_p, (A, noise))
        extrapolated = statespace.ssm_util.preconditioner_apply(extrapolated_p, p)
        cond = statespace.ssm_util.preconditioner_apply_cond(cond_p, p, p_inv)

        # Gather and return
        u = statespace.random.qoi(extrapolated)
        ssv = variables.SSV(u, extrapolated)
        return ssv, cond


class PreconFixedPoint(Extrapolation):
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
        extrapolated_p = statespace.conditional.apply(m_ext_p, cond)

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
        extrapolated_p, cond_p = statespace.conditional.revert(rv_p, (A, noise))
        extrapolated = statespace.ssm_util.preconditioner_apply(extrapolated_p, p)
        cond = statespace.ssm_util.preconditioner_apply_cond(cond_p, p, p_inv)

        # Merge conditionals
        cond = statespace.conditional.merge(bw0, cond)

        # Gather and return
        u = statespace.random.qoi(extrapolated)
        ssv = variables.SSV(u, extrapolated)
        return ssv, cond

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


def _unflatten(nodetype, _aux, children):
    return nodetype(*children)


for impl in [PreconFilter, PreconSmoother, PreconFixedPoint]:
    jax.tree_util.register_pytree_node(
        nodetype=impl,
        flatten_func=_flatten,
        unflatten_func=functools.partial(_unflatten, impl),
    )


class IBMExtrapolationFactory(ExtrapolationFactory):
    def __init__(self, args):
        self.args = args

    def string_repr(self):
        num_derivatives = self.filter().num_derivatives
        return f"<IBM with num_derivatives={num_derivatives}>"

    def filter(self):
        return PreconFilter(*self.args)

    def smoother(self):
        return PreconSmoother(*self.args)

    def fixedpoint(self):
        return PreconFixedPoint(*self.args)


jax.tree_util.register_pytree_node(
    nodetype=IBMExtrapolationFactory,
    flatten_func=lambda a: (a.args, ()),
    unflatten_func=lambda a, b: IBMExtrapolationFactory(b),
)


def ibm_factory(num_derivatives) -> IBMExtrapolationFactory:
    discretise = statespace.ssm_util.ibm_transitions(num_derivatives)
    return IBMExtrapolationFactory(args=(discretise, num_derivatives))
