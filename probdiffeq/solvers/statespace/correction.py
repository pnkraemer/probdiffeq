"""Correction-model API."""

import abc
import functools

import jax
import jax.numpy as jnp

from probdiffeq.impl import impl
from probdiffeq.solvers.statespace import cubature


class Correction(abc.ABC):
    """Correction model interface."""

    def __init__(self, ode_order):
        self.ode_order = ode_order

    @abc.abstractmethod
    def init(self, x, /):
        raise NotImplementedError

    @abc.abstractmethod
    def estimate_error(self, ssv, corr, /, vector_field, t, p):
        raise NotImplementedError

    @abc.abstractmethod
    def complete(self, ssv, corr, /):
        raise NotImplementedError

    @abc.abstractmethod
    def extract(self, ssv, corr, /):
        raise NotImplementedError


class ODEConstraintTaylor(Correction):
    def __init__(self, ode_order, linearise_fun, string_repr):
        super().__init__(ode_order=ode_order)

        self.linearise = linearise_fun
        self.string_repr = string_repr

    def __repr__(self):
        return self.string_repr

    def init(self, ssv, /):
        obs_like = impl.ssm_util.prototype_qoi()
        return ssv, obs_like

    def estimate_error(self, hidden_state, _corr, /, vector_field, t, p):
        def f_wrapped(s):
            return vector_field(*s, t=t, p=p)

        A, b = self.linearise(f_wrapped, hidden_state.mean)
        observed = impl.transform.marginalise(hidden_state, (A, b))

        error_estimate = _estimate_error(observed)
        return error_estimate, observed, (A, b)

    def complete(self, hidden_state, corr, /):
        A, b = corr
        observed, (_gain, corrected) = impl.transform.revert(hidden_state, (A, b))
        return corrected, observed

    def extract(self, ssv, _corr, /):
        return ssv


class ODEConstraintStatistical(Correction):
    def __init__(self, ode_order, linearise_fun, string_repr):
        super().__init__(ode_order=ode_order)

        self.linearise = linearise_fun
        self.string_repr = string_repr

    def __repr__(self):
        return self.string_repr

    def init(self, ssv, /):
        obs_like = impl.ssm_util.prototype_qoi()
        return ssv, obs_like

    def estimate_error(self, hidden_state, _corr, /, vector_field, t, p):
        f_wrapped = functools.partial(vector_field, t=t, p=p)
        A, b = self.linearise(f_wrapped, hidden_state)
        observed = impl.conditional.marginalise(hidden_state, (A, b))

        error_estimate = _estimate_error(observed)
        return error_estimate, observed, (A, b, f_wrapped)

    def complete(self, hidden_state, corr, /):
        # Re-linearise (because the linearisation point changed)
        *_, f_wrapped = corr
        A, b = self.linearise(f_wrapped, hidden_state)

        # Condition
        observed, (_gain, corrected) = impl.conditional.revert(hidden_state, (A, b))
        return corrected, observed

    def extract(self, hidden_state, _corr, /):
        return hidden_state


def _estimate_error(observed, /):
    # todo: the functions involved in error estimation are still a bit patchy.
    #  for instance, they assume that they are called in exactly this error estimation
    #  context. Same for prototype_qoi etc.
    zero_data = jnp.zeros(())
    output_scale = impl.random.mahalanobis_norm_relative(zero_data, rv=observed)
    error_estimate_unscaled = impl.random.standard_deviation(observed)
    return output_scale * error_estimate_unscaled


def _constraint_flatten(node):
    children = ()
    aux = node.ode_order, node.linearise, node.string_repr
    return children, aux


def _constraint_unflatten(aux, _children, *, nodetype):
    ode_order, lin, string_repr = aux
    return nodetype(ode_order=ode_order, linearise_fun=lin, string_repr=string_repr)


for nodetype in [ODEConstraintTaylor, ODEConstraintStatistical]:
    jax.tree_util.register_pytree_node(
        nodetype=nodetype,
        flatten_func=_constraint_flatten,
        unflatten_func=functools.partial(_constraint_unflatten, nodetype=nodetype),
    )


def taylor_order_zero(*, ode_order) -> ODEConstraintTaylor:
    return ODEConstraintTaylor(
        ode_order=ode_order,
        linearise_fun=impl.linearise.ode_taylor_0th(ode_order=ode_order),
        string_repr=f"<TS0 with ode_order={ode_order}>",
    )


def taylor_order_one(*, ode_order) -> ODEConstraintTaylor:
    return ODEConstraintTaylor(
        ode_order=ode_order,
        linearise_fun=impl.linearise.ode_taylor_1st(ode_order=ode_order),
        string_repr=f"<TS1 with ode_order={ode_order}>",
    )


def statistical_order_one(cubature_fun=cubature.third_order_spherical):
    linearise_fun = impl.linearise.ode_statistical_1st(cubature_fun)
    return ODEConstraintStatistical(
        ode_order=1,
        linearise_fun=linearise_fun,
        string_repr=f"<SLR1 with ode_order={1}>",
    )


def statistical_order_zero(cubature_fun=cubature.third_order_spherical):
    linearise_fun = impl.linearise.ode_statistical_0th(cubature_fun)
    return ODEConstraintStatistical(
        ode_order=1,
        linearise_fun=linearise_fun,
        string_repr=f"<SLR0 with ode_order={1}>",
    )
