"""Correction-model API."""

import abc
import functools

import jax
import jax.numpy as jnp

from probdiffeq.backend import statespace
from probdiffeq.statespace import cubature, variables


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
    def complete(self, ssv, corr, /, vector_field, t, p):
        raise NotImplementedError

    @abc.abstractmethod
    def extract(self, ssv, corr, /):
        raise NotImplementedError


class ODEConstraint(Correction):
    def __init__(self, ode_order, linearise_fun, string_repr):
        super().__init__(ode_order=ode_order)

        self.linearise = linearise_fun
        self.string_repr = string_repr

    def __repr__(self):
        return self.string_repr

    def init(self, ssv, /):
        obs_like = statespace.random.qoi_like()
        return ssv, obs_like

    def estimate_error(self, ssv, corr, /, vector_field, t, p):
        def f_wrapped(s):
            return vector_field(*s, t=t, p=p)

        A, b = self.linearise(f_wrapped, ssv.hidden_state.mean)
        observed = statespace.cond.transform.marginalise(ssv.hidden_state, (A, b))

        error_estimate = estimate_error(observed)
        return error_estimate, observed, (A, b)

    def complete(self, ssv, corr, /, vector_field, t, p):
        A, b = corr
        obs, (cor, _gn) = statespace.cond.transform.revert(ssv.hidden_state, (A, b))
        u = statespace.random.qoi(cor)
        ssv = variables.SSV(u, cor)
        return ssv, obs

    def extract(self, ssv, corr, /):
        return ssv


class ODEConstraintNoisy(Correction):
    def __init__(self, ode_order, linearise_fun, string_repr):
        super().__init__(ode_order=ode_order)

        self.linearise = linearise_fun
        self.string_repr = string_repr

    def __repr__(self):
        return self.string_repr

    def init(self, ssv, /):
        obs_like = statespace.random.qoi_like()
        return ssv, obs_like

    def estimate_error(self, ssv, corr, /, vector_field, t, p):
        f_wrapped = functools.partial(vector_field, t=t, p=p)
        A, b = self.linearise(f_wrapped, ssv.hidden_state)
        observed = statespace.cond.conditional.marginalise(ssv.hidden_state, (A, b))

        error_estimate = estimate_error(observed)
        return error_estimate, observed, (A, b)

    def complete(self, ssv, corr, /, vector_field, t, p):
        # Re-linearise (because the linearisation point changed)
        f_wrapped = functools.partial(vector_field, t=t, p=p)
        A, b = self.linearise(f_wrapped, ssv.hidden_state)

        # Condition
        obs, (cor, _gn) = statespace.cond.conditional.revert(ssv.hidden_state, (A, b))
        u = statespace.random.qoi(cor)
        ssv = variables.SSV(u, cor)
        return ssv, obs

    def extract(self, ssv, corr, /):
        return ssv


def estimate_error(observed, /):
    zero_data = jnp.zeros_like(statespace.random.mean(observed))
    output_scale = statespace.random.mahalanobis_norm(zero_data, rv=observed)
    error_estimate_unscaled = statespace.random.standard_deviation(observed)
    return output_scale * error_estimate_unscaled


def taylor_order_zero(*, ode_order) -> ODEConstraint:
    return ODEConstraint(
        ode_order=ode_order,
        linearise_fun=statespace.linearise_ode.constraint_0th(ode_order=ode_order),
        string_repr=f"<TS0 with ode_order={ode_order}>",
    )


def taylor_order_one(*, ode_order) -> ODEConstraint:
    return ODEConstraint(
        ode_order=ode_order,
        linearise_fun=statespace.linearise_ode.constraint_1st(ode_order=ode_order),
        string_repr=f"<TS1 with ode_order={ode_order}>",
    )


def statistical_order_one(cubature_fun=cubature.third_order_spherical):
    linearise_fun = statespace.linearise_ode.constraint_statistical_1st(cubature_fun)
    return ODEConstraintNoisy(
        ode_order=1,
        linearise_fun=linearise_fun,
        string_repr=f"<SLR1 with ode_order={1}>",
    )


def statistical_order_zero(cubature_fun=cubature.third_order_spherical):
    linearise_fun = statespace.linearise_ode.constraint_statistical_0th(cubature_fun)
    return ODEConstraintNoisy(
        ode_order=1,
        linearise_fun=linearise_fun,
        string_repr=f"<SLR0 with ode_order={1}>",
    )


def _constraint_flatten(node):
    children = ()
    aux = node.ode_order, node.linearise, node.string_repr
    return children, aux


def _constraint_unflatten(aux, _children, *, nodetype):
    ode_order, lin, string_repr = aux
    return nodetype(ode_order=ode_order, linearise_fun=lin, string_repr=string_repr)


for nodetype in [ODEConstraint, ODEConstraintNoisy]:
    jax.tree_util.register_pytree_node(
        nodetype=nodetype,
        flatten_func=_constraint_flatten,
        unflatten_func=functools.partial(_constraint_unflatten, nodetype=nodetype),
    )
