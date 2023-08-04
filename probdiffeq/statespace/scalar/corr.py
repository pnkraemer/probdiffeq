"""Implementations for scalar initial value problems."""

import functools

import jax
import jax.numpy as jnp

from probdiffeq.statespace import _corr
from probdiffeq.statespace.scalar import linearise_ode, variables


def taylor_order_zero(*, ode_order):
    fun = linearise_ode.constraint_0th(ode_order=ode_order)
    return _ODEConstraint(
        ode_order=ode_order,
        linearise_fun=fun,
        string_repr=f"<TS0 with ode_order={ode_order}>",
    )


class _ODEConstraint(_corr.Correction):
    def __init__(self, ode_order, linearise_fun, string_repr):
        super().__init__(ode_order=ode_order)

        self.linearise = linearise_fun
        self.string_repr = string_repr

    def __repr__(self):
        return self.string_repr

    def init(self, ssv, /):
        bias_like = jnp.empty(())
        chol_like = jnp.empty(())
        obs_like = variables.NormalQOI(bias_like, chol_like)
        return ssv, obs_like

    def estimate_error(self, ssv, corr, /, vector_field, t, p):
        def f_wrapped(s):
            return vector_field(*s, t=t, p=p)

        A, b = self.linearise(f_wrapped, ssv.hidden_state.mean)
        observed = variables.marginalise_deterministic_qoi(ssv.hidden_state, (A, b))

        error_estimate = estimate_error(observed)
        return error_estimate, observed, (A, b)

    def complete(self, ssv, corr, /, vector_field, t, p):
        A, b = corr
        obs, (cor, _gn) = variables.revert_deterministic_qoi(ssv.hidden_state, (A, b))
        print(cor.mean.shape)
        u = cor.mean[0]
        ssv = variables.SSV(u, cor)
        return ssv, obs

    def extract(self, ssv, corr, /):
        return ssv


def estimate_error(observed, /):
    mahalanobis_norm = observed.mahalanobis_norm(jnp.zeros(()))
    output_scale = mahalanobis_norm
    error_estimate_unscaled = observed.marginal_stds()
    return output_scale * error_estimate_unscaled


def _constraint_flatten(node):
    children = ()
    aux = node.ode_order, node.linearise, node.string_repr
    return children, aux


def _constraint_unflatten(aux, _children, *, nodetype):
    ode_order, lin, string_repr = aux
    return nodetype(ode_order=ode_order, linearise_fun=lin, string_repr=string_repr)


for nodetype in [_ODEConstraint]:
    jax.tree_util.register_pytree_node(
        nodetype=nodetype,
        flatten_func=_constraint_flatten,
        unflatten_func=functools.partial(_constraint_unflatten, nodetype=nodetype),
    )
