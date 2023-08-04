"""Corrections."""

import functools

import jax
import jax.numpy as jnp

from probdiffeq.statespace import _corr
from probdiffeq.statespace.blockdiag import linearise_ode, variables
from probdiffeq.statespace.scalar import corr as scalar_corr
from probdiffeq.statespace.scalar import variables as scalar_variables


def taylor_order_zero(*, ode_shape, ode_order):
    fun = linearise_ode.constraint_0th(ode_order=ode_order)
    return _BlockDiagODEConstraint(
        ode_shape=ode_shape,
        ode_order=ode_order,
        linearise_fun=fun,
        string_repr=f"<TS0 with ode_order={ode_order}>",
    )


class _BlockDiagODEConstraint(_corr.Correction):
    def __init__(self, *, ode_order, ode_shape, linearise_fun, string_repr):
        super().__init__(ode_order=ode_order)
        assert len(ode_shape) == 1
        self.ode_shape = ode_shape

        self.linearise = linearise_fun
        self.string_repr = string_repr

    def __repr__(self):
        return self.string_repr

    def init(self, ssv, /):
        m_like = jnp.zeros(self.ode_shape)
        chol_like = jnp.zeros(self.ode_shape)
        obs_like = scalar_variables.NormalQOI(m_like, chol_like)
        return ssv, obs_like

    def estimate_error(self, ssv, corr, /, vector_field, t, p):
        def f_wrapped(s):
            return vector_field(*s, t=t, p=p)

        A, b = self.linearise(f_wrapped, ssv.hidden_state.mean)

        observed = variables.marginalise_deterministic(ssv.hidden_state, (A, b))

        error_estimate = _estimate_error(observed)
        return error_estimate, observed, (A, b)

    def complete(self, ssv, corr, /, vector_field, t, p):
        A, b = corr

        observed, (cor, _gn) = variables.revert_deterministic(ssv.hidden_state, (A, b))

        u = cor.mean[..., 0]
        ssv = scalar_variables.SSV(u, cor)
        return ssv, observed

    def extract(self, ssv, corr, /):
        return ssv


def _estimate_error(observed, /):
    return jax.vmap(scalar_corr.estimate_error)(observed)


def _constraint_flatten(node):
    children = ()
    aux = node.ode_order, node.ode_shape, node.linearise, node.string_repr
    return children, aux


def _constraint_unflatten(aux, _children, *, nodetype):
    ode_order, ode_shape, lin, string_repr = aux
    return nodetype(
        ode_order=ode_order,
        ode_shape=ode_shape,
        linearise_fun=lin,
        string_repr=string_repr,
    )


for nodetype in [_BlockDiagODEConstraint]:
    jax.tree_util.register_pytree_node(
        nodetype=nodetype,
        flatten_func=_constraint_flatten,
        unflatten_func=functools.partial(_constraint_unflatten, nodetype=nodetype),
    )
