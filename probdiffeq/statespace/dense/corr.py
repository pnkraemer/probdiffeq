"""Corrections."""

import functools

import jax
import jax.numpy as jnp

from probdiffeq.statespace import corr, cubature
from probdiffeq.statespace.dense import linearise_ode, variables


def taylor_order_zero(*, ode_shape, ode_order):
    fun = linearise_ode.constraint_0th(ode_order=ode_order, ode_shape=ode_shape)
    return _DenseODEConstraint(
        ode_shape=ode_shape,
        ode_order=ode_order,
        linearise_fun=fun,
        string_repr=f"<TS0 with ode_order={ode_order}>",
    )


def taylor_order_one(*, ode_shape, ode_order):
    fun = linearise_ode.constraint_1st(ode_order=ode_order, ode_shape=ode_shape)
    return _DenseODEConstraint(
        ode_shape=ode_shape,
        ode_order=ode_order,
        linearise_fun=fun,
        string_repr=f"<TS1 with ode_order={ode_order}>",
    )


def statistical_order_zero(
    ode_shape,
    ode_order,
    cubature_rule_fn=cubature.third_order_spherical,
):
    linearise_fun = linearise_ode.constraint_statistical_0th(
        ode_shape=ode_shape, cubature_fun=cubature_rule_fn
    )
    return _DenseODEConstraintNoisy(
        ode_shape=ode_shape,
        ode_order=ode_order,
        linearise_fun=linearise_fun,
        string_repr=f"<SLR0 with ode_order={ode_order}>",
    )


def statistical_order_one(
    ode_shape,
    ode_order,
    cubature_rule_fn=cubature.third_order_spherical,
):
    linearise_fun = linearise_ode.constraint_statistical_1st(
        ode_shape=ode_shape, cubature_fun=cubature_rule_fn
    )
    return _DenseODEConstraintNoisy(
        ode_shape=ode_shape,
        ode_order=ode_order,
        linearise_fun=linearise_fun,
        string_repr=f"<SLR1 with ode_order={ode_order}>",
    )


class _DenseODEConstraint(corr.Correction):
    def __init__(self, ode_shape, ode_order, linearise_fun, string_repr):
        super().__init__(ode_order=ode_order)
        assert len(ode_shape) == 1
        self.ode_shape = ode_shape

        self.linearise = linearise_fun
        self.string_repr = string_repr

    def __repr__(self):
        return self.string_repr

    def init(self, ssv, /):
        m_like = jnp.zeros(self.ode_shape)
        chol_like = jnp.zeros(self.ode_shape + self.ode_shape)
        obs_like = variables.DenseNormal(m_like, chol_like, target_shape=None)
        return ssv, obs_like

    def estimate_error(self, ssv: variables.DenseSSV, corr, /, vector_field, t, p):
        def f_wrapped(s):
            return vector_field(*s, t=t, p=p)

        A, b = self.linearise(f_wrapped, ssv.hidden_state.mean)
        observed = variables.marginalise_deterministic(ssv.hidden_state, (A, b))

        error_estimate = _estimate_error(observed)
        return error_estimate, observed, (A, b)

    def complete(self, ssv, corr, /, vector_field, t, p):
        A, b = corr
        observed, (cor, _gn) = variables.revert_deterministic(ssv.hidden_state, (A, b))

        u = jnp.reshape(cor.mean, ssv.target_shape, order="F")[0, :]
        ssv = variables.DenseSSV(u, cor, target_shape=ssv.target_shape)
        return ssv, observed

    def extract(self, ssv, corr, /):
        return ssv


class _DenseODEConstraintNoisy(corr.Correction):
    def __init__(self, ode_shape, ode_order, linearise_fun, string_repr):
        super().__init__(ode_order=ode_order)
        assert len(ode_shape) == 1
        self.ode_shape = ode_shape

        self.linearise = linearise_fun
        self.string_repr = string_repr

    def __repr__(self):
        return self.string_repr

    def init(self, ssv, /):
        m_like = jnp.zeros(self.ode_shape)
        chol_like = jnp.zeros(self.ode_shape + self.ode_shape)
        obs_like = variables.DenseNormal(m_like, chol_like, target_shape=None)
        return ssv, obs_like

    def estimate_error(self, ssv: variables.DenseSSV, corr, /, vector_field, t, p):
        f_wrapped = functools.partial(vector_field, t=t, p=p)
        A, b = self.linearise(f_wrapped, ssv.hidden_state)
        observed = variables.marginalise_stochastic(ssv.hidden_state, (A, b))

        error_estimate = _estimate_error(observed)
        return error_estimate, observed, (A, b)

    def complete(self, ssv, corr, /, vector_field, t, p):
        # Re-linearise (because the linearisation point changed)
        f_wrapped = functools.partial(vector_field, t=t, p=p)
        A, b = self.linearise(f_wrapped, ssv.hidden_state)

        # Condition
        observed, (cor, _gn) = variables.revert_stochastic(ssv.hidden_state, (A, b))

        u = jnp.reshape(cor.mean, ssv.target_shape, order="F")[0, :]
        ssv = variables.DenseSSV(u, cor, target_shape=ssv.target_shape)
        return ssv, observed

    def extract(self, ssv, corr, /):
        return ssv


def _estimate_error(observed, /):
    mahalanobis_norm = observed.mahalanobis_norm(jnp.zeros_like(observed.mean))
    output_scale = mahalanobis_norm / jnp.sqrt(observed.mean.size)
    error_estimate_unscaled = observed.marginal_stds()
    return output_scale * error_estimate_unscaled


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


for nodetype in [_DenseODEConstraint, _DenseODEConstraintNoisy]:
    jax.tree_util.register_pytree_node(
        nodetype=nodetype,
        flatten_func=_constraint_flatten,
        unflatten_func=functools.partial(_constraint_unflatten, nodetype=nodetype),
    )
