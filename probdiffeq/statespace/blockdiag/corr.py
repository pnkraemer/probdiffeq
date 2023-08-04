"""Corrections."""

import functools

import jax
import jax.numpy as jnp

from probdiffeq.statespace import _corr, cubature
from probdiffeq.statespace.blockdiag import linearise_ode, variables
from probdiffeq.statespace.scalar import corr as scalar_corr
from probdiffeq.statespace.scalar import variables as scalar_variables

# todo: implementing blockdiags via vmap is a bit awkward to maintain.
#  maybe change to verbose implementation and use vmap only in implementation functions
#  (i.e. don't depend on scalar... but this is a bit too big of a refactor for now)


def taylor_order_zero(*, ode_shape, ode_order):
    fun = linearise_ode.constraint_0th(ode_order=ode_order)
    return _BlockDiagODEConstraint(
        ode_shape=ode_shape,
        ode_order=ode_order,
        linearise_fun=fun,
        string_repr=f"<TS0 with ode_order={ode_order}>",
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

        error_estimate = _error_estimate(observed)
        return error_estimate, observed, (A, b)

    def complete(self, ssv, corr, /, vector_field, t, p):
        A, b = corr

        observed, (cor, _gn) = variables.revert_deterministic(ssv.hidden_state, (A, b))

        u = cor.mean[..., 0]
        ssv = scalar_variables.SSV(u, cor)
        return ssv, observed

    def extract(self, ssv, corr, /):
        return ssv


def _error_estimate(observed, /):
    return jax.vmap(scalar_corr.estimate_error)(observed)


#
# def taylor_order_zero(*args, **kwargs):
#     ts0 = scalar_corr.taylor_order_zero(*args, **kwargs)
#     return _BlockDiag(ts0)
#
# def statistical_order_one(ode_shape, ode_order):
#     cubature_fn = cubature.blockdiag(cubature.third_order_spherical)
#     cubature_rule = cubature_fn(input_shape=ode_shape)
#     return _BlockDiagStatisticalFirstOrder(
#         ode_shape=ode_shape, ode_order=ode_order, cubature_rule=cubature_rule
#     )
# #
#
# @jax.tree_util.register_pytree_node_class
# class _BlockDiagStatisticalFirstOrder(_corr.Correction):
#     """First-order statistical linear regression in state-space models \
#      with block-diagonal covariance structure.
#
#     !!! warning "Warning: highly EXPERIMENTAL feature!"
#         This feature is highly experimental.
#         There is no guarantee that it works correctly.
#         It might be deleted tomorrow
#         and without any deprecation policy.
#
#     """
#
#     def __init__(self, ode_shape, ode_order, cubature_rule):
#         if ode_order > 1:
#             raise ValueError
#
#         super().__init__(ode_order=ode_order)
#         self.ode_shape = ode_shape
#
#         self._mm = scalar_corr.StatisticalFirstOrder(
#             ode_order=ode_order, cubature_rule=cubature_rule
#         )
#
#     @property
#     def cubature_rule(self):
#         return self._mm.cubature_rule
#
#     def tree_flatten(self):
#         # todo: should this call super().tree_flatten()?
#         children = (self.cubature_rule,)
#         aux = self.ode_order, self.ode_shape
#         return children, aux
#
#     @classmethod
#     def tree_unflatten(cls, aux, children):
#         (cubature_rule,) = children
#         ode_order, ode_shape = aux
#         return cls(
#             ode_order=ode_order, ode_shape=ode_shape, cubature_rule=cubature_rule
#         )
#
#     def init(self, ssv, /):
#         return jax.vmap(type(self._mm).init)(self._mm, ssv)
#
#     def extract(self, ssv, corr, /):
#         return ssv
#
#     def estimate_error(self, ssv, corr, /, vector_field, t, p):
#         # Vmap relevant functions
#         vmap_f = jax.vmap(jax.tree_util.Partial(vector_field, t=t, p=p))
#         cache = (vmap_f,)
#
#         # Evaluate vector field at sigma-points
#         sigma_points_fn = jax.vmap(type(self._mm).transform_sigma_points)
#         sigma_points, _, _ = sigma_points_fn(self._mm, ssv.hidden_state)
#
#         fx = vmap_f(sigma_points.T).T  # (d, S).T = (S, d) -> (S, d) -> transpose again
#         center_fn = jax.vmap(scalar_corr.StatisticalFirstOrder.center)
#         fx_mean, _, fx_centered_normed = center_fn(self._mm, fx)
#
#         # Compute output scale and error estimate
#         calibrate_fn = jax.vmap(scalar_corr.StatisticalFirstOrder.calibrate)
#         error_estimate, output_scale, marginals = calibrate_fn(
#             self._mm, fx_mean, fx_centered_normed, ssv.hidden_state
#         )
#         return output_scale * error_estimate, marginals, cache
#
#     def complete(self, ssv, corr, /, vector_field, t, p):
#         (vmap_f,) = corr
#
#         H, noise = self.linearize(ssv, vmap_f)
#
#         compl_fn = scalar_corr.StatisticalFirstOrder.complete_post_linearize
#         fn = jax.vmap(compl_fn)
#         return fn(self._mm, H, ssv.hidden_state, noise)
#
#     def linearize(self, ssv, vmap_f):
#         # Transform the sigma-points
#         sigma_points_fn = jax.vmap(type(self._mm).transform_sigma_points)
#         sigma_points, _, sigma_points_centered_normed = sigma_points_fn(
#             self._mm, ssv.hidden_state
#         )
#
#         # Evaluate the vector field at the sigma-points
#         fx = vmap_f(sigma_points.T).T  # (d, S).T = (S, d) -> (S, d) -> transpose again
#         center_fn = jax.vmap(scalar_corr.StatisticalFirstOrder.center)
#         fx_mean, _, fx_centered_normed = center_fn(self._mm, fx)
#
#         # Complete the linearization
#         lin_fn = jax.vmap(scalar_corr.StatisticalFirstOrder.linearization_matrices)
#         return lin_fn(
#             self._mm,
#             fx_centered_normed,
#             fx_mean,
#             sigma_points_centered_normed,
#             ssv.hidden_state,
#         )
#
#
# @jax.tree_util.register_pytree_node_class
# class _BlockDiag(_corr.Correction):
#     def __init__(self, corr, /):
#         super().__init__(ode_order=corr.ode_order)
#         self.corr = corr
#
#     def __repr__(self):
#         return f"{self.__class__.__name__}({self.corr})"
#
#     def tree_flatten(self):
#         children = (self.corr,)
#         aux = ()
#         return children, aux
#
#     @classmethod
#     def tree_unflatten(cls, _aux, children):
#         (corr,) = children
#         return cls(corr)
#
#     def init(self, ssv, /):
#         return jax.vmap(type(self.corr).init)(self.corr, ssv)
#
#     def estimate_error(self, ssv, corr, /, vector_field, t, p):
#         select_fn = jax.vmap(type(self.corr).select_derivatives)
#         m0, m1 = select_fn(self.corr, ssv.hidden_state)
#
#         fx = vector_field(*m0.T, t=t, p=p)
#
#         marginalise_fn = jax.vmap(type(self.corr).marginalise_observation)
#         cache, obs_unbatch = marginalise_fn(self.corr, fx, m1, ssv.hidden_state)
#
#         mahalanobis_fn = scalar_variables.NormalQOI.mahalanobis_norm
#         mahalanobis_fn_vmap = jax.vmap(mahalanobis_fn)
#         output_scale = mahalanobis_fn_vmap(obs_unbatch, jnp.zeros_like(m1))
#         error_estimate = output_scale * obs_unbatch.cov_sqrtm_lower
#
#         return error_estimate, obs_unbatch, cache
#
#     def complete(self, ssv, corr, /, vector_field, t, p):
#         fn = jax.vmap(type(self.corr).complete, in_axes=(0, 0, 0, None, None, None))
#         return fn(self.corr, ssv, corr, vector_field, t, p)
#
#     def extract(self, ssv, corr):
#         return ssv
#
#     def _cov_sqrtm_lower(self, cov_sqrtm_lower):
#         return cov_sqrtm_lower[:, self.ode_order, ...]


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
