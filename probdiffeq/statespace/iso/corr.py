# """Corrections."""
# import functools
#
# import jax
# import jax.numpy as jnp
#
# from probdiffeq.statespace import corr
# from probdiffeq.statespace.iso import linearise_ode, variables
#
#
# def taylor_order_zero(*, ode_order):
#     fun = linearise_ode.constraint_0th(ode_order=ode_order)
#     return _IsoODEConstraint(
#         ode_order=ode_order,
#         linearise_fun=fun,
#         string_repr=f"<TS0 with ode_order={ode_order}>",
#     )
#
#
# class _IsoODEConstraint(corr.Correction):
#     def __init__(self, ode_order, linearise_fun, string_repr):
#         super().__init__(ode_order=ode_order)
#
#         self.linearise = linearise_fun
#         self.string_repr = string_repr
#
#     def __repr__(self):
#         return self.string_repr
#
#     def init(self, ssv, /):
#         bias_like = jnp.empty_like(ssv.hidden_state.mean[0, :])
#         chol_like = jnp.empty(())
#         obs_like = variables.IsoNormalQOI(bias_like, chol_like)
#         return ssv, obs_like
#
#     def estimate_error(self, ssv, corr, /, vector_field, t, p):
#         def f_wrapped(s):
#             return vector_field(*s, t=t, p=p)
#
#         A, b = self.linearise(f_wrapped, ssv.hidden_state.mean)
#         observed = variables.marginalise_deterministic_qoi(ssv.hidden_state, (A, b))
#
#         error_estimate = _estimate_error(observed)
#         return error_estimate, observed, (A, b)
#
#     def complete(self, ssv, corr, /, vector_field, t, p):
#         A, b = corr
#         obs, (cor, _gn) = variables.revert_deterministic_qoi(ssv.hidden_state, (A, b))
#
#         u = cor.mean[0, :]
#         ssv = variables.IsoSSV(u, cor)
#         return ssv, obs
#
#     def extract(self, ssv, corr, /):
#         return ssv
#
#
# def _estimate_error(obs, /):
#     mahalanobis_norm = obs.mahalanobis_norm(jnp.zeros_like(obs.mean))
#     output_scale = mahalanobis_norm / jnp.sqrt(obs.mean.size)
#
#     error_estimate_unscaled = obs.marginal_std() * jnp.ones_like(obs.mean)
#     return error_estimate_unscaled * output_scale
#
#
# def _constraint_flatten(node):
#     children = ()
#     aux = node.ode_order, node.linearise, node.string_repr
#     return children, aux
#
#
# def _constraint_unflatten(aux, _children, *, nodetype):
#     ode_order, lin, string_repr = aux
#     return nodetype(ode_order=ode_order, linearise_fun=lin, string_repr=string_repr)
#
#
# for nodetype in [_IsoODEConstraint]:
#     jax.tree_util.register_pytree_node(
#         nodetype=nodetype,
#         flatten_func=_constraint_flatten,
#         unflatten_func=functools.partial(_constraint_unflatten, nodetype=nodetype),
#     )
