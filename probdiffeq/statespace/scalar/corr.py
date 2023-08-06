# """Implementations for scalar initial value problems."""
#
# import functools
#
# import jax
# import jax.numpy as jnp
#
# from probdiffeq import _sqrt_util
# from probdiffeq.backend import statespace
# from probdiffeq.statespace import corr, variables
#
#
# def taylor_order_zero(*, ode_order):
#     return _ODEConstraint(
#         ode_order=ode_order,
#         linearise_fun=statespace.linearise_ode.constraint_0th(ode_order=ode_order),
#         string_repr=f"<TS0 with ode_order={ode_order}>",
#     )
#
#
# class _ODEConstraint(corr.Correction):
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
#         obs_like = statespace.random.qoi_like()
#         return ssv, obs_like
#
#     def estimate_error(self, ssv, corr, /, vector_field, t, p):
#         def f_wrapped(s):
#             return vector_field(*s, t=t, p=p)
#
#         A, b = self.linearise(f_wrapped, ssv.hidden_state.mean)
#         observed = statespace.cond.transform.marginalise(ssv.hidden_state, (A, b))
#
#         error_estimate = estimate_error(observed)
#         return error_estimate, observed, (A, b)
#
#     def complete(self, ssv, corr, /, vector_field, t, p):
#         A, b = corr
#         obs, (cor, _gn) = statespace.cond.transform.revert(ssv.hidden_state, (A, b))
#         u = statespace.random.qoi(cor)
#         ssv = variables.SSV(u, cor)
#         return ssv, obs
#
#     def extract(self, ssv, corr, /):
#         return ssv
#
#
# def estimate_error(observed, /):
#     zero_data = jnp.zeros_like(statespace.random.mean(observed))
#     output_scale = statespace.random.mahalanobis_norm(zero_data, rv=observed)
#     error_estimate_unscaled = statespace.random.standard_deviation(observed)
#     return output_scale * error_estimate_unscaled
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
# for nodetype in [_ODEConstraint]:
#     jax.tree_util.register_pytree_node(
#         nodetype=nodetype,
#         flatten_func=_constraint_flatten,
#         unflatten_func=functools.partial(_constraint_unflatten, nodetype=nodetype),
#     )
#


def correct_affine_qoi_noisy(rv, affine, *, stdev):
    # Read inputs
    A, b = affine

    # Apply observation model to covariance
    cov_sqrtm = rv.cov_sqrtm_lower
    cov_sqrtm_obs_nonsquare = jnp.dot(A, cov_sqrtm[0, ...])

    # Revert the conditional covariances
    cov_sqrtm_obs_upper, (
        cov_sqrtm_cor_upper,
        gain,
    ) = _sqrt_util.revert_conditional(
        R_X_F=cov_sqrtm_obs_nonsquare[None, :].T,
        R_X=rv.cov_sqrtm_lower.T,
        R_YX=jnp.ones((1, 1)) * stdev,
    )
    cov_sqrtm_obs = cov_sqrtm_obs_upper.T
    cov_sqrtm_cor = cov_sqrtm_cor_upper.T
    gain = gain[:, 0]  # "squeeze"; output shape is (), not (1,)

    # Gather the observed variable
    mean_obs = jnp.dot(A, rv.mean[0, ...]) + b
    observed = variables.NormalQOI(mean=mean_obs, cov_sqrtm_lower=cov_sqrtm_obs)

    # Gather the corrected variable
    mean_cor = rv.mean - gain * mean_obs
    corrected = variables.NormalHiddenState(
        mean=mean_cor, cov_sqrtm_lower=cov_sqrtm_cor
    )
    return observed, (corrected, gain)


def correct_affine_ode_2nd(rv, affine):
    # Read inputs
    A, b = affine

    # Apply observation model to covariance
    cov_sqrtm = rv.cov_sqrtm_lower
    cov_sqrtm_obs_nonsquare = cov_sqrtm[2, ...] - jnp.dot(A, cov_sqrtm[0, ...])

    # Revert the conditional covariances
    cov_sqrtm_obs_upper, (
        cov_sqrtm_cor_upper,
        gain,
    ) = _sqrt_util.revert_conditional_noisefree(
        R_X_F=cov_sqrtm_obs_nonsquare[None, :].T, R_X=rv.cov_sqrtm_lower.T
    )
    cov_sqrtm_obs = cov_sqrtm_obs_upper.T
    cov_sqrtm_cor = cov_sqrtm_cor_upper.T
    gain = gain[:, 0]  # "squeeze"; output shape is (), not (1,)

    # Gather the observed variable
    mean_obs = rv.mean[2, ...] - jnp.dot(A, rv.mean[0, ...]) - b
    observed = variables.NormalQOI(mean=mean_obs, cov_sqrtm_lower=cov_sqrtm_obs)

    # Gather the corrected variable
    mean_cor = rv.mean - gain * mean_obs
    corrected = variables.NormalHiddenState(
        mean=mean_cor, cov_sqrtm_lower=cov_sqrtm_cor
    )
    return observed, (corrected, gain)
