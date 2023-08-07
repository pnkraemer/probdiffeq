# """Linearisation."""
#
# import jax
#
# from probdiffeq import _sqrt_util
# from probdiffeq.statespace.dense import variables
#
# # todo:
# #  statistical linear regression (Jacobian)
# #  statistical linear regression (Bayesian cubature)
#
#
# def ts0(fn, m):
#     """Linearise a function with a zeroth-order Taylor series."""
#     return fn(m)
#
#
# def ts1(fn, m):
#     """Linearise a function with a first-order Taylor series."""
#     b, jvp = jax.linearize(fn, m)
#     return jvp, b - jvp(m)
#
#
# def slr1(fn, x, *, cubature_rule):
#     """Linearise a function with first-order statistical linear regression."""
#     # Create sigma-points
#     pts_centered = cubature_rule.points @ x.cov_sqrtm_lower.T
#     pts = x.mean[None, :] + pts_centered
#     pts_centered_normed = pts_centered * cubature_rule.weights_sqrtm[:, None]
#
#     # Evaluate the nonlinear function
#     fx = jax.vmap(fn)(pts)
#     fx_mean = cubature_rule.weights_sqrtm**2 @ fx
#     fx_centered = fx - fx_mean[None, :]
#     fx_centered_normed = fx_centered * cubature_rule.weights_sqrtm[:, None]
#
#     # Compute statistical linear regression matrices
#     _, (cov_sqrtm_cond, linop_cond) = _sqrt_util.revert_conditional_noisefree(
#         R_X_F=pts_centered_normed, R_X=fx_centered_normed
#     )
#     mean_cond = fx_mean - linop_cond @ x.mean
#     rv_cond = variables.DenseNormal(mean_cond, cov_sqrtm_cond.T, target_shape=None)
#     return lambda v: linop_cond @ v, rv_cond
