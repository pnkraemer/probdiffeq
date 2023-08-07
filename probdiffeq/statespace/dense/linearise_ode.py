# """Linearisation."""
#
# import functools
#
# import jax
# import jax.numpy as jnp
#
# from probdiffeq import _sqrt_util
# from probdiffeq.statespace.dense import linearise, variables
#
# # todo:
# #  statistical linear regression (zeroth order)
# #  statistical linear regression (cov-free)
# #  statistical linear regression (Jacobian)
# #  statistical linear regression (Bayesian cubature)
#
#
# def constraint_0th(*, ode_shape, ode_order):
#     def linearise_fun_wrapped(fun, mean):
#         select = functools.partial(_select_derivative, ode_shape=ode_shape)
#         a0 = functools.partial(select, i=slice(0, ode_order))
#         a1 = functools.partial(select, i=ode_order)
#
#         if jnp.shape(a0(mean)) != (ode_order,) + ode_shape:
#             raise ValueError(f"{jnp.shape(a0(mean))} != {(ode_order,) + ode_shape}")
#
#         fx = linearise.ts0(fun, a0(mean))
#         return _autobatch_linop(a1), -fx
#
#     return linearise_fun_wrapped
#
#
# def constraint_1st(*, ode_shape, ode_order):
#     def new(fun, mean, /):
#         select = functools.partial(_select_derivative, ode_shape=ode_shape)
#         a0 = functools.partial(select, i=slice(0, ode_order))
#         a1 = functools.partial(select, i=ode_order)
#
#         if jnp.shape(a0(mean)) != (ode_order,) + ode_shape:
#             raise ValueError(f"{jnp.shape(a0(mean))} != {(ode_order,) + ode_shape}")
#
#         jvp, fx = linearise.ts1(fun, a0(mean))
#
#         def A(x):
#             x1 = a1(x)
#             x0 = a0(x)
#             return x1 - jvp(x0)
#
#         return _autobatch_linop(A), -fx
#
#     return new
#
#
# def constraint_statistical_0th(*, ode_shape, cubature_fun):
#     cubature = cubature_fun(input_shape=ode_shape)
#     linearise_fun = functools.partial(linearise.slr0, cubature_rule=cubature)
#
#     def new(fun, rv, /):
#         # Projection functions
#         select = functools.partial(_select_derivative, ode_shape=ode_shape)
#         a0 = _autobatch_linop(functools.partial(select, i=0))
#         a1 = _autobatch_linop(functools.partial(select, i=1))
#
#         # Extract the linearisation point
#         m0, r_0_nonsquare = a0(rv.mean), a0(rv.cov_sqrtm_lower)
#         r_0_square = _sqrt_util.triu_via_qr(r_0_nonsquare.T)
#         linearisation_pt = variables.DenseNormal(m0, r_0_square.T, target_shape=None)
#
#         # Gather the variables and return
#         noise = linearise_fun(fun, linearisation_pt)
#         bias = variables.DenseNormal(
#             -noise.mean, noise.cov_sqrtm_lower, target_shape=noise.target_shape
#         )
#         return a1, bias
#
#     return new
#
#
# def constraint_statistical_1st(*, ode_shape, cubature_fun):
#     cubature = cubature_fun(input_shape=ode_shape)
#     linearise_fun = functools.partial(linearise.slr1, cubature_rule=cubature)
#
#     def new(fun, rv):
#         # Projection functions
#         select = functools.partial(_select_derivative, ode_shape=ode_shape)
#         a0 = _autobatch_linop(functools.partial(select, i=0))
#         a1 = _autobatch_linop(functools.partial(select, i=1))
#
#         # Extract the linearisation point
#         m0, r_0_nonsquare = a0(rv.mean), a0(rv.cov_sqrtm_lower)
#         r_0_square = _sqrt_util.triu_via_qr(r_0_nonsquare.T)
#         linearisation_pt = variables.DenseNormal(m0, r_0_square.T, target_shape=None)
#
#         # Gather the variables and return
#         J, noise = linearise_fun(fun, linearisation_pt)
#
#         def A(x):
#             return a1(x) - J(a0(x))
#
#         mean, cov_lower = noise.mean, noise.cov_sqrtm_lower
#         bias = variables.DenseNormal(-mean, cov_lower, target_shape=noise.target_shape)
#         return _autobatch_linop(A), bias
#
#     return new
#
#
# def _select_derivative(x, i, *, ode_shape):
#     (d,) = ode_shape
#     x_reshaped = jnp.reshape(x, (-1, d), order="F")
#     return x_reshaped[i, ...]
#
#
# def _autobatch_linop(fun):
#     def fun_(x):
#         if jnp.ndim(x) > 1:
#             return jax.vmap(fun_, in_axes=1, out_axes=1)(x)
#         return fun(x)
#
#     return fun_
