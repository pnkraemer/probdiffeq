# """Variables."""
#
# import jax
# import jax.numpy as jnp
#
# from probdiffeq import _sqrt_util
# from probdiffeq.statespace import variables
#
#
# def merge_conditionals(previous, incoming, /):
#     if previous.transition.ndim > 2:
#         return jax.vmap(merge_conditionals)(previous, incoming)
#
#     A = previous.transition
#     (b, B_sqrtm_lower) = previous.noise.mean, previous.noise.cov_sqrtm_lower
#
#     C = incoming.transition
#     (d, D_sqrtm) = (incoming.noise.mean, incoming.noise.cov_sqrtm_lower)
#
#     g = A @ C
#     xi = A @ d + b
#     R_stack = ((A @ D_sqrtm).T, B_sqrtm_lower.T)
#     Xi = _sqrt_util.sum_of_sqrtm_factors(R_stack=R_stack).T
#
#     noise = NormalHiddenState(mean=xi, cov_sqrtm_lower=Xi)
#     return ConditionalHiddenState(g, noise=noise)
#
#
# def identity_conditional(ndim) -> "ConditionalHiddenState":
#     transition = jnp.eye(ndim)
#
#     mean = jnp.zeros((ndim,))
#     cov_sqrtm = jnp.zeros((ndim, ndim))
#     noise = NormalHiddenState(mean, cov_sqrtm)
#     return ConditionalHiddenState(transition, noise)
#
#
# def standard_normal(ndim, *, output_scale=1.0):
#     mean = jnp.zeros((ndim,))
#     cov_sqrtm = jnp.eye(ndim) * output_scale
#     return NormalHiddenState(mean, cov_sqrtm)
#
#
# def marginalise_deterministic_qoi(rv, trafo):
#     A, b = trafo
#     mean, cov_sqrtm_lower = rv.mean, rv.cov_sqrtm_lower
#
#     cov_sqrtm_lower_new = _sqrt_util.triu_via_qr(A(cov_sqrtm_lower)[:, None])
#     cov_sqrtm_lower_squeezed = jnp.reshape(cov_sqrtm_lower_new, ())
#     return NormalQOI(A(mean) + b, cov_sqrtm_lower_squeezed)
#
#
# def revert_deterministic_qoi(rv, trafo):
#     # Extract information
#     A, b = trafo
#     mean, cov_sqrtm_lower = rv.mean, rv.cov_sqrtm_lower
#
#     # QR-decomposition
#     # (todo: rename revert_conditional_noisefree to revert_transformation_cov_sqrt())
#     r_obs, (r_cor, gain) = _sqrt_util.revert_conditional_noisefree(
#         R_X_F=A(cov_sqrtm_lower)[:, None], R_X=cov_sqrtm_lower.T
#     )
#     cov_sqrtm_lower_obs = jnp.reshape(r_obs, ())
#     cov_sqrtm_lower_cor = r_cor.T
#     gain = jnp.squeeze(gain, axis=-1)
#
#     # Gather terms and return
#     m_cor = mean - gain * (A(mean) + b)
#     corrected = NormalHiddenState(m_cor, cov_sqrtm_lower_cor)
#     observed = NormalQOI(A(mean) + b, cov_sqrtm_lower_obs)
#     return observed, (corrected, gain)
#
#
# @jax.tree_util.register_pytree_node_class
# class ConditionalHiddenState(variables.Conditional):
#     def __call__(self, x, /):
#         if self.transition.ndim > 2:
#             return jax.vmap(ConditionalHiddenState.__call__)(self, x)
#
#         m = self.transition @ x + self.noise.mean
#         return NormalHiddenState(m, self.noise.cov_sqrtm_lower)
#
#     def scale_covariance(self, output_scale):
#         noise = self.noise.scale_covariance(output_scale=output_scale)
#         return ConditionalHiddenState(transition=self.transition, noise=noise)
#
#     def marginalise(self, rv, /):
#         # Todo: this auto-batch is a bit hacky,
#         #  but single-handedly replaces the entire BatchConditional class
#         if rv.mean.ndim > 1:
#             return jax.vmap(ConditionalHiddenState.marginalise)(self, rv)
#
#         m0, l0 = rv.mean, rv.cov_sqrtm_lower
#
#         m_new = self.transition @ m0 + self.noise.mean
#         l_new = _sqrt_util.sum_of_sqrtm_factors(
#             R_stack=((self.transition @ l0).T, self.noise.cov_sqrtm_lower.T)
#         ).T
#
#         return NormalHiddenState(m_new, l_new)
#
#
# @jax.tree_util.register_pytree_node_class
# class ConditionalQOI(variables.Conditional):
#     def __call__(self, x, /):
#         if self.transition.ndim > 1:
#             return jax.vmap(ConditionalQOI.__call__)(self, x)
#         m = self.transition * x + self.noise.mean
#         return NormalHiddenState(m, self.noise.cov_sqrtm_lower)
#
#
# @jax.tree_util.register_pytree_node_class
# class NormalQOI(variables.Normal):
#     # Normal RV. Shapes (), (). No QOI.
#
#     def transform_unit_sample(self, base, /):
#         m, l_sqrtm = self.mean, self.cov_sqrtm_lower
#         return m + l_sqrtm * base
#
#     def condition_on_qoi_observation(self, u, /, observation_std):
#         raise NotImplementedError
#
#     def extract_qoi_from_sample(self, u, /):
#         raise NotImplementedError
#
#     def scale_covariance(self, output_scale):
#         return NormalQOI(self.mean, output_scale * self.cov_sqrtm_lower)
#
#     def logpdf(self, u, /):
#         if jnp.ndim(u) > 0:
#             return jax.vmap(NormalQOI.logpdf)(self, u)
#
#         x1 = 2.0 * jnp.log(self.marginal_stds())  # logdet
#         x2 = self.mahalanobis_norm_squared(u)
#         x3 = jnp.log(jnp.pi * 2)
#         return -0.5 * (x1 + x2 + x3)
#
#     def marginal_stds(self):
#         return jnp.abs(self.cov_sqrtm_lower)
#
#     def mahalanobis_norm_squared(self, u, /):
#         res_white_scalar = self.residual_white(u)
#         return res_white_scalar**2.0
#
#     def mahalanobis_norm(self, u, /):
#         res_white_scalar = self.residual_white(u)
#         return jnp.abs(res_white_scalar)
#
#     def residual_white(self, u, /):
#         obs_pt, l_obs = self.mean, self.cov_sqrtm_lower
#         res_white = (obs_pt - u) / l_obs
#         return res_white
#
#     def marginal_nth_derivative(self, n):
#         raise NotImplementedError
#
#
# @jax.tree_util.register_pytree_node_class
# class SSV(variables.SSV):
#     # Normal RV. Shapes (n,), (n,n); zeroth state is the QOI.
#
#     def observe_qoi(self, observation_std):
#         # what is this for? batched calls? If so, that seems wrong.
#         #  the scalar state should not worry about the context it is called in.
#         if self.hidden_state.cov_sqrtm_lower.ndim > 2:
#             fn = SSV.observe_qoi
#             fn_vmap = jax.vmap(fn, in_axes=(0, None), out_axes=(0, 0))
#             return fn_vmap(self, observation_std)
#
#         hc = self.hidden_state.cov_sqrtm_lower[0]
#         m_obs = self.hidden_state.mean[0]
#
#         r_yx = observation_std  # * jnp.eye(1)
#         r_obs_mat, (r_cor, gain_mat) = _sqrt_util.revert_conditional(
#             R_X=self.hidden_state.cov_sqrtm_lower.T,
#             R_X_F=hc[:, None],
#             R_YX=r_yx[None, None],
#         )
#         r_obs = jnp.reshape(r_obs_mat, ())
#         gain = jnp.reshape(gain_mat, (-1,))
#
#         m_cor = self.hidden_state.mean - gain * m_obs
#
#         obs = NormalQOI(m_obs, r_obs.T)
#         cor = NormalHiddenState(m_cor, r_cor.T)
#         return obs, ConditionalQOI(gain, cor)
#
#
# @jax.tree_util.register_pytree_node_class
# class NormalHiddenState(variables.Normal):
#     def logpdf(self, u, /):
#         raise NotImplementedError
#
#     def mahalanobis_norm(self, u, /):
#         raise NotImplementedError
#
#     def scale_covariance(self, output_scale):
#         return NormalHiddenState(
#             mean=self.mean,
#             cov_sqrtm_lower=output_scale[..., None, None] * self.cov_sqrtm_lower,
#         )
#
#     def transform_unit_sample(self, base, /):
#         m, l_sqrtm = self.mean, self.cov_sqrtm_lower
#         return (m[..., None] + l_sqrtm @ base[..., None])[..., 0]
#
#     def marginal_nth_derivative(self, n):
#         if self.mean.ndim > 1:
#             # if the variable has batch-axes, vmap the result
#             fn = NormalHiddenState.marginal_nth_derivative
#             vect_fn = jax.vmap(fn, in_axes=(0, None))
#             return vect_fn(self, n)
#
#         if n >= self.mean.shape[0]:
#             msg = f"The {n}th derivative not available in the state-space variable."
#             raise ValueError(msg)
#
#         mean = self.mean[n]
#         cov_sqrtm_lower_nonsquare = self.cov_sqrtm_lower[n, :]
#         cov_sqrtm_lower = _sqrt_util.triu_via_qr(cov_sqrtm_lower_nonsquare[:, None]).T
#         return NormalQOI(mean=mean, cov_sqrtm_lower=jnp.reshape(cov_sqrtm_lower, ()))
#
#     def extract_qoi_from_sample(self, u, /):
#         if u.ndim == 1:
#             return u[0]
#         return jax.vmap(self.extract_qoi_from_sample)(u)
#
#     def cov_dense(self):
#         if self.cov_sqrtm_lower.ndim > 2:
#             return jax.vmap(NormalHiddenState.cov_dense)(self)
#
#         return self.cov_sqrtm_lower @ self.cov_sqrtm_lower.T
