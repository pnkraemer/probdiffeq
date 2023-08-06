# """Extrapolation behaviour for scalar state-space models."""
# from typing import Tuple
#
# import jax
# import jax.numpy as jnp
#
# from probdiffeq import _markov, _sqrt_util
# from probdiffeq.backend import statespace
# from probdiffeq.statespace import _ibm_util, extra, variables
#
#
# def ibm_discretise_fwd(
#     dts, /, *, num_derivatives, output_scale=1.0
# ) -> _markov.MarkovSeqPreconFwd:
#     """Construct the discrete transition densities of an IBM prior.
#
#     Initialises with a scaled standard normal distribution.
#     """
#     init = statespace.ssm_util.standard_normal(num_derivatives + 1, output_scale)
#     discretise = statespace.ssm_util.ibm_transitions(num_derivatives, output_scale)
#     cond, precon = jax.vmap(discretise)(dts)
#     # transitions_vmap = jax.vmap(ibm_transitions_precon, in_axes=(0, None, None))
#     # cond, precon = transitions_vmap(dts, num_derivatives, output_scale)
#
#     return _markov.MarkovSeqPreconFwd(
#         init=init, conditional=cond, preconditioner=precon
#     )
#
#
# def ibm_transitions_precon(dt, /, num_derivatives, output_scale):
#     """Compute the discrete transition densities for the IBM on a pre-specified grid."""
#     a, q_sqrtm = _ibm_util.system_matrices_1d(
#         num_derivatives=num_derivatives, output_scale=output_scale
#     )
#     q0 = jnp.zeros((num_derivatives + 1,))
#     noise = variables.NormalHiddenState(q0, q_sqrtm)
#     transitions = variables.ConditionalHiddenState(transition=a, noise=noise)
#
#     precon_fun = _ibm_util.preconditioner_prepare(num_derivatives=num_derivatives)
#     p, p_inv = precon_fun(dt)
#
#     return transitions, (p, p_inv)
#
#
# def extrapolate_precon_with_reversal(
#     rv,
#     conditional,
#     preconditioner: Tuple[jax.Array, jax.Array],
# ):
#     """Extrapolate and compute smoothing gains in a preconditioned model.
#
#     Careful: the reverse-conditional is preconditioned.
#     """
#     # Read quantities
#     a = conditional.transition
#     q0, q_sqrtm = conditional.noise.mean, conditional.noise.cov_sqrtm_lower
#     p, p_inv = preconditioner
#     m0, l0 = rv.mean, rv.cov_sqrtm_lower
#
#     # Apply preconditioner
#     m0_p = p_inv * m0
#     l0_p = p_inv[:, None] * l0
#
#     # Extrapolate with reversal
#     m_ext_p = a @ m0_p + q0
#     r_ext_p, (r_rev_p, gain_p) = _sqrt_util.revert_conditional(
#         R_X_F=(a @ l0_p).T,
#         R_X=l0_p.T,
#         R_YX=q_sqrtm.T,
#     )
#     l_ext_p = r_ext_p.T
#     l_rev_p = r_rev_p.T
#
#     # Catch up with the mean
#     m_rev_p = m0_p - gain_p @ m_ext_p
#
#     # Unapply preconditioner for the state variable
#     # (the system matrices remain preconditioned)
#     m_ext = p * m_ext_p
#     l_ext = p[:, None] * l_ext_p
#
#     # Gather and return variables
#     marginal = variables.NormalHiddenState(mean=m_ext, cov_sqrtm_lower=l_ext)
#     reversal_p = variables.NormalHiddenState(mean=m_rev_p, cov_sqrtm_lower=l_rev_p)
#     conditional = variables.ConditionalHiddenState(transition=gain_p, noise=reversal_p)
#     return marginal, conditional
#
#
# def extrapolate_precon(
#     rv,
#     conditional,
#     preconditioner: Tuple[jax.Array, jax.Array],
# ):
#     # Read quantities
#     a = conditional.transition
#     q0, q_sqrtm = conditional.noise.mean, conditional.noise.cov_sqrtm_lower
#     p, p_inv = preconditioner
#     m0, l0 = rv.mean, rv.cov_sqrtm_lower
#
#     # Apply preconditioner
#     m0_p = p_inv * m0
#     l0_p = p_inv[:, None] * l0
#
#     # Extrapolate with reversal
#     m_ext_p = a @ m0_p + q0
#     r_ext_p = _sqrt_util.sum_of_sqrtm_factors(R_stack=((a @ l0_p).T, q_sqrtm.T))
#     l_ext_p = r_ext_p.T
#
#     # Unapply preconditioner for the state variable
#     m_ext = p * m_ext_p
#     l_ext = p[:, None] * l_ext_p
#
#     # Gather and return variables
#     marginal = variables.NormalHiddenState(mean=m_ext, cov_sqrtm_lower=l_ext)
#     return marginal
#
