"""Extrapolation model interfaces."""


import jax.numpy as jnp

from probdiffeq.impl import impl


def ibm_adaptive(num_derivatives):
    output_scale = jnp.ones_like(impl.prototypes.output_scale())
    discretise = impl.ssm_util.ibm_transitions(num_derivatives, output_scale)
    return discretise, num_derivatives
