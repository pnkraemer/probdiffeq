"""Extrapolation model interfaces."""


import jax
import jax.numpy as jnp

from probdiffeq.impl import impl


def ibm_adaptive(num_derivatives):
    output_scale = jnp.ones_like(impl.prototypes.output_scale())
    discretise = impl.ssm_util.ibm_transitions(num_derivatives, output_scale)
    return discretise, num_derivatives


def ibm_discretised(ts, *, num_derivatives):
    discretise, _ = ibm_adaptive(num_derivatives)
    transitions = jax.vmap(discretise)(jnp.diff(ts))

    output_scale = jnp.ones_like(impl.prototypes.output_scale())
    init = impl.ssm_util.standard_normal(num_derivatives + 1, output_scale=output_scale)
    return init, transitions
