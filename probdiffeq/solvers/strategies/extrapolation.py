"""Extrapolation model interfaces."""


import jax
import jax.numpy as jnp

from probdiffeq.impl import impl
from probdiffeq.solvers import markov


def ibm_adaptive(num_derivatives):
    output_scale = jnp.ones_like(impl.prototypes.output_scale())
    discretise = impl.ssm_util.ibm_transitions(num_derivatives, output_scale)
    return discretise, num_derivatives


def ibm_discretise_fwd(dts, /, *, num_derivatives):
    discretise = impl.ssm_util.ibm_transitions(num_derivatives)
    return jax.vmap(discretise)(dts)


def unit_markov_sequence(num_derivatives):
    cond = impl.ssm_util.identity_conditional(num_derivatives + 1)
    init = impl.ssm_util.standard_normal(num_derivatives + 1, 1.0)
    return markov.MarkovSeqRev(init=init, conditional=cond)
