"""Prior models."""


from probdiffeq.backend import functools
from probdiffeq.backend import numpy as np
from probdiffeq.impl import impl
from probdiffeq.solvers import markov


def ibm_adaptive(num_derivatives, output_scale=None):
    """Construct an adaptive(/continuous-time), multiply-integrated Wiener process."""
    output_scale = output_scale or np.ones_like(impl.prototypes.output_scale())
    discretise = impl.ssm_util.ibm_transitions(num_derivatives, output_scale)
    return discretise, num_derivatives


def ibm_discretised(ts, *, num_derivatives, output_scale=None):
    """Compute a time-discretised, multiply-integrated Wiener process."""
    discretise, _ = ibm_adaptive(num_derivatives, output_scale=output_scale)
    transitions, (p, p_inv) = functools.vmap(discretise)(np.diff(ts))

    preconditioner_apply_vmap = functools.vmap(impl.ssm_util.preconditioner_apply_cond)
    conditionals = preconditioner_apply_vmap(transitions, p, p_inv)

    output_scale = np.ones_like(impl.prototypes.output_scale())
    init = impl.ssm_util.standard_normal(num_derivatives + 1, output_scale=output_scale)
    return markov.MarkovSeq(init, conditionals)
