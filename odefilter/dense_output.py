"""Do fun stuff with the solution objects."""

from functools import partial
from typing import TypeVar

import jax
import jax.numpy as jnp

S = TypeVar("S")  # Filter/smoother
R = TypeVar("R")  # RV


# todo: why do we need this function??
@partial(jax.jit, static_argnames=["shape"])
def sample(key, *, solution, solver, shape=()):
    return solver.strategy.sample(key, posterior=solution.posterior, shape=shape)


# todo: the functions herein should only depend on posteriors / strategies!


def offgrid_marginals_searchsorted(*, ts, solution, solver):
    """Dense output for a whole grid via jax.numpy.searchsorted.

    !!! warning
        The elements in ts and the elements in the solution grid must be disjoint.
        Otherwise, anything can happen and the solution will be incorrect.
        We do not check for this case! (Because we want to jit!)

    !!! warning
        The elements in ts must be strictly in (t0, t1).
        Again there is no check and anything can happen if you don't follow
        this rule.
    """
    # todo: support "method" argument to be passed to searchsorted.

    # side="left" and side="right" are equivalent
    # because we _assume_ that the point sets are disjoint.
    indices = jnp.searchsorted(solution.t, ts)

    # Solution slicing to the rescue
    solution_left = solution[indices - 1]
    solution_right = solution[indices]

    # Vmap to the rescue :) It does not like kw-only arguments, though.
    @jax.vmap
    def marginals_vmap(sprev, t, s):
        return offgrid_marginals(
            t=t, solution=s, solution_previous=sprev, solver=solver
        )

    return marginals_vmap(solution_left, ts, solution_right)


def offgrid_marginals(*, solution, t, solution_previous, solver):
    return solver.strategy.offgrid_marginals(
        marginals=solution.marginals,
        posterior_previous=solution_previous.posterior,
        t=t,
        t0=solution_previous.t,
        t1=solution.t,
        scale_sqrtm=solution.output_scale_sqrtm,
    )
