"""Cases for problems."""


import jax.numpy as jnp
import pytest_cases

from odefilter import (
    backends,
    controls,
    implementations,
    information,
    inits,
    ivpsolve,
    odefilters,
    solvers,
)


@pytest_cases.case
def case_problem_logistic():
    return lambda x, t: x * (1 - x), (jnp.asarray([0.5]),), 0.0, 10.0, ()
