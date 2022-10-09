"""Cases for problems."""


import jax.numpy as jnp
import pytest_cases


@pytest_cases.case
def case_problem_logistic():
    return lambda x, t: x * (1 - x), (jnp.asarray([0.5]),), 0.0, 10.0, ()
