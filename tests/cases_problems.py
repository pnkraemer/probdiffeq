"""Cases for problems."""


import jax
import pytest_cases
from diffeqzoo import ivps


@pytest_cases.case
def case_lv():

    f, u0, tspan, f_args = ivps.lotka_volterra()

    return jax.jit(lambda x, t, *p: f(x, *p)), (u0,), *tspan, f_args
