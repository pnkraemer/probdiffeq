"""Test cases for ODE problems."""


from typing import Literal, NamedTuple

import diffeqzoo.ivps
import jax
import pytest_cases
import pytest_cases.filters


class Tag(NamedTuple):
    shape: Literal[(2,)]  # todo: scalar problems
    order: Literal[1]  # todo: second-order problems
    stiff: Literal[True, False]


@pytest_cases.case(tags=(Tag(shape=(2,), order=1, stiff=False),))
def case_lotka_volterra():
    f, u0, (t0, t1), f_args = diffeqzoo.ivps.lotka_volterra()
    t1 = 2.0

    @jax.jit
    def vf(x, *, t, p):  # pylint: disable=unused-argument
        return f(x, *p)

    # Only very short time-intervals are sufficient for a unit test.
    return vf, (u0,), t0, t1, f_args
