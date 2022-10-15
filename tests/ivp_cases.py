"""Test cases: IVPs."""

import jax
from diffeqzoo import ivps
from pytest_cases import case


@case
def problem_lotka():
    f, u0, (t0, t1), f_args = ivps.lotka_volterra()
    t1 = 0.2

    @jax.jit
    def vf(_t, x, *p):
        return f(x, *p)

    # Only very short time-intervals are sufficient for a unit test.
    return vf, (u0,), t0, t1, f_args
