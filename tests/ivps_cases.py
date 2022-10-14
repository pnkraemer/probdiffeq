"""Test cases: IVPs."""

import jax
from diffeqzoo import ivps
from pytest_cases import case


@case
def problem_lotka():
    f, u0, tspan, f_args = ivps.lotka_volterra()

    @jax.jit
    def vf(_t, x, *p):
        return f(x, *p)

    return vf, (u0,), *tspan, f_args
