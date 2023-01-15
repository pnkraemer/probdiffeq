"""Test cases for ODE problems."""


import dataclasses
from typing import Callable, Literal, NamedTuple, Tuple

import diffeqzoo.ivps
import diffrax
import jax
import pytest_cases
import pytest_cases.filters


class Tag(NamedTuple):
    shape: Literal[(2,)]  # todo: scalar problems
    order: Literal[1]  # todo: second-order problems
    stiff: Literal[True, False]


# todo: turn into a dataclass, which _has_ to be accessed as problem.t0, problem.u0,
#  etc.. Remove "args" field to ensure that the reference solution
#  always matches the problem. Otherwise, it might get hard to debug...
@dataclasses.dataclass
class ODEProblem:
    vector_field: Callable
    initial_values: Tuple
    t0: float
    t1: float
    args: Tuple
    solution: Callable


@pytest_cases.case(tags=(Tag(shape=(2,), order=1, stiff=False),))
def case_lotka_volterra():
    f, u0, (t0, t1), f_args = diffeqzoo.ivps.lotka_volterra()
    t1 = 2.0

    @jax.jit
    def vf(x, *, t, p):  # pylint: disable=unused-argument
        return f(x, *p)

    def vf_diffrax(t, y, args):
        return vf(y, t=t, p=args)

    term = diffrax.ODETerm(vf_diffrax)
    solver = diffrax.Dopri5()
    solution_object = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=0.1,
        y0=u0,
        args=f_args,
        saveat=diffrax.SaveAt(dense=True),
        stepsize_controller=diffrax.PIDController(atol=1e-10, rtol=1e-10),
    )

    @jax.jit
    def solution(t):
        return solution_object.evaluate(t)

    # Only very short time-intervals are sufficient for a unit test.
    return ODEProblem(vf, (u0,), t0, t1, f_args, solution)
