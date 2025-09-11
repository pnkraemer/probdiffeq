"""Test the exactness of differentiation-based routines on first-order problems."""

from probdiffeq import taylor
from probdiffeq.backend import functools, ode, testing
from probdiffeq.backend import numpy as np


@testing.case()
def case_odejet_via_jvp():
    return taylor.odejet_via_jvp


@testing.case()
def case_taylor_mode_scan():
    return taylor.odejet_padded_scan


@testing.fixture(name="pb_with_solution")
def fixture_pb_with_solution():
    vf, (u0, du0), (t0, _) = ode.ivp_van_der_pol_2nd()
    vf = functools.partial(vf, t=t0)

    solution = np.load("./tests/test_taylor/data/van_der_pol_second_solution.npy")
    return (vf, (u0, du0)), solution


@testing.parametrize_with_cases("taylor_fun", cases=".", prefix="case_")
@testing.parametrize("num", [1, 3])
def test_approximation_identical_to_reference(pb_with_solution, taylor_fun, num):
    (f, init), solution = pb_with_solution

    derivatives = taylor_fun(f, init, num=num)
    assert len(derivatives) == num + 2
    assert testing.allclose(derivatives, list(solution[: len(derivatives)]))
