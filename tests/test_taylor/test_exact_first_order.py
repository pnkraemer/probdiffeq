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


@testing.case()
def case_taylor_mode_unroll():
    return taylor.odejet_unroll


@testing.fixture(name="pb_with_solution")
def fixture_pb_with_solution():
    vf, (u0,), (t0, _) = ode.ivp_three_body_1st()
    vf = functools.partial(vf, t=t0)

    solution = np.load("./tests/test_taylor/data/three_body_first_solution.npy")
    return (vf, (u0,)), solution


@testing.parametrize_with_cases("taylor_fun", cases=".", prefix="case_")
@testing.parametrize("num", [1, 6])
def test_approximation_identical_to_reference(pb_with_solution, taylor_fun, num):
    (f, init), solution = pb_with_solution

    derivatives = taylor_fun(f, init, num=num)
    assert len(derivatives) == num + 1
    assert testing.allclose(derivatives, list(solution[: len(derivatives)]))


@testing.parametrize("num_doublings", [1, 2])
def test_approximation_identical_to_reference_doubling(pb_with_solution, num_doublings):
    """Separately test the doubling-function, because its API is different."""
    (f, init), solution = pb_with_solution

    derivatives = taylor.odejet_doubling_unroll(f, init, num_doublings=num_doublings)
    assert len(derivatives) == np.sum(2 ** np.arange(0, num_doublings + 1))
    assert testing.allclose(derivatives, list(solution[: len(derivatives)]))
