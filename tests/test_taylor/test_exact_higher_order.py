"""Test the exactness of differentiation-based routines on first-order problems."""


from probdiffeq.backend import numpy as np
from probdiffeq.backend import ode, testing
from probdiffeq.taylor import autodiff


@testing.case()
def case_forward_mode_recursive():
    return autodiff.forward_mode_recursive


@testing.case()
def case_taylor_mode_scan():
    return autodiff.taylor_mode_scan


@testing.fixture(name="pb_with_solution")
def fixture_pb_with_solution():
    vf, (u0, du0), (t0, _) = ode.ivp_van_der_pol_2nd()

    solution = np.load("./tests/test_taylor/data/van_der_pol_second_solution.npy")
    return (lambda *ys: vf(*ys, t=t0), (u0, du0)), solution


@testing.parametrize_with_cases("taylor_fun", cases=".", prefix="case_")
@testing.parametrize("num", [1, 3])
def test_approximation_identical_to_reference(pb_with_solution, taylor_fun, num):
    (f, init), solution = pb_with_solution

    derivatives = taylor_fun(f, init, num=num)
    for dy, dy_ref in zip(derivatives, solution):
        assert np.allclose(dy, dy_ref)
