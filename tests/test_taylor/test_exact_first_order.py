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


@testing.case()
def case_taylor_mode_unroll():
    return autodiff.taylor_mode_unroll


@testing.fixture(name="pb_with_solution")
def fixture_pb_with_solution():
    vf, (u0,), (t0, _) = ode.ivp_three_body_1st()

    solution = np.load("./tests/test_taylor/data/three_body_first_solution.npy")
    return (lambda y: vf(y, t=t0), (u0,)), solution


@testing.parametrize_with_cases("taylor_fun", cases=".", prefix="case_")
@testing.parametrize("num", [1, 6])
def test_approximation_identical_to_reference(pb_with_solution, taylor_fun, num):
    (f, init), solution = pb_with_solution

    derivatives = taylor_fun(f, init, num=num)
    assert len(derivatives) == num + 1
    for dy, dy_ref in zip(derivatives, solution):
        assert np.allclose(dy, dy_ref)


@testing.parametrize("num_doublings", [1, 2])
def test_approximation_identical_to_reference_doubling(pb_with_solution, num_doublings):
    """Separately test the doubling-function, because its API is different."""
    (f, init), solution = pb_with_solution

    derivatives = autodiff.taylor_mode_doubling(f, init, num_doublings=num_doublings)
    assert len(derivatives) == np.sum(2 ** np.arange(0, num_doublings + 1))
    for dy, dy_ref in zip(derivatives, solution):
        assert np.allclose(dy, dy_ref)
