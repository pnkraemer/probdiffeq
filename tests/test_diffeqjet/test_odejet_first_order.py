"""Test the exactness of differentiation-based routines on first-order problems."""

from probdiffeq import probdiffeq
from probdiffeq.backend import func, np, ode, testing


@testing.case()
def case_odejet_via_jvp():
    return probdiffeq.jetexpand_ode_via_jvp


@testing.case()
def case_odejet_padded_scan():
    return probdiffeq.jetexpand_ode_padded_scan


@testing.case()
def case_odejet_unroll():
    return probdiffeq.jetexpand_ode_unroll


@testing.fixture(name="pb_with_solution")
def fixture_pb_with_solution():
    vf, (u0,), (t0, _) = ode.ivp_three_body_1st()
    vf = func.partial(vf, t=t0)

    solution = np.load("./tests/test_diffeqjet/data/three_body_first_solution.npy")
    return (vf, (u0,)), solution


@testing.parametrize_with_cases("taylor_fun", cases=".", prefix="case_odejet_")
@testing.parametrize("num", [0, 1, 4])
def test_approximation_identical_to_reference_odejet(
    pb_with_solution, taylor_fun, num
) -> None:
    (f, init), solution = pb_with_solution

    derivatives = taylor_fun(f, init, num=num)
    assert len(derivatives) == num + 1
    assert testing.allclose(derivatives, list(solution[: len(derivatives)]))


@testing.case()
def case_doubling_odejet_unroll():
    return probdiffeq.jetexpand_ode_doubling_unroll


@testing.parametrize_with_cases("taylor_fun", cases=".", prefix="case_doubling_odejet_")
@testing.parametrize("num_doublings", [0, 1, 2])
def test_approximation_identical_to_reference_doubling(
    pb_with_solution, taylor_fun, num_doublings
) -> None:
    """Separately test the doubling-function, because its API is different."""
    (f, init), solution = pb_with_solution

    derivatives = taylor_fun(f, init, num_doublings=num_doublings)
    assert len(derivatives) == np.sum(2 ** np.arange(0, num_doublings + 1))
    assert testing.allclose(derivatives, list(solution[: len(derivatives)]))
