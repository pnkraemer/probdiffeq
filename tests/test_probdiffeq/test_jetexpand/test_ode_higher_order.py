"""Test the exactness of differentiation-based routines on first-order problems."""

from probdiffeq import probdiffeq
from probdiffeq.backend import np, ode, testing


@testing.fixture(name="num")
@testing.parametrize("num", [0, 1, 4])
def fixture_number_of_extensions(num):
    return num


@testing.case()
def case_jetexpand_ode_via_jvp(num):
    return probdiffeq.jetexpand_ode_via_jvp(num=num)


@testing.case()
def case_taylor_mode_scan(num):
    return probdiffeq.jetexpand_ode_padded_scan(num=num)


@testing.fixture(name="problem_with_solution")
def fixture_problem_with_solution():
    vf, (u0, du0), (t0, _) = ode.ivp_van_der_pol_2nd()

    path = "./tests/test_probdiffeq/test_jetexpand/data/van_der_pol_second_solution.npy"
    solution = np.load(path)
    return (probdiffeq.ode(vf), (u0, du0), t0), solution


@testing.parametrize_with_cases("taylor_fun", cases=".", prefix="case_")
def test_approximation_identical_to_reference(problem_with_solution, taylor_fun, num):
    (f, (u0, du0), t0), solution = problem_with_solution

    derivatives = taylor_fun(f, (u0, du0), t=t0)
    assert len(derivatives) == num + 2
    assert testing.allclose(derivatives, list(solution[: len(derivatives)]))
