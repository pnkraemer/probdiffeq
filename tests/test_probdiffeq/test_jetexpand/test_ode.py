"""Test the exactness of differentiation-based routines on first-order problems."""

from probdiffeq import probdiffeq
from probdiffeq.backend import np, ode, testing
from probdiffeq.backend.typing import Callable


@testing.fixture(name="problem_with_solution")
def fixture_problem_with_solution():
    vf, (u0,), (t0, _) = ode.ivp_three_body_1st()

    solution = np.load(
        "./tests/test_probdiffeq/test_jetexpand/data/three_body_first_solution.npy"
    )
    return (probdiffeq.ode(vf), (u0,), {"t": t0}), solution


@testing.fixture(name="num")
@testing.parametrize("num", [0, 1, 4])
def fixture_number_of_extensions(num):
    return num


@testing.case()
def case_odejet_via_jvp(num):
    return probdiffeq.jetexpand_ode_via_jvp(num=num)


@testing.case()
def case_odejet_padded_scan(num):
    return probdiffeq.jetexpand_ode_padded_scan(num=num)


@testing.case()
def case_odejet_unroll(num):
    return probdiffeq.jetexpand_ode_unroll(num=num)


@testing.parametrize_with_cases("taylor_fun", cases=".", prefix="case_odejet_")
def test_approximation_identical_to_reference_odejet(
    problem_with_solution, num, taylor_fun: Callable
) -> None:
    (f, init, vf_kwargs), solution = problem_with_solution

    derivatives = taylor_fun(f, init, **vf_kwargs)
    assert len(derivatives) == num + 1
    assert testing.allclose(derivatives, list(solution[: len(derivatives)]))


@testing.parametrize("num_doublings", [0, 1, 2])
def test_approximation_identical_to_reference_doubling(
    problem_with_solution, num_doublings
) -> None:
    """Separately test the doubling-function, because its API is different."""
    (f, init, vf_kwargs), solution = problem_with_solution

    expand = probdiffeq.jetexpand_ode_doubling_unroll(num_doublings=num_doublings)
    derivatives = expand(f, init, **vf_kwargs)
    assert len(derivatives) == np.sum(2 ** np.arange(0, num_doublings + 1))
    assert testing.allclose(derivatives, list(solution[: len(derivatives)]))


@testing.parametrize_with_cases("taylor_fun", cases=".", prefix="case_")
def test_raises_error_for_non_ode_input(taylor_fun, num) -> None:
    def f(y, /, *, t):
        return y

    with testing.raises(TypeError, match="Expected type"):
        taylor_fun(f, (1.0,), t=0.0)
