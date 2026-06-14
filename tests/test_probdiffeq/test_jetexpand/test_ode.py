"""Test the exactness of differentiation-based routines on first-order problems."""

from probdiffeq import probdiffeq
from probdiffeq.backend import np, ode, testing
from probdiffeq.backend.typing import Callable


@testing.fixture(name="problem_with_solution")
def fixture_problem_with_solution():
    """Load the three-body first-order problem with a precomputed reference solution."""
    vf, (u0,), (t0, _) = ode.ivp_three_body_1st()

    solution = np.load(
        "./tests/test_probdiffeq/test_jetexpand/data/three_body_first_solution.npy"
    )
    return (probdiffeq.ode(vf), (u0,), {"t": t0}), solution


@testing.fixture(name="num")
@testing.parametrize("num", [0, 1, 4])
def fixture_number_of_extensions(num):
    """Return the number of Taylor coefficient extensions."""
    return num


@testing.case()
def case_odejet_via_jvp(num):
    """Use the JVP-based jet expansion."""
    return probdiffeq.jetexpand_ode_via_jvp(num=num)


@testing.case()
def case_odejet_padded_scan(num):
    """Use the padded-scan jet expansion."""
    return probdiffeq.jetexpand_ode_padded_scan(num=num)


@testing.case()
def case_odejet_unroll(num):
    """Use the unrolled jet expansion."""
    return probdiffeq.jetexpand_ode_unroll(num=num)


@testing.parametrize_with_cases("taylor_fun", cases=".", prefix="case_odejet_")
def test_approximation_identical_to_reference_odejet(
    problem_with_solution, num, taylor_fun: Callable
) -> None:
    """Assert that the jet expansion matches the precomputed reference for the three-body problem."""
    (f, init, vf_kwargs), solution = problem_with_solution

    derivatives, _ = taylor_fun(f, init, **vf_kwargs)
    assert len(derivatives) == num + 1
    assert testing.allclose(derivatives, list(solution[: len(derivatives)]))


@testing.parametrize("num_doublings", [0, 1, 2])
def test_approximation_identical_to_reference_doubling(
    problem_with_solution, num_doublings
) -> None:
    """Separately test the doubling-function, because its API is different."""
    (f, init, vf_kwargs), solution = problem_with_solution

    expand = probdiffeq.jetexpand_ode_doubling_unroll(num_doublings=num_doublings)
    derivatives, _ = expand(f, init, **vf_kwargs)
    assert len(derivatives) == np.sum(2 ** np.arange(0, num_doublings + 1))
    assert testing.allclose(derivatives, list(solution[: len(derivatives)]))


@testing.parametrize_with_cases("taylor_fun", cases=".", prefix="case_")
def test_raises_error_for_non_ode_input(taylor_fun) -> None:
    """Assert that a raw function without the ode decorator raises a TypeError."""

    def f(y, /, *, t):
        del t
        return y

    with testing.raises(TypeError, match="Expected type"):
        taylor_fun(f, (1.0,), t=0.0)
