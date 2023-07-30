"""Tests for interaction with the solution object."""
import diffeqzoo.ivps
import jax
import jax.numpy as jnp

from probdiffeq import ivpsolve, test_util
from probdiffeq.backend import testing


@testing.fixture(name="problem")
def fixture_problem():
    f, u0, (t0, _), f_args = diffeqzoo.ivps.lotka_volterra()
    t1 = 2.0  # Short time-intervals are sufficient for this test.

    @jax.jit
    def vf(x, *, t, p):  # pylint: disable=unused-argument
        return f(x, *p)

    return vf, u0, (t0, t1), f_args


@testing.fixture(name="approximate_solution")
def fixture_approximate_solution(problem):
    vf, u0, (t0, t1), f_args = problem
    solver = test_util.generate_solver(num_derivatives=1)
    sol = ivpsolve.solve_with_python_while_loop(
        vf,
        (u0,),
        t0=t0,
        t1=t1,
        parameters=f_args,
        solver=solver,
        atol=1e-2,
        rtol=1e-2,
    )
    return sol, solver


def test_getitem_vmap_result_possible(problem):
    vf, u0, (t0, t1), f_args = problem
    solver = test_util.generate_solver(num_derivatives=1)
    save_at = jnp.linspace(t0, t1, endpoint=True, num=4)

    @jax.vmap
    def solve(init):
        return ivpsolve.solve_and_save_at(
            vf,
            (init,),
            save_at=save_at,
            parameters=f_args,
            solver=solver,
            atol=1e-2,
            rtol=1e-2,
        )

    solution_vmapped = solve(jnp.stack((u0, u0 + 0.1, u0 + 0.2)))

    assert isinstance(solution_vmapped[0], type(solution_vmapped))
    assert isinstance(solution_vmapped[1], type(solution_vmapped))
    assert isinstance(solution_vmapped[2], type(solution_vmapped))


def test_getitem_terminal_values_possible(approximate_solution):
    solution, _ = approximate_solution
    solution_t1 = solution[-1]
    assert isinstance(solution_t1, type(solution))


@testing.parametrize("item", [-2, 0, slice(1, -1, 1)])
def test_getitem_non_batched_solution_impossible(approximate_solution, item):
    solution, _ = approximate_solution
    # Allowed slicing:
    # solution_t1 is not batched now, so further slicing should be impossible
    solution_t1 = solution[-1]

    with testing.raises(ValueError, match="not batched"):
        _ = solution_t1[item]


@testing.parametrize("item", [-2, 0, slice(1, -1, 1)])
def test_getitem_nonterminal_values_impossible(approximate_solution, item):
    solution, _ = approximate_solution

    with testing.raises(ValueError, match="non-terminal"):
        _ = solution[item]


def test_marginal_nth_derivative_of_solution(approximate_solution):
    """Assert that each $n$th derivative matches the quantity of interest's shape."""
    sol, _ = approximate_solution

    # Assert that the marginals have the same shape as the qoi.
    for i in (0, 1):
        derivatives = sol.marginals.marginal_nth_derivative(i)
        assert derivatives.mean.shape == sol.u.shape

    # if the requested derivative is not in the state-space model, raise a ValueError
    with testing.raises(ValueError):
        sol.marginals.marginal_nth_derivative(100)
