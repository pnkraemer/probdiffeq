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
    return ivpsolve.solve_with_python_while_loop(
        vf,
        (u0,),
        t0=t0,
        t1=t1,
        parameters=f_args,
        solver=solver,
        output_scale=1.0,
        atol=1e-2,
        rtol=1e-2,
    )


def test_getitem_possible_for_terminal_values(approximate_solution):
    solution_t1 = approximate_solution[-1]
    assert isinstance(solution_t1, type(approximate_solution))


@testing.parametrize("item", [-2, 0, slice(1, -1, 1)])
def test_getitem_impossible_for_nonterminal_values(approximate_solution, item):
    with testing.raises(ValueError, match="non-terminal"):
        _ = approximate_solution[item]


@testing.parametrize("item", [-1, -2, 0, slice(1, -1, 1)])
def test_getitem_impossible_at_single_time_for_any_item(approximate_solution, item):
    # Allowed slicing:
    # solution_t1 is not batched now, so further slicing should be impossible
    solution_t1 = approximate_solution[-1]

    with testing.raises(ValueError, match="not batched"):
        _ = solution_t1[item]


def test_iter_impossible(approximate_solution):
    with testing.raises(ValueError, match="not batched"):
        for _ in approximate_solution:
            pass


@testing.fixture(name="approximate_solution_batched")
def fixture_approximate_solution_batched(problem):
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
            output_scale=1.0,
            atol=1e-2,
            rtol=1e-2,
        )

    return solve(jnp.stack((u0, u0 + 0.1, u0 + 0.2)))


def test_batched_getitem_possible(approximate_solution_batched):
    solution_type = type(approximate_solution_batched)
    for idx in (0, 1, 2):
        approximate_solution = approximate_solution_batched[idx]
        assert isinstance(approximate_solution, solution_type)
        assert jnp.allclose(approximate_solution.t, approximate_solution_batched.t[idx])
        assert jnp.allclose(approximate_solution.u, approximate_solution_batched.u[idx])


def test_batched_iter_possible(approximate_solution_batched):
    solution_type = type(approximate_solution_batched)
    for idx, approximate_solution in enumerate(approximate_solution_batched):
        assert isinstance(approximate_solution, solution_type)
        assert jnp.allclose(approximate_solution.t, approximate_solution_batched.t[idx])
        assert jnp.allclose(approximate_solution.u, approximate_solution_batched.u[idx])


def test_marginal_nth_derivative_of_solution(approximate_solution):
    """Assert that each $n$th derivative matches the quantity of interest's shape."""
    # Assert that the marginals have the same shape as the qoi.
    for i in (0, 1):
        derivatives = approximate_solution.marginals.marginal_nth_derivative(i)
        assert derivatives.mean.shape == approximate_solution.u.shape

    # if the requested derivative is not in the state-space model, raise a ValueError
    with testing.raises(ValueError):
        approximate_solution.marginals.marginal_nth_derivative(100)
