"""Compare simulate_terminal_values to solve_with_python_while_loop."""

import diffeqzoo.ivps
import jax

from probdiffeq import ivpsolve, test_util
from probdiffeq.backend import testing


@testing.fixture(name="problem_args_kwargs")
def fixture_problem_args_kwargs():
    f, u0, (t0, _), f_args = diffeqzoo.ivps.lotka_volterra()
    t1 = 2.0  # Short time-intervals are sufficient for this test.

    @jax.jit
    def vf(x, *, t, p):  # pylint: disable=unused-argument
        return f(x, *p)

    return (vf, (u0,)), {"t0": t0, "t1": t1, "parameters": f_args}


@testing.fixture(name="solver_kwargs")
def fixture_solver_kwargs():
    solver = test_util.generate_solver(num_derivatives=2)
    return {"solver": solver, "output_scale": 1.0, "atol": 1e-2, "rtol": 1e-2}


@testing.fixture(name="solution_python_loop")
def fixture_solution_with_python_while_loop(problem_args_kwargs, solver_kwargs):
    args, kwargs = problem_args_kwargs
    return ivpsolve.solve_with_python_while_loop(*args, **kwargs, **solver_kwargs)


@testing.fixture(name="simulation_terminal_values")
def fixture_simulation_terminal_values(problem_args_kwargs, solver_kwargs):
    args, kwargs = problem_args_kwargs
    return ivpsolve.simulate_terminal_values(*args, **kwargs, **solver_kwargs)


def test_terminal_values_identical(solution_python_loop, simulation_terminal_values):
    """The terminal values must be identical."""
    expected = jax.tree_util.tree_map(lambda s: s[-1], solution_python_loop)
    received = simulation_terminal_values
    assert testing.tree_all_allclose(received, expected)
