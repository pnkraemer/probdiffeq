"""Tests for solving IVPs on adaptive grids."""
import jax
import jax.numpy as jnp

from probdiffeq import ivpsolve, taylor, test_util
from probdiffeq.backend import testing
from probdiffeq.strategies import filters, smoothers


@testing.fixture(scope="session", name="solution_solve")
@testing.parametrize_with_cases("ode_problem", cases="..problem_cases")
@testing.parametrize_with_cases("impl_fn", cases="..impl_cases")
@testing.parametrize_with_cases("solver_fn", cases="..ivpsolver_cases")
@testing.parametrize("strat_fn", [filters.Filter, smoothers.Smoother])
def fixture_solution_solve_with_python_while_loop(
    ode_problem, solver_fn, impl_fn, strat_fn, solver_config
):
    solver = test_util.generate_solver(
        solver_factory=solver_fn,
        strategy_factory=strat_fn,
        impl_factory=impl_fn,
        ode_shape=(2,),
        num_derivatives=4,
    )
    solution = ivpsolve.solve_with_python_while_loop(
        ode_problem.vector_field,
        ode_problem.initial_values,
        t0=ode_problem.t0,
        t1=ode_problem.t1,
        parameters=ode_problem.args,
        solver=solver,
        atol=solver_config.atol_solve,
        rtol=solver_config.rtol_solve,
        taylor_fn=taylor.taylor_mode_fn,
    )

    return solution.u, jax.vmap(ode_problem.solution)(solution.t)


def test_solve_computes_correct_terminal_value(solution_solve, solver_config):
    u, u_ref = solution_solve
    assert jnp.allclose(
        u,
        u_ref,
        atol=solver_config.atol_assert,
        rtol=solver_config.rtol_assert,
    )
