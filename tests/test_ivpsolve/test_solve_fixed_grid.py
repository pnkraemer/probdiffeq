"""Tests for solving IVPs on fixed grids."""
import functools

import jax
import jax.numpy as jnp
import jax.test_util

from probdiffeq import ivpsolve, test_util
from probdiffeq.backend import testing
from probdiffeq.implementations import recipes
from probdiffeq.strategies import filters, smoothers


@testing.fixture(scope="session", name="solution_fixed_grid")
@testing.parametrize_with_cases("ode_problem", cases="..problem_cases")
@testing.parametrize_with_cases("impl_fn", cases="..impl_cases")
@testing.parametrize_with_cases("solver_fn", cases="..ivpsolver_cases")
@testing.parametrize("strat_fn", [filters.Filter, smoothers.Smoother])
def fixture_solution_fixed_grid(
    ode_problem, solver_fn, impl_fn, strat_fn, solver_config
):
    ode_shape = ode_problem.initial_values[0].shape
    solver = test_util.generate_solver(
        solver_factory=solver_fn,
        strategy_factory=strat_fn,
        impl_factory=impl_fn,
        ode_shape=ode_shape,
        num_derivatives=4,
    )

    t0, t1 = ode_problem.t0, ode_problem.t1
    grid = solver_config.grid_for_fixed_grid_fn(t0, t1)

    solution = ivpsolve.solve_fixed_grid(
        ode_problem.vector_field,
        ode_problem.initial_values,
        grid=grid,
        parameters=ode_problem.args,
        solver=solver,
    )
    return (solution.t, solution.u), (grid, jax.vmap(ode_problem.solution)(grid))


def test_terminal_values_correct(solution_fixed_grid, solver_config):
    (t, u), (t_ref, u_ref) = solution_fixed_grid
    atol, rtol = solver_config.atol_assert, solver_config.rtol_assert
    assert jnp.allclose(t[-1], t_ref[-1], atol=atol, rtol=rtol)
    assert jnp.allclose(u[-1], u_ref[-1], atol=atol, rtol=rtol)


# todo: all solver implementations
@testing.fixture(scope="session", name="parameter_to_solution")
@testing.parametrize_with_cases("ode_problem", cases="..problem_cases")
@testing.parametrize("impl_fn", [recipes.BlockDiagTS0.from_params])
@testing.parametrize_with_cases("solver_fn", cases="..ivpsolver_cases")
@testing.parametrize("strat_fn", [filters.Filter, smoothers.Smoother])
def fixture_parameter_to_solution(
    ode_problem, solver_config, impl_fn, solver_fn, strat_fn
):
    """Parameter-to-solution map. To be differentiated."""

    def fn(u0):
        ode_shape = ode_problem.initial_values[0].shape
        solver = test_util.generate_solver(
            solver_factory=solver_fn,
            strategy_factory=strat_fn,
            impl_factory=impl_fn,
            ode_shape=ode_shape,
            num_derivatives=1,  # Low order traces more quickly
        )

        t0, t1 = ode_problem.t0, ode_problem.t1
        grid = solver_config.grid_for_fixed_grid_fn(t0, t1)

        solution = ivpsolve.solve_fixed_grid(
            ode_problem.vector_field,
            u0,
            grid=grid,
            parameters=ode_problem.args,
            solver=solver,
        )
        return solution.u

    return fn, ode_problem.initial_values


def test_jvp(parameter_to_solution):
    fn, primals = parameter_to_solution
    jvp = functools.partial(jax.jvp, fn)
    jax.test_util.check_jvp(fn, jvp, (primals,))


def test_vjp(parameter_to_solution):
    fn, primals = parameter_to_solution
    vjp = functools.partial(jax.vjp, fn)
    jax.test_util.check_vjp(fn, vjp, (primals,))
