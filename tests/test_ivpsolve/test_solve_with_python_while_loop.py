"""Tests for solving IVPs on adaptive grids."""
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from probdiffeq import ivpsolve, ivpsolvers, test_util
from probdiffeq.backend import testing
from probdiffeq.ssm import recipes
from probdiffeq.strategies import filters, smoothers


class _SolveWithPythonWhileLoopConfig(NamedTuple):
    ode_problem: Any
    impl_fn: Any
    solver_fn: Any
    strat_fn: Any
    solver_config: Any
    output_scale: Any


@testing.case
@testing.parametrize_with_cases("ode_problem", cases="..problem_cases", has_tag=["nd"])
@testing.parametrize_with_cases("impl_fn", cases="..impl_cases", has_tag=["nd"])
def case_setup_all_nd_configs(ode_problem, impl_fn, solver_config):
    return _SolveWithPythonWhileLoopConfig(
        ode_problem=ode_problem,
        impl_fn=impl_fn,
        solver_fn=ivpsolvers.MLESolver,
        strat_fn=filters.Filter,
        solver_config=solver_config,
        output_scale=1.0,
    )


@testing.case
@testing.parametrize_with_cases(
    "ode_problem", cases="..problem_cases", has_tag=["scalar"]
)
@testing.parametrize_with_cases("impl_fn", cases="..impl_cases", has_tag=["scalar"])
def case_setup_all_scalar_configs(ode_problem, impl_fn, solver_config):
    return _SolveWithPythonWhileLoopConfig(
        ode_problem=ode_problem,
        impl_fn=impl_fn,
        solver_fn=ivpsolvers.MLESolver,
        strat_fn=filters.Filter,
        solver_config=solver_config,
        output_scale=1.0,
    )


@testing.case
@testing.parametrize_with_cases("ode_problem", cases="..problem_cases", has_tag="nd")
@testing.parametrize("strat_fn", [filters.Filter, smoothers.Smoother])
def case_setup_all_strategies(ode_problem, strat_fn, solver_config):
    return _SolveWithPythonWhileLoopConfig(
        ode_problem=ode_problem,
        impl_fn=recipes.ts0_blockdiag,
        solver_fn=ivpsolvers.MLESolver,
        strat_fn=strat_fn,
        solver_config=solver_config,
        output_scale=1.0,
    )


@testing.case
@testing.parametrize_with_cases("ode_problem", cases="..problem_cases", has_tag="nd")
@testing.parametrize_with_cases("solver_fn", cases="..ivpsolver_cases")
def case_setup_all_ivpsolvers(ode_problem, solver_fn, solver_config):
    return _SolveWithPythonWhileLoopConfig(
        ode_problem=ode_problem,
        solver_fn=solver_fn,
        impl_fn=recipes.ts0_blockdiag,
        strat_fn=filters.Filter,
        solver_config=solver_config,
        output_scale=1.0,
    )


@testing.fixture(name="solution_solve")
@testing.parametrize_with_cases(
    "setup", cases=".", prefix="case_setup_", scope="session"
)
def fixture_solution_solve_with_python_while_loop(setup):
    solver = test_util.generate_solver(
        solver_factory=setup.solver_fn,
        strategy_factory=setup.strat_fn,
        impl_factory=setup.impl_fn,
        ode_shape=setup.ode_problem.initial_values[0].shape,
        num_derivatives=4,
    )
    solution = ivpsolve.solve_with_python_while_loop(
        setup.ode_problem.vector_field,
        setup.ode_problem.initial_values,
        t0=setup.ode_problem.t0,
        t1=setup.ode_problem.t1,
        parameters=setup.ode_problem.args,
        output_scale=setup.output_scale,
        solver=solver,
        atol=setup.solver_config.atol_solve,
        rtol=setup.solver_config.rtol_solve,
    )

    return solution.u, jax.vmap(setup.ode_problem.solution)(solution.t)


def test_solve_computes_correct_terminal_value(solution_solve, solver_config):
    u, u_ref = solution_solve
    atol = solver_config.atol_assert
    rtol = solver_config.rtol_assert
    assert jnp.allclose(u, u_ref, atol=atol, rtol=rtol)
