"""Tests for solving IVPs for the terminal value."""

import functools
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.test_util
import pytest_cases

from probdiffeq import solution_routines, solvers, taylor, test_util
from probdiffeq.implementations import recipes
from probdiffeq.strategies import filters, smoothers

# Generate interesting test cases


class _SimulateTerminalValuesConfig(NamedTuple):
    ode_problem: Any
    solver_fn: Any
    impl_fn: Any
    strat_fn: Any
    solver_config: Any
    loop_fn: Any


@pytest_cases.case
@pytest_cases.parametrize_with_cases("ode_problem", cases=".problem_cases")
@pytest_cases.parametrize_with_cases("impl_fn", cases=".impl_cases")
def case_setup_all_implementations(ode_problem, impl_fn, solver_config):
    return _SimulateTerminalValuesConfig(
        ode_problem=ode_problem,
        solver_fn=solvers.MLESolver,
        impl_fn=impl_fn,
        strat_fn=filters.Filter,
        solver_config=solver_config,
        loop_fn=jax.lax.while_loop,
    )


@pytest_cases.case
@pytest_cases.parametrize_with_cases("ode_problem", cases=".problem_cases")
@pytest_cases.parametrize(
    "strat_fn", [filters.Filter, smoothers.Smoother, smoothers.FixedPointSmoother]
)
def case_setup_all_strategies(ode_problem, strat_fn, solver_config):
    return _SimulateTerminalValuesConfig(
        ode_problem=ode_problem,
        solver_fn=solvers.MLESolver,
        impl_fn=recipes.BlockDiagTS0.from_params,
        strat_fn=strat_fn,
        solver_config=solver_config,
        loop_fn=jax.lax.while_loop,
    )


@pytest_cases.case
@pytest_cases.parametrize_with_cases("ode_problem", cases=".problem_cases")
@pytest_cases.parametrize_with_cases("solver_fn", cases=".solver_cases")
def case_setup_all_solvers(ode_problem, solver_fn, solver_config):
    return _SimulateTerminalValuesConfig(
        ode_problem=ode_problem,
        solver_fn=solver_fn,
        impl_fn=recipes.BlockDiagTS0.from_params,
        strat_fn=filters.Filter,
        solver_config=solver_config,
        loop_fn=jax.lax.while_loop,
    )


@pytest_cases.case(id="jax.lax.while_loop")
def case_loop_lax():
    return jax.lax.while_loop


@pytest_cases.case(id="eqx.bounded_while_loop")
def case_loop_eqx():
    def lo(cond_fun, body_fun, init_val):
        return eqx.internal.while_loop(
            cond_fun, body_fun, init_val, kind="bounded", max_steps=50
        )

    return lo


@pytest_cases.case
@pytest_cases.parametrize_with_cases("ode_problem", cases=".problem_cases")
@pytest_cases.parametrize_with_cases("loop_fn", cases=".", prefix="case_loop_")
def case_setup_all_loops(ode_problem, loop_fn, solver_config):
    return _SimulateTerminalValuesConfig(
        ode_problem=ode_problem,
        solver_fn=solvers.MLESolver,
        impl_fn=recipes.BlockDiagTS0.from_params,
        strat_fn=filters.Filter,
        solver_config=solver_config,
        loop_fn=loop_fn,
    )


# Compute the IVP solution for given setups


@pytest_cases.fixture(scope="session", name="solution_terminal_values")
@pytest_cases.parametrize_with_cases(
    "setup", cases=".", prefix="case_setup_", scope="session"
)
def fixture_solution_terminal_values(setup):
    ode_shape = setup.ode_problem.initial_values[0].shape
    solver = test_util.generate_solver(
        solver_factory=setup.solver_fn,
        strategy_factory=setup.strat_fn,
        impl_factory=setup.impl_fn,
        ode_shape=ode_shape,
        num_derivatives=4,
    )
    solution = solution_routines.simulate_terminal_values(
        setup.ode_problem.vector_field,
        setup.ode_problem.initial_values,
        t0=setup.ode_problem.t0,
        t1=setup.ode_problem.t1,
        parameters=setup.ode_problem.args,
        solver=solver,
        atol=setup.solver_config.atol_solve,
        rtol=setup.solver_config.rtol_solve,
        taylor_fn=taylor.taylor_mode_fn,
        while_loop_fn_temporal=setup.loop_fn,
        while_loop_fn_per_step=setup.loop_fn,
    )

    sol = (solution.t, solution.u)
    sol_ref = (setup.ode_problem.t1, setup.ode_problem.solution(setup.ode_problem.t1))
    return sol, sol_ref


# Actual tests


def test_terminal_values_correct(solution_terminal_values, solver_config):
    (t, u), (t_ref, u_ref) = solution_terminal_values

    atol = solver_config.atol_assert
    rtol = solver_config.rtol_assert
    assert jnp.allclose(t, t_ref, atol=atol, rtol=rtol)
    assert jnp.allclose(u, u_ref, atol=atol, rtol=rtol)


@pytest_cases.parametrize_with_cases("ode_problem", cases=".problem_cases")
def test_jvp(ode_problem, solver_config):
    ode_shape = ode_problem.initial_values[0].shape
    solver = test_util.generate_solver(
        solver_factory=solvers.MLESolver,
        strategy_factory=filters.Filter,
        impl_factory=recipes.BlockDiagTS0.from_params,
        ode_shape=ode_shape,
        num_derivatives=2,
    )

    fn = functools.partial(
        _init_to_terminal_value,
        ode_problem=ode_problem,
        solver=solver,
        solver_config=solver_config,
    )
    u0 = ode_problem.initial_values[0]

    jvp = functools.partial(jax.jvp, fn)
    jax.test_util.check_jvp(fn, jvp, (u0,))


def _init_to_terminal_value(init, ode_problem, solver, solver_config):
    solution = solution_routines.simulate_terminal_values(
        ode_problem.vector_field,
        (init,),
        t0=ode_problem.t0,
        t1=ode_problem.t1,
        parameters=ode_problem.args,
        solver=solver,
        atol=solver_config.atol_solve,
        rtol=solver_config.rtol_solve,
    )
    return solution.u.T @ solution.u
