"""Tests for solving IVPs for the terminal value."""

import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.test_util
import pytest_cases

from probdiffeq import solution_routines, solvers, taylor, test_util
from probdiffeq.implementations import recipes
from probdiffeq.strategies import filters, smoothers


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


@pytest_cases.fixture(scope="session", name="solution_terminal_values")
@pytest_cases.parametrize_with_cases("ode_problem", cases=".problem_cases")
@pytest_cases.parametrize_with_cases("impl_fn", cases=".impl_cases")
@pytest_cases.parametrize_with_cases("solver_fn", cases=".solver_cases")
@pytest_cases.parametrize_with_cases("loop_fn", cases=".", prefix="case_loop_")
@pytest_cases.parametrize(
    "strat_fn", [filters.Filter, smoothers.Smoother, smoothers.FixedPointSmoother]
)
def fixture_solution_terminal_values(
    ode_problem, solver_fn, impl_fn, strat_fn, solver_config, loop_fn
):
    ode_shape = ode_problem.initial_values[0].shape
    solver = test_util.generate_solver(
        solver_factory=solver_fn,
        strategy_factory=strat_fn,
        impl_factory=impl_fn,
        ode_shape=ode_shape,
        num_derivatives=4,
    )
    solution = solution_routines.simulate_terminal_values(
        ode_problem.vector_field,
        ode_problem.initial_values,
        t0=ode_problem.t0,
        t1=ode_problem.t1,
        parameters=ode_problem.args,
        solver=solver,
        atol=solver_config.atol_solve,
        rtol=solver_config.rtol_solve,
        taylor_fn=taylor.taylor_mode_fn,
        while_loop_fn=loop_fn,
    )
    return (solution.t, solution.u), (
        ode_problem.t1,
        ode_problem.solution(ode_problem.t1),
    )


def test_terminal_values_correct(solution_terminal_values, solver_config):
    (t, u), (t_ref, u_ref) = solution_terminal_values

    atol = solver_config.atol_assert
    rtol = solver_config.rtol_assert
    assert jnp.allclose(t, t_ref, atol=atol, rtol=rtol)
    assert jnp.allclose(u, u_ref, atol=atol, rtol=rtol)


@pytest_cases.parametrize_with_cases("ode_problem", cases=".problem_cases")
@pytest_cases.parametrize("impl_fn", [recipes.BlockDiagTS0.from_params])
@pytest_cases.parametrize("solver_fn", [solvers.MLESolver])
@pytest_cases.parametrize("strat_fn", [filters.Filter])
def test_jvp(ode_problem, solver_fn, impl_fn, strat_fn, solver_config, loop_fn):
    ode_shape = ode_problem.initial_values[0].shape
    solver = test_util.generate_solver(
        solver_factory=solver_fn,
        strategy_factory=strat_fn,
        impl_factory=impl_fn,
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
