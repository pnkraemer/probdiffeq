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


@pytest_cases.fixture(scope="session", name="solution_terminal_values")
@pytest_cases.parametrize_with_cases("ode_problem", cases=".problem_cases")
@pytest_cases.parametrize_with_cases("impl_fn", cases=".impl_cases")
@pytest_cases.parametrize_with_cases("solver_fn", cases=".solver_cases")
@pytest_cases.parametrize(
    "strat_fn", [filters.Filter, smoothers.Smoother, smoothers.FixedPointSmoother]
)
def fixture_solution_terminal_values(
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
def test_simulate_terminal_values_jvp(
    ode_problem, solver_fn, impl_fn, strat_fn, solver_config
):
    ode_shape = ode_problem.initial_values[0].shape
    solver = test_util.generate_solver(
        solver_factory=solver_fn,
        strategy_factory=strat_fn,
        impl_factory=impl_fn,
        ode_shape=ode_shape,
        num_derivatives=2,
    )

    @jax.jit
    def init_to_terminal_value(init):
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
        return solution.u

    fn, u0 = init_to_terminal_value, ode_problem.initial_values[0]
    jvp = functools.partial(jax.jvp, fn)
    jax.test_util.check_jvp(fn, jvp, (u0,))


@pytest_cases.parametrize_with_cases("ode_problem", cases=".problem_cases")
@pytest_cases.parametrize("impl_fn", [recipes.DenseTS1.from_params])
@pytest_cases.parametrize("solver_fn", [solvers.MLESolver])
@pytest_cases.parametrize("strat_fn", [filters.Filter])
def test_simulate_terminal_values_vjp(
    ode_problem, solver_fn, impl_fn, strat_fn, solver_config
):
    ode_shape = ode_problem.initial_values[0].shape
    solver = test_util.generate_solver(
        solver_factory=solver_fn,
        strategy_factory=strat_fn,
        impl_factory=impl_fn,
        ode_shape=ode_shape,
        num_derivatives=1,
    )

    def loop_fn(**kwargs):
        def lo(cond_fun, body_fun, init_val):
            return eqx.internal.while_loop(cond_fun, body_fun, init_val, **kwargs)

        return lo

    @jax.jit
    def init_to_terminal_value(init):
        solution = solution_routines.simulate_terminal_values(
            ode_problem.vector_field,
            (init,),
            t0=ode_problem.t0,
            t1=ode_problem.t1,
            parameters=ode_problem.args,
            solver=solver,
            atol=solver_config.atol_solve,
            rtol=solver_config.rtol_solve,
            while_loop_fn=loop_fn(kind="bounded", max_steps=100),
        )
        return solution.u.T @ solution.u

    fn, u0 = init_to_terminal_value, ode_problem.initial_values[0]
    print(jax.value_and_grad(fn)(u0))
    vjp = functools.partial(jax.vjp, fn)
    jax.test_util.check_vjp(fn, vjp, (u0,))
