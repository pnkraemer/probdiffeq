"""Tests for solving IVPs for checkpoints."""
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from probdiffeq import ivpsolve, ivpsolvers, taylor, test_util
from probdiffeq.backend import testing
from probdiffeq.implementations import recipes
from probdiffeq.strategies import filters, smoothers

# Generate interesting test cases


class _SolveAndSaveAtConfig(NamedTuple):
    ode_problem: Any
    solver_fn: Any
    impl_fn: Any
    strat_fn: Any
    loop_fn: Any
    solver_config: Any


@testing.case
@testing.parametrize_with_cases("ode_problem", cases="..problem_cases")
@testing.parametrize_with_cases("impl_fn", cases="..impl_cases")
def case_setup_all_implementations(ode_problem, impl_fn, solver_config):
    return _SolveAndSaveAtConfig(
        ode_problem=ode_problem,
        solver_fn=ivpsolvers.MLESolver,
        impl_fn=impl_fn,
        strat_fn=filters.Filter,
        solver_config=solver_config,
        loop_fn=jax.lax.while_loop,
    )


@testing.case
@testing.parametrize_with_cases("ode_problem", cases="..problem_cases")
@testing.parametrize("strat_fn", [filters.Filter, smoothers.FixedPointSmoother])
def case_setup_all_strategies(ode_problem, strat_fn, solver_config):
    return _SolveAndSaveAtConfig(
        ode_problem=ode_problem,
        solver_fn=ivpsolvers.MLESolver,
        impl_fn=recipes.BlockDiagTS0.from_params,
        strat_fn=strat_fn,
        solver_config=solver_config,
        loop_fn=jax.lax.while_loop,
    )


@testing.case
@testing.parametrize_with_cases("ode_problem", cases="..problem_cases")
@testing.parametrize_with_cases("solver_fn", cases="..ivpsolver_cases")
def case_setup_all_ivpsolvers(ode_problem, solver_fn, solver_config):
    return _SolveAndSaveAtConfig(
        ode_problem=ode_problem,
        solver_fn=solver_fn,
        impl_fn=recipes.BlockDiagTS0.from_params,
        strat_fn=filters.Filter,
        solver_config=solver_config,
        loop_fn=jax.lax.while_loop,
    )


@testing.case(id="jax.lax.while_loop")
def case_loop_lax():
    return jax.lax.while_loop


@testing.case(id="eqx.bounded_while_loop")
def case_loop_eqx():
    def lo(cond_fun, body_fun, init_val):
        return eqx.internal.while_loop(
            cond_fun, body_fun, init_val, kind="bounded", max_steps=50
        )

    return lo


@testing.case
@testing.parametrize_with_cases("ode_problem", cases="..problem_cases")
@testing.parametrize_with_cases("loop_fn", cases="..", prefix="case_loop_")
def case_setup_all_loops(ode_problem, loop_fn, solver_config):
    return _SolveAndSaveAtConfig(
        ode_problem=ode_problem,
        solver_fn=ivpsolvers.MLESolver,
        impl_fn=recipes.BlockDiagTS0.from_params,
        strat_fn=filters.Filter,
        solver_config=solver_config,
        loop_fn=loop_fn,
    )


@testing.fixture(scope="session", name="solution_save_at")
@testing.parametrize_with_cases(
    "setup", cases="..", prefix="case_setup_", scope="session"
)
def fixture_solution_save_at(setup):
    ode_shape = setup.ode_problem.initial_values[0].shape
    solver = test_util.generate_solver(
        solver_factory=setup.solver_fn,
        strategy_factory=setup.strat_fn,
        impl_factory=setup.impl_fn,
        ode_shape=ode_shape,
        num_derivatives=4,
    )

    t0, t1 = setup.ode_problem.t0, setup.ode_problem.t1
    save_at = setup.solver_config.grid_for_save_at_fn(t0, t1)

    solution = ivpsolve.solve_and_save_at(
        setup.ode_problem.vector_field,
        setup.ode_problem.initial_values,
        save_at=save_at,
        parameters=setup.ode_problem.args,
        solver=solver,
        atol=setup.solver_config.atol_solve,
        rtol=setup.solver_config.rtol_solve,
        taylor_fn=taylor.taylor_mode_fn,
        while_loop_fn_temporal=setup.loop_fn,
        while_loop_fn_per_step=setup.loop_fn,
    )
    return solution.u, jax.vmap(setup.ode_problem.solution)(solution.t)


def test_solution_correct(solution_save_at, solver_config):
    u, u_ref = solution_save_at
    assert jnp.allclose(
        u,
        u_ref,
        atol=solver_config.atol_assert,
        rtol=solver_config.rtol_assert,
    )


@testing.parametrize_with_cases("ode_problem", cases="..problem_cases")
def test_smoother_warning(ode_problem):
    """A non-fixed-point smoother is not usable in save-at-simulation."""
    ts = jnp.linspace(ode_problem.t0, ode_problem.t1, num=3)
    solver = test_util.generate_solver(strategy_factory=smoothers.Smoother)

    # todo: does this compute the full solve? We only want to catch a warning!
    with testing.warns():
        ivpsolve.solve_and_save_at(
            ode_problem.vector_field,
            ode_problem.initial_values,
            save_at=ts,
            parameters=ode_problem.args,
            solver=solver,
        )
