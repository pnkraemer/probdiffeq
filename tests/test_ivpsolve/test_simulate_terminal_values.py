"""Tests for solving IVPs for the terminal value."""

import functools
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.test_util

from probdiffeq import controls, ivpsolve, ivpsolvers, taylor, test_util
from probdiffeq.backend import testing
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
    control: Any


@testing.case
@testing.parametrize_with_cases("ode_problem", cases="..problem_cases", has_tag="nd")
@testing.parametrize_with_cases("impl_fn", cases="..impl_cases", has_tag="nd")
def case_setup_all_implementations_nd(ode_problem, impl_fn, solver_config):
    return _SimulateTerminalValuesConfig(
        ode_problem=ode_problem,
        solver_fn=ivpsolvers.MLESolver,
        impl_fn=impl_fn,
        strat_fn=filters.Filter,
        solver_config=solver_config,
        loop_fn=jax.lax.while_loop,
        control=controls.ProportionalIntegral(),
    )


@testing.case
@testing.parametrize_with_cases(
    "ode_problem", cases="..problem_cases", has_tag="scalar"
)
@testing.parametrize_with_cases("impl_fn", cases="..impl_cases", has_tag="scalar")
def case_setup_all_implementations_scalar(ode_problem, impl_fn, solver_config):
    return _SimulateTerminalValuesConfig(
        ode_problem=ode_problem,
        solver_fn=ivpsolvers.MLESolver,
        impl_fn=impl_fn,
        strat_fn=filters.Filter,
        solver_config=solver_config,
        loop_fn=jax.lax.while_loop,
        control=controls.ProportionalIntegral(),
    )


@testing.case
@testing.parametrize_with_cases("ode_problem", cases="..problem_cases", has_tag="nd")
@testing.parametrize(
    "strat_fn", [filters.Filter, smoothers.Smoother, smoothers.FixedPointSmoother]
)
def case_setup_all_strategies(ode_problem, strat_fn, solver_config):
    return _SimulateTerminalValuesConfig(
        ode_problem=ode_problem,
        solver_fn=ivpsolvers.MLESolver,
        impl_fn=recipes.ts0_blockdiag,
        strat_fn=strat_fn,
        solver_config=solver_config,
        loop_fn=jax.lax.while_loop,
        control=controls.ProportionalIntegral(),
    )


@testing.case
@testing.parametrize_with_cases("ode_problem", cases="..problem_cases", has_tag="nd")
@testing.parametrize_with_cases("solver_fn", cases="..ivpsolver_cases")
def case_setup_all_ivpsolvers(ode_problem, solver_fn, solver_config):
    return _SimulateTerminalValuesConfig(
        ode_problem=ode_problem,
        solver_fn=solver_fn,
        impl_fn=recipes.ts0_blockdiag,
        strat_fn=filters.Filter,
        solver_config=solver_config,
        loop_fn=jax.lax.while_loop,
        control=controls.ProportionalIntegral(),
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
@testing.parametrize_with_cases("ode_problem", cases="..problem_cases", has_tag="nd")
@testing.parametrize_with_cases("loop_fn", cases=".", prefix="case_loop_")
def case_setup_all_loops(ode_problem, loop_fn, solver_config):
    return _SimulateTerminalValuesConfig(
        ode_problem=ode_problem,
        solver_fn=ivpsolvers.MLESolver,
        impl_fn=recipes.ts0_blockdiag,
        strat_fn=filters.Filter,
        solver_config=solver_config,
        loop_fn=loop_fn,
        control=controls.ProportionalIntegral(),
    )


@testing.case
def case_control_pi():
    return controls.ProportionalIntegral()


@testing.case
def case_control_pi_clipped():
    return controls.ProportionalIntegralClipped()


@testing.case
def case_control_i():
    return controls.Integral()


@testing.case
def case_control_i_clipped():
    return controls.IntegralClipped()


# todo: test for all_taylor_fns as well


@testing.case
@testing.parametrize_with_cases("ode_problem", cases="..problem_cases", has_tag="nd")
@testing.parametrize_with_cases("control", cases=".", prefix="case_control_")
def case_setup_all_controls(ode_problem, control, solver_config):
    return _SimulateTerminalValuesConfig(
        ode_problem=ode_problem,
        solver_fn=ivpsolvers.MLESolver,
        impl_fn=recipes.ts0_blockdiag,
        strat_fn=filters.Filter,
        solver_config=solver_config,
        loop_fn=jax.lax.while_loop,
        control=control,
    )


# Compute the IVP solution for given setups


@testing.fixture(name="solution_terminal_values")
@testing.parametrize_with_cases(
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

    # todo: move to solver config?
    #  (But this would involve knowing the IVP at solver-config-creation time,
    #  which would be a non-trivial change.)
    ode = setup.ode_problem.vector_field
    u0s = setup.ode_problem.initial_values
    t0 = setup.ode_problem.t0
    parameters = setup.ode_problem.args
    dt0 = ivpsolve.propose_dt0(ode, u0s, t0=t0, parameters=parameters)

    solution = ivpsolve.simulate_terminal_values(
        ode,
        u0s,
        t0=t0,
        t1=setup.ode_problem.t1,
        parameters=parameters,
        dt0=dt0,
        solver=solver,
        atol=setup.solver_config.atol_solve,
        rtol=setup.solver_config.rtol_solve,
        taylor_fn=taylor.taylor_mode_fn,
        while_loop_fn_temporal=setup.loop_fn,
        while_loop_fn_per_step=setup.loop_fn,
        control=setup.control,
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


@testing.parametrize_with_cases("ode_problem", cases="..problem_cases", has_tag="nd")
def test_jvp(ode_problem, solver_config):
    ode_shape = ode_problem.initial_values[0].shape
    solver = test_util.generate_solver(
        solver_factory=ivpsolvers.MLESolver,
        strategy_factory=filters.Filter,
        impl_factory=recipes.ts0_blockdiag,
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
    solution = ivpsolve.simulate_terminal_values(
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
