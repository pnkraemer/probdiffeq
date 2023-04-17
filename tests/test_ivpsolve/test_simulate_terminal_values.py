"""Tests for solving IVPs for the terminal value."""

import functools
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.test_util

from probdiffeq import controls, ivpsolve, ivpsolvers, taylor, test_util
from probdiffeq.backend import testing
from probdiffeq.statespace import recipes
from probdiffeq.strategies import filters, smoothers

# Generate interesting test cases


class _SimulateTerminalValuesConfig(NamedTuple):
    ivp: Any
    solver_fn: Any
    impl_fn: Any
    strategy_fn: Any
    solver_config: Any
    loop_fn: Any
    control: Any
    output_scale: Any


_ALL_STRATEGY_FNS = [filters.Filter, smoothers.Smoother, smoothers.FixedPointSmoother]


@testing.case(tags=["jvp"])
@testing.parametrize_with_cases("ivp", cases="..problem_cases", has_tag="nd")
@testing.parametrize_with_cases("impl_fn", cases="..statespace_cases", has_tag="nd")
@testing.parametrize("strategy_fn", _ALL_STRATEGY_FNS)
def case_setup_all_strategy_statespace_combinations_nd(
    ivp, impl_fn, strategy_fn, solver_config
):
    return _SimulateTerminalValuesConfig(
        ivp=ivp,
        solver_fn=ivpsolvers.MLESolver,
        impl_fn=impl_fn,
        strategy_fn=strategy_fn,
        solver_config=solver_config,
        loop_fn=jax.lax.while_loop,
        control=controls.ProportionalIntegral(),
        output_scale=1.0,
    )


@testing.case
@testing.parametrize_with_cases("ivp", cases="..problem_cases", has_tag="scalar")
@testing.parametrize_with_cases("impl_fn", cases="..statespace_cases", has_tag="scalar")
@testing.parametrize("strategy_fn", _ALL_STRATEGY_FNS)
def case_setup_all_strategy_statespace_combinations_scalar(
    ivp, impl_fn, strategy_fn, solver_config
):
    return _SimulateTerminalValuesConfig(
        ivp=ivp,
        solver_fn=ivpsolvers.MLESolver,
        impl_fn=impl_fn,
        strategy_fn=strategy_fn,
        solver_config=solver_config,
        loop_fn=jax.lax.while_loop,
        control=controls.ProportionalIntegral(),
        output_scale=1.0,
    )


@testing.case(tags=["jvp"])
@testing.parametrize_with_cases("ivp", cases="..problem_cases", has_tag="nd")
@testing.parametrize_with_cases("solver_fn", cases="..ivpsolver_cases")
@testing.parametrize("strategy_fn", _ALL_STRATEGY_FNS)
def case_setup_all_ivpsolvers_strategy_combinations(
    ivp, solver_fn, strategy_fn, solver_config
):
    return _SimulateTerminalValuesConfig(
        ivp=ivp,
        solver_fn=solver_fn,
        impl_fn=recipes.ts0_blockdiag,
        strategy_fn=strategy_fn,
        solver_config=solver_config,
        loop_fn=jax.lax.while_loop,
        control=controls.ProportionalIntegral(),
        output_scale=1.0,
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


@testing.case(tags=["jvp"])
@testing.parametrize_with_cases("ivp", cases="..problem_cases", has_tag="nd")
@testing.parametrize_with_cases("loop_fn", cases=".", prefix="case_loop_")
def case_setup_all_loops(ivp, loop_fn, solver_config):
    return _SimulateTerminalValuesConfig(
        ivp=ivp,
        solver_fn=ivpsolvers.MLESolver,
        impl_fn=recipes.ts0_blockdiag,
        strategy_fn=filters.Filter,
        solver_config=solver_config,
        loop_fn=loop_fn,
        control=controls.ProportionalIntegral(),
        output_scale=1.0,
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
@testing.parametrize_with_cases("ivp", cases="..problem_cases", has_tag="nd")
@testing.parametrize_with_cases("control", cases=".", prefix="case_control_")
def case_setup_all_controls(ivp, control, solver_config):
    return _SimulateTerminalValuesConfig(
        ivp=ivp,
        solver_fn=ivpsolvers.MLESolver,
        impl_fn=recipes.ts0_blockdiag,
        strategy_fn=filters.Filter,
        solver_config=solver_config,
        loop_fn=jax.lax.while_loop,
        control=control,
        output_scale=1.0,
    )


# Compute the IVP solution for given setups


@testing.fixture(name="solution_terminal_values")
@testing.parametrize_with_cases("setup", cases=".", prefix="case_setup_")
def fixture_solution_terminal_values(setup):
    ode_shape = setup.ivp.initial_values[0].shape
    solver = test_util.generate_solver(
        solver_factory=setup.solver_fn,
        strategy_factory=setup.strategy_fn,
        impl_factory=setup.impl_fn,
        ode_shape=ode_shape,
        num_derivatives=4,
    )

    # todo: move to solver config?
    #  (But this would involve knowing the IVP at solver-config-creation time,
    #  which would be a non-trivial change.)
    ode = setup.ivp.vector_field
    u0s = setup.ivp.initial_values
    t0 = setup.ivp.t0
    parameters = setup.ivp.args
    dt0 = ivpsolve.propose_dt0(ode, u0s, t0=t0, parameters=parameters)

    solution = ivpsolve.simulate_terminal_values(
        ode,
        u0s,
        t0=t0,
        t1=setup.ivp.t1,
        parameters=parameters,
        output_scale=setup.output_scale,
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
    sol_ref = (setup.ivp.t1, setup.ivp.solution(setup.ivp.t1))
    return sol, sol_ref


# Actual tests


def test_terminal_values_correct(solution_terminal_values, solver_config):
    (t, u), (t_ref, u_ref) = solution_terminal_values

    atol = solver_config.atol_assert
    rtol = solver_config.rtol_assert
    assert jnp.allclose(t, t_ref, atol=atol, rtol=rtol)
    assert jnp.allclose(u, u_ref, atol=atol, rtol=rtol)


@testing.parametrize_with_cases("ivp", cases="..problem_cases", has_tag="nd")
def test_jvp(ivp, solver_config):
    ode_shape = ivp.initial_values[0].shape
    solver = test_util.generate_solver(
        solver_factory=ivpsolvers.CalibrationFreeSolver,
        strategy_factory=filters.Filter,
        impl_factory=recipes.ts0_blockdiag,
        ode_shape=ode_shape,
        num_derivatives=1,
    )

    fn = functools.partial(
        _init_to_terminal_value,
        ivp=ivp,
        solver=solver,
        solver_config=solver_config,
    )
    u0 = ivp.initial_values[0]
    jvp = functools.partial(jax.jvp, fn)

    # Autodiff tests are sometimes a bit flaky...
    # There is also no clear mathematical argument that solutions that
    # have been computed with atol A and rtol R should have
    # gradients that coincide with their finite difference approximations with
    # accuracies A and R. Therefore, we relax them a little bit.
    # todo: move autodiff_atol_assert, autodiff_rtol_assert into solver config.
    atol, rtol = 10 * solver_config.atol_assert, 10 * solver_config.rtol_assert
    jax.test_util.check_jvp(f=fn, f_jvp=jvp, args=(u0,), atol=atol, rtol=rtol)


def _init_to_terminal_value(init, ivp, solver, solver_config):
    solution = ivpsolve.simulate_terminal_values(
        ivp.vector_field,
        (init,),
        t0=ivp.t0,
        t1=ivp.t1,
        parameters=ivp.args,
        solver=solver,
        atol=solver_config.atol_solve,
        rtol=solver_config.rtol_solve,
    )
    return solution.u.T @ solution.u
