"""Tests for solving IVPs for checkpoints."""
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from probdiffeq import ivpsolve, ivpsolvers, taylor, test_util
from probdiffeq.backend import testing
from probdiffeq.statespace import recipes
from probdiffeq.strategies import filters, smoothers

# Generate interesting test cases


class _SolveAndSaveAtConfig(NamedTuple):
    ivp: Any
    solver_fn: Any
    impl_fn: Any
    strategy_fn: Any
    loop_fn: Any
    solver_config: Any
    output_scale: Any


@testing.case
@testing.parametrize_with_cases("ivp", cases="..problem_cases", has_tag=["nd"])
@testing.parametrize_with_cases("impl_fn", cases="..statespace_cases", has_tag=["nd"])
@testing.parametrize("strategy_fn", [filters.filter, smoothers.smoother_fixedpoint])
def case_setup_all_strategy_statespace_combinations_nd(
    ivp, impl_fn, strategy_fn, solver_config
):
    return _SolveAndSaveAtConfig(
        ivp=ivp,
        solver_fn=ivpsolvers.solver_mle,
        impl_fn=impl_fn,
        strategy_fn=strategy_fn,
        solver_config=solver_config,
        loop_fn=jax.lax.while_loop,
        output_scale=1.0,
    )


@testing.case
@testing.parametrize_with_cases("ivp", cases="..problem_cases", has_tag=["scalar"])
@testing.parametrize_with_cases(
    "impl_fn", cases="..statespace_cases", has_tag=["scalar"]
)
@testing.parametrize("strategy_fn", [filters.filter, smoothers.smoother_fixedpoint])
def case_setup_all_stratgy_statespace_combinations_scalar(
    ivp, impl_fn, strategy_fn, solver_config
):
    return _SolveAndSaveAtConfig(
        ivp=ivp,
        solver_fn=ivpsolvers.solver_mle,
        impl_fn=impl_fn,
        strategy_fn=strategy_fn,
        solver_config=solver_config,
        loop_fn=jax.lax.while_loop,
        output_scale=1.0,
    )


@testing.case
@testing.parametrize_with_cases("ivp", cases="..problem_cases", has_tag=["nd"])
@testing.parametrize_with_cases("solver_fn", cases="..ivpsolver_cases")
@testing.parametrize("strategy_fn", [filters.filter, smoothers.smoother_fixedpoint])
def case_setup_all_solvers_strategies(ivp, solver_fn, strategy_fn, solver_config):
    return _SolveAndSaveAtConfig(
        ivp=ivp,
        solver_fn=solver_fn,
        impl_fn=recipes.ts0_blockdiag,
        strategy_fn=strategy_fn,
        solver_config=solver_config,
        loop_fn=jax.lax.while_loop,
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


@testing.case
@testing.parametrize_with_cases("ivp", cases="..problem_cases", has_tag=["nd"])
@testing.parametrize_with_cases("loop_fn", cases=".", prefix="case_loop_")
def case_setup_all_loops(ivp, loop_fn, solver_config):
    return _SolveAndSaveAtConfig(
        ivp=ivp,
        solver_fn=ivpsolvers.solver_mle,
        impl_fn=recipes.ts0_blockdiag,
        strategy_fn=filters.filter,
        solver_config=solver_config,
        loop_fn=loop_fn,
        output_scale=1.0,
    )


@testing.fixture(name="solution_save_at")
@testing.parametrize_with_cases("setup", cases=".", prefix="case_setup_")
def fixture_solution_save_at(setup):
    ode_shape = setup.ivp.initial_values[0].shape
    solver = test_util.generate_solver(
        solver_factory=setup.solver_fn,
        strategy_factory=setup.strategy_fn,
        impl_factory=setup.impl_fn,
        ode_shape=ode_shape,
        num_derivatives=4,
    )

    t0, t1 = setup.ivp.t0, setup.ivp.t1
    save_at = setup.solver_config.grid_for_save_at_fn(t0, t1)

    # todo: move to solver config?
    #  (But this would involve knowing the IVP at solver-config-creation time,
    #  which would be a non-trivial change.)
    ode = setup.ivp.vector_field
    u0s = setup.ivp.initial_values
    t0 = setup.ivp.t0
    parameters = setup.ivp.args
    dt0 = ivpsolve.propose_dt0(ode, u0s, t0=t0, parameters=parameters)

    solution = ivpsolve.solve_and_save_at(
        setup.ivp.vector_field,
        setup.ivp.initial_values,
        save_at=save_at,
        parameters=setup.ivp.args,
        solver=solver,
        dt0=dt0,
        output_scale=setup.output_scale,
        atol=setup.solver_config.atol_solve,
        rtol=setup.solver_config.rtol_solve,
        taylor_fn=taylor.taylor_mode_fn,
        while_loop_fn_temporal=setup.loop_fn,
        while_loop_fn_per_step=setup.loop_fn,
    )
    return solution.u, jax.vmap(setup.ivp.solution)(solution.t)


def test_solution_correct(solution_save_at, solver_config):
    u, u_ref = solution_save_at
    assert jnp.allclose(
        u,
        u_ref,
        atol=solver_config.atol_assert,
        rtol=solver_config.rtol_assert,
    )


@testing.parametrize_with_cases("ivp", cases="..problem_cases", has_tag=["nd"])
def test_smoother_warning(ivp):
    """A non-fixed-point smoother is not usable in save-at-simulation."""
    ts = jnp.linspace(ivp.t0, ivp.t1, num=3)
    solver = test_util.generate_solver(strategy_factory=smoothers.smoother)

    # todo: does this compute the full solve? We only want to catch a warning!
    with testing.warns():
        ivpsolve.solve_and_save_at(
            ivp.vector_field,
            ivp.initial_values,
            save_at=ts,
            parameters=ivp.args,
            solver=solver,
        )
