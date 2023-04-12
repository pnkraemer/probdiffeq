"""Tests for solving IVPs on fixed grids."""
import functools
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.test_util

from probdiffeq import ivpsolve, ivpsolvers, test_util
from probdiffeq.backend import testing
from probdiffeq.statespace import recipes
from probdiffeq.statespace.dense import corr as dense_corr
from probdiffeq.strategies import filters, smoothers


class _SolveFixedGridConfig(NamedTuple):
    ode_problem: Any
    solver_fn: Any
    impl_fn: Any
    strat_fn: Any
    solver_config: Any
    output_scale: Any


@testing.case
@testing.parametrize_with_cases("ode_problem", cases="..problem_cases", has_tag="nd")
@testing.parametrize_with_cases("impl_fn", cases="..statespace_cases", has_tag="nd")
def case_setup_all_statespace_nd(ode_problem, impl_fn, solver_config):
    return _SolveFixedGridConfig(
        ode_problem=ode_problem,
        solver_fn=ivpsolvers.MLESolver,
        impl_fn=impl_fn,
        strat_fn=filters.Filter,
        solver_config=solver_config,
        output_scale=1.0,
    )


@testing.case
@testing.parametrize_with_cases(
    "ode_problem", cases="..problem_cases", has_tag="scalar"
)
@testing.parametrize_with_cases("impl_fn", cases="..statespace_cases", has_tag="scalar")
def case_setup_all_statespace_scalar(ode_problem, impl_fn, solver_config):
    return _SolveFixedGridConfig(
        ode_problem=ode_problem,
        solver_fn=ivpsolvers.MLESolver,
        impl_fn=impl_fn,
        strat_fn=filters.Filter,
        solver_config=solver_config,
        output_scale=1.0,
    )


@testing.case
@testing.parametrize_with_cases("ode_problem", cases="..problem_cases", has_tag="nd")
@testing.parametrize(
    "strat_fn", [filters.Filter, smoothers.Smoother, smoothers.FixedPointSmoother]
)
def case_setup_all_strategies(ode_problem, strat_fn, solver_config):
    return _SolveFixedGridConfig(
        ode_problem=ode_problem,
        solver_fn=ivpsolvers.MLESolver,
        impl_fn=recipes.ts0_blockdiag,
        strat_fn=strat_fn,
        solver_config=solver_config,
        output_scale=1.0,
    )


@testing.case
@testing.parametrize_with_cases("ode_problem", cases="..problem_cases", has_tag="nd")
@testing.parametrize_with_cases("solver_fn", cases="..ivpsolver_cases")
def case_setup_all_ivpsolvers(ode_problem, solver_fn, solver_config):
    return _SolveFixedGridConfig(
        ode_problem=ode_problem,
        solver_fn=solver_fn,
        impl_fn=recipes.ts0_blockdiag,
        strat_fn=filters.Filter,
        solver_config=solver_config,
        output_scale=1.0,
    )


# Compute the IVP solution for given setups


@testing.fixture(name="solution_fixed_grid")
@testing.parametrize_with_cases(
    "setup", cases=".", prefix="case_setup_", scope="session"
)
def fixture_solution_fixed_grid(setup):
    ode_shape = setup.ode_problem.initial_values[0].shape
    solver = test_util.generate_solver(
        solver_factory=setup.solver_fn,
        strategy_factory=setup.strat_fn,
        impl_factory=setup.impl_fn,
        ode_shape=ode_shape,
        num_derivatives=4,
    )

    t0, t1 = setup.ode_problem.t0, setup.ode_problem.t1
    grid = setup.solver_config.grid_for_fixed_grid_fn(t0, t1)

    solution = ivpsolve.solve_fixed_grid(
        setup.ode_problem.vector_field,
        setup.ode_problem.initial_values,
        grid=grid,
        parameters=setup.ode_problem.args,
        solver=solver,
        output_scale=setup.output_scale,
    )

    sol = (solution.t, solution.u)
    sol_ref = (grid, jax.vmap(setup.ode_problem.solution)(grid))
    return sol, sol_ref


def test_terminal_values_correct(solution_fixed_grid, solver_config):
    (t, u), (t_ref, u_ref) = solution_fixed_grid
    atol, rtol = solver_config.atol_assert, solver_config.rtol_assert
    assert jnp.allclose(t[-1], t_ref[-1], atol=atol, rtol=rtol)
    assert jnp.allclose(u[-1], u_ref[-1], atol=atol, rtol=rtol)


@testing.fixture(name="parameter_to_solution")
@testing.parametrize_with_cases(
    "setup", cases=".", prefix="case_setup_", scope="session"
)
def fixture_parameter_to_solution(setup):
    """Parameter-to-solution map. To be differentiated."""
    ode_shape = setup.ode_problem.initial_values[0].shape
    solver = test_util.generate_solver(
        solver_factory=setup.solver_fn,
        strategy_factory=setup.strat_fn,
        impl_factory=setup.impl_fn,
        ode_shape=ode_shape,
        num_derivatives=1,  # Low order traces more quickly
    )
    t0, t1 = setup.ode_problem.t0, setup.ode_problem.t1
    grid = setup.solver_config.grid_for_fixed_grid_fn(t0, t1)

    def fn(u0):
        solution = ivpsolve.solve_fixed_grid(
            setup.ode_problem.vector_field,
            u0,
            grid=grid,
            parameters=setup.ode_problem.args,
            solver=solver,
        )
        return solution.u

    # DenseSLR1(ThirdOrderSpherical) has a NaN vector-Jacobian product.
    # Therefore, we skip all autodiff-DenseSLR1 tests
    # until the VJP/JVP behaviour has been cleaned up.
    # (It is easier to skip them all for now than to investigate how to skip
    # this very specific instance.) See: Issue #500.
    corr = solver.strategy.implementation.correction
    skip_jvp_and_vjp = isinstance(corr, dense_corr._DenseStatisticalFirstOrder)
    return fn, setup.ode_problem.initial_values, skip_jvp_and_vjp


def test_jvp(parameter_to_solution, solver_config):
    fn, primals, skip_jvp = parameter_to_solution
    if skip_jvp:
        reason1 = "DenseSLR1 is not guaranteed to have valid JVPs at the moment. "
        reason2 = "See: #500."
        testing.skip(reason1 + reason2)

    jvp = functools.partial(jax.jvp, fn)

    # Autodiff tests are sometimes a bit flaky...
    # There is also no clear mathematical argument that solutions that
    # have been computed with atol A and rtol R should have
    # gradients that coincide with their finite difference approximations with
    # accuracies A and R. Therefore, we relax them a little bit.
    # todo: move autodiff_atol_assert, autodiff_rtol_assert into solver config.
    atol, rtol = 10 * solver_config.atol_assert, 10 * solver_config.rtol_assert
    jax.test_util.check_jvp(f=fn, f_jvp=jvp, args=(primals,), atol=atol, rtol=rtol)


def test_vjp(parameter_to_solution, solver_config):
    fn, primals, skip_vjp = parameter_to_solution
    if skip_vjp:
        reason1 = "DenseSLR1 is not guaranteed to have valid VJPs at the moment. "
        reason2 = "See: #500."
        testing.skip(reason1 + reason2)
    vjp = functools.partial(jax.vjp, fn)

    # Autodiff tests are sometimes a bit flaky...
    # There is also no clear mathematical argument that solutions that
    # have been computed with atol A and rtol R should have
    # gradients that coincide with their finite difference approximations with
    # accuracies A and R. Therefore, we relax them a little bit.
    # todo: move autodiff_atol_assert, autodiff_rtol_assert into solver config.
    atol, rtol = 10 * solver_config.atol_assert, 10 * solver_config.rtol_assert
    jax.test_util.check_vjp(f=fn, f_vjp=vjp, args=(primals,), atol=atol, rtol=rtol)
