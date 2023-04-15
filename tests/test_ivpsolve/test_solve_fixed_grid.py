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
@testing.parametrize(
    "strat_fn", [filters.Filter, smoothers.Smoother, smoothers.FixedPointSmoother]
)
def case_setup_all_strategy_statespace_nd(
    ode_problem, strat_fn, impl_fn, solver_config
):
    return _SolveFixedGridConfig(
        ode_problem=ode_problem,
        solver_fn=ivpsolvers.MLESolver,
        impl_fn=impl_fn,
        strat_fn=strat_fn,
        solver_config=solver_config,
        output_scale=1.0,
    )


@testing.case
@testing.parametrize_with_cases(
    "ode_problem", cases="..problem_cases", has_tag="scalar"
)
@testing.parametrize_with_cases("impl_fn", cases="..statespace_cases", has_tag="scalar")
@testing.parametrize(
    "strat_fn", [filters.Filter, smoothers.Smoother, smoothers.FixedPointSmoother]
)
def case_setup_all_strategy_statespace_scalar(
    ode_problem, strat_fn, impl_fn, solver_config
):
    return _SolveFixedGridConfig(
        ode_problem=ode_problem,
        solver_fn=ivpsolvers.MLESolver,
        impl_fn=impl_fn,
        strat_fn=strat_fn,
        solver_config=solver_config,
        output_scale=1.0,
    )


@testing.case
@testing.parametrize_with_cases("ode_problem", cases="..problem_cases", has_tag="nd")
@testing.parametrize(
    "strat_fn", [filters.Filter, smoothers.Smoother, smoothers.FixedPointSmoother]
)
@testing.parametrize_with_cases("solver_fn", cases="..ivpsolver_cases")
def case_setup_all_solvers_strategies(ode_problem, solver_fn, strat_fn, solver_config):
    return _SolveFixedGridConfig(
        ode_problem=ode_problem,
        solver_fn=solver_fn,
        impl_fn=recipes.ts0_blockdiag,
        strat_fn=strat_fn,
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

    skip_jvp_and_vjp = _skip_autodiff_test(solver)
    return fn, setup.ode_problem.initial_values, skip_jvp_and_vjp


def _skip_autodiff_test(solver):
    # Some solver-strategy combinations have NaN VJPs and sometimes JVPs as well.
    # We skip those tests until the VJP/JVP behaviour has been cleaned up.
    # See: Issue #500.

    # Ignore no-lambda flake8 check here. If we don't define those check-functions
    # in-line, the whole function becomes  too cluttered.
    _SLR1 = dense_corr._DenseStatisticalFirstOrder
    _SLR0 = dense_corr._DenseStatisticalZerothOrder
    _TS1 = dense_corr._DenseTaylorFirstOrder
    is_smoother = lambda x: isinstance(x, smoothers.Smoother)  # noqa: E731
    is_fp_smoother = lambda x: isinstance(x, smoothers.FixedPointSmoother)  # noqa: E731
    is_slr1 = lambda x: isinstance(x, _SLR1)  # noqa: E731
    is_slr0 = lambda x: isinstance(x, _SLR0)  # noqa: E731
    is_ts1 = lambda x: isinstance(x, _TS1)  # noqa: E731

    strategy = solver.strategy
    correction = solver.strategy.correction

    if is_slr1(correction):
        return True
    if is_slr0(correction) and is_smoother(strategy):
        return True
    if is_slr0(correction) and is_fp_smoother(strategy):
        return True
    if is_ts1(correction) and is_smoother(strategy):
        return True
    if is_ts1(correction) and is_fp_smoother(strategy):
        return True

    # All good now, no skipping.
    return False


def test_jvp(parameter_to_solution, solver_config):
    fn, primals, skip = parameter_to_solution
    if skip:
        reason1 = "Some corrections in combination with and some smoothers "
        reason2 = "are not guaranteed to have valid JVPs at the moment. "
        reason3 = "See: #500."
        testing.skip(reason1 + reason2 + reason3)

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
    fn, primals, skip = parameter_to_solution
    if skip:
        reason1 = "Some corrections in combination with and some smoothers "
        reason2 = "are not guaranteed to have valid VJPs at the moment. "
        reason3 = "See: #500."
        testing.skip(reason1 + reason2 + reason3)

    vjp = functools.partial(jax.vjp, fn)

    # Autodiff tests are sometimes a bit flaky...
    # There is also no clear mathematical argument that solutions that
    # have been computed with atol A and rtol R should have
    # gradients that coincide with their finite difference approximations with
    # accuracies A and R. Therefore, we relax them a little bit.
    # todo: move autodiff_atol_assert, autodiff_rtol_assert into solver config.
    atol, rtol = 10 * solver_config.atol_assert, 10 * solver_config.rtol_assert
    jax.test_util.check_vjp(f=fn, f_vjp=vjp, args=(primals,), atol=atol, rtol=rtol)
