"""Tests for marginal log likelihoods."""
import jax.numpy as jnp

from probdiffeq import ivpsolve, ivpsolvers, solution, test_util
from probdiffeq.backend import testing
from probdiffeq.statespace import recipes
from probdiffeq.strategies import filters, smoothers


@testing.fixture(name="solution_save_at")
@testing.parametrize_with_cases("ode_problem", cases="..problem_cases", has_tag="nd")
@testing.parametrize(
    "impl_fn",
    # one for each SSM factorisation
    [
        lambda num_derivatives, **kwargs: recipes.ts0_iso(
            num_derivatives=num_derivatives
        ),
        recipes.ts0_blockdiag,
        recipes.ts0_dense,
    ],
    ids=["IsoTS0", "BlockDiagTS0", "DenseTS0"],
)
@testing.parametrize("strategy_fn", [filters.filter, smoothers.smoother_fixedpoint])
def fixture_solution_save_at(ode_problem, impl_fn, strategy_fn):
    solver = test_util.generate_solver(
        strategy_factory=strategy_fn,
        impl_factory=impl_fn,
        ode_shape=ode_problem.initial_values[0].shape,
        num_derivatives=2,
    )

    save_at = jnp.linspace(ode_problem.t0, ode_problem.t1, endpoint=True, num=4)
    sol = ivpsolve.solve_and_save_at(
        ode_problem.vector_field,
        ode_problem.initial_values,
        save_at=save_at,
        parameters=ode_problem.args,
        solver=solver,
        atol=1e-1,
        rtol=1e-2,
    )
    return sol, solver


def test_log_marginal_likelihood(solution_save_at):
    sol, solver = solution_save_at
    data = sol.u + 0.005
    k = sol.u.shape[0]
    if isinstance(solver.strategy, filters._Filter):  # noqa: E731
        reason = "No time-series log marginal likelihoods for Filters."
        testing.skip(reason)
    mll = solution.log_marginal_likelihood(
        observation_std=jnp.ones((k,)),
        u=data,
        posterior=sol.posterior,
        strategy=solver.strategy,
    )
    assert mll.shape == ()
    assert not jnp.isnan(mll)
    assert not jnp.isinf(mll)


def test_log_marginal_likelihood_error_for_wrong_std_shape_1(solution_save_at):
    sol, solver = solution_save_at
    data = sol.u + 0.005
    k = sol.u.shape[0]

    with testing.raises(ValueError, match="does not match"):
        _ = solution.log_marginal_likelihood(
            observation_std=jnp.ones((k + 1,)),
            u=data,
            posterior=sol.posterior,
            strategy=solver.strategy,
        )


def test_log_marginal_likelihood_error_for_wrong_std_shape_2(solution_save_at):
    sol, solver = solution_save_at
    data = sol.u + 0.005
    k = sol.u.shape[0]

    with testing.raises(ValueError, match="does not match"):
        _ = solution.log_marginal_likelihood(
            observation_std=jnp.ones((k, 1)),
            u=data,
            posterior=sol.posterior,
            strategy=solver.strategy,
        )


def test_log_marginal_likelihood_error_for_terminal_values(solution_save_at):
    sol, solver = solution_save_at
    data = sol.u + 0.005

    with testing.raises(ValueError, match="expected"):
        _ = solution.log_marginal_likelihood(
            observation_std=jnp.ones_like(data[-1]),
            u=data[-1],
            posterior=sol[-1].posterior,
            strategy=solver.strategy,
        )


def test_log_marginal_likelihood_terminal_values(solution_save_at):
    sol, solver = solution_save_at
    data = sol.u + 0.005

    mll = solution.log_marginal_likelihood_terminal_values(
        observation_std=jnp.ones(()),
        u=data[-1],
        posterior=sol[-1].posterior,
        strategy=solver.strategy,
    )
    assert mll.shape == ()
    assert not jnp.isnan(mll)
    assert not jnp.isinf(mll)


def test_log_marginal_likelihood_terminal_values_error_for_wrong_shapes(
    solution_save_at,
):
    sol, solver = solution_save_at
    data = sol.u + 0.005

    with testing.raises(ValueError, match="expected"):
        _ = solution.log_marginal_likelihood_terminal_values(
            observation_std=jnp.ones((1,)),
            u=data[-1],
            posterior=sol[-1].posterior,
            strategy=solver.strategy,
        )

    with testing.raises(ValueError, match="expected"):
        _ = solution.log_marginal_likelihood_terminal_values(
            observation_std=jnp.ones(()),
            u=data[-1:],
            posterior=sol[-1:].posterior,
            strategy=solver.strategy,
        )


@testing.parametrize_with_cases("ode_problem", cases="..problem_cases", has_tag="nd")
@testing.parametrize("strategy_fn", [filters.filter, smoothers.smoother])
def test_filter_ts0_iso_terminal_value_nll(ode_problem, strategy_fn):
    """Issue #477."""
    recipe = recipes.ts0_iso(num_derivatives=4)
    strategy = strategy_fn(*recipe)
    solver = ivpsolvers.CalibrationFreeSolver(strategy)
    sol = ivpsolve.simulate_terminal_values(
        ode_problem.vector_field,
        initial_values=ode_problem.initial_values,
        t0=ode_problem.t0,
        t1=ode_problem.t1,
        parameters=ode_problem.args,
        solver=solver,
    )
    data = sol.u + 0.1
    mll = solution.log_marginal_likelihood_terminal_values(
        observation_std=1e-2, u=data, posterior=sol.posterior, strategy=strategy
    )
    assert mll.shape == ()
    assert not jnp.isnan(mll)
    assert not jnp.isinf(mll)


@testing.parametrize_with_cases("ode_problem", cases="..problem_cases", has_tag="nd")
def test_nmll_raises_error_for_filter(ode_problem):
    """Non-terminal value calls are not possible for filters."""
    recipe = recipes.ts0_iso(num_derivatives=4)
    strategy = filters.filter(*recipe)
    solver = ivpsolvers.CalibrationFreeSolver(strategy)
    grid = jnp.linspace(ode_problem.t0, ode_problem.t1)

    sol = ivpsolve.solve_fixed_grid(
        ode_problem.vector_field,
        initial_values=ode_problem.initial_values,
        grid=grid,
        parameters=ode_problem.args,
        solver=solver,
    )
    data = sol.u + 0.1
    std = jnp.ones((sol.u.shape[0],))  # values irrelevant
    with testing.raises(TypeError, match="ilter"):
        _ = solution.log_marginal_likelihood(
            observation_std=std, u=data, posterior=sol.posterior, strategy=strategy
        )
