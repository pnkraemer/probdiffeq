"""Tests for log-marginal-likelihood functionality."""
import diffeqzoo.ivps
import jax
import jax.numpy as jnp

from probdiffeq import ivpsolve, solution, test_util
from probdiffeq.backend import testing
from probdiffeq.ivpsolvers import uncalibrated
from probdiffeq.statespace import recipes
from probdiffeq.strategies import filters, smoothers


@testing.fixture(name="problem")
def fixture_problem():
    f, u0, (t0, _), f_args = diffeqzoo.ivps.lotka_volterra()
    t1 = 2.0  # Short time-intervals are sufficient for this test.

    @jax.jit
    def vf(x, *, t, p):  # pylint: disable=unused-argument
        return f(x, *p)

    return vf, u0, (t0, t1), f_args


@testing.case()
def case_isotropic_factorisation():
    return recipes.ts0_iso, 2.0


@testing.case()  # this implies success of the scalar solver
def case_blockdiag_factorisation():
    return recipes.ts0_blockdiag, jnp.ones((2,)) * 2.0


@testing.case()
def case_dense_factorisation():
    return recipes.ts0_dense, 2.0


@testing.fixture(name="solution_save_at")
@testing.parametrize_with_cases("factorisation", cases=".", prefix="case_")
def fixture_solution_save_at(problem, factorisation):
    vf, u0, (t0, t1), params = problem
    impl_fn, output_scale = factorisation
    solver = test_util.generate_solver(
        strategy_factory=smoothers.smoother_fixedpoint,
        impl_factory=impl_fn,
        solver_factory=uncalibrated.solver,
        ode_shape=jnp.shape(u0),
        num_derivatives=2,
    )

    save_at = jnp.linspace(t0, t1, endpoint=True, num=4)
    sol = ivpsolve.solve_and_save_at(
        vf,
        (u0,),
        save_at=save_at,
        parameters=params,
        solver=solver,
        atol=1e-2,
        rtol=1e-2,
        output_scale=output_scale,
    )
    return sol, solver


def test_output_is_a_scalar_and_not_nan_and_not_inf(solution_save_at):
    sol, solver = solution_save_at
    data = sol.u + 0.005
    lml = solution.log_marginal_likelihood(
        observation_std=1.0,
        u=data,
        posterior=sol.posterior,
        strategy=solver.strategy,
    )
    assert lml.shape == ()
    assert not jnp.isnan(lml)
    assert not jnp.isinf(lml)


def test_that_function_raises_error_for_wrong_std_shape_too_many(solution_save_at):
    """Test that the log-marginal-likelihood function complains about the wrong shape.

    Specifically, about receiving more standard-deviations than data-points.
    """
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


def test_that_function_raises_error_for_wrong_std_shape_wrong_ndim(solution_save_at):
    """Test that the log-marginal-likelihood function complains about the wrong shape.

    Specifically, about receiving non-scalar standard-deviations.
    """
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


def test_raises_error_for_terminal_values(solution_save_at):
    """Test that the log-marginal-likelihood function complains when called incorrectly.

    Specifically, raise an error when calling log_marginal_likelihood even though
    log_marginal_likelihood_terminal_values was meant.
    """
    sol, solver = solution_save_at
    data = sol.u + 0.005

    posterior_t1 = jax.tree_util.tree_map(lambda s: s[-1], sol)
    with testing.raises(ValueError, match="expected"):
        _ = solution.log_marginal_likelihood(
            observation_std=jnp.ones_like(data[-1]),
            u=data[-1],
            posterior=posterior_t1,
            strategy=solver.strategy,
        )


def test_raises_error_for_filter(problem):
    """Non-terminal value calls are not possible for filters."""
    vf, u0, (t0, t1), params = problem

    recipe = recipes.ts0_iso(num_derivatives=4)
    strategy, calibration = filters.filter(*recipe)
    solver = uncalibrated.solver(strategy, calibration)
    grid = jnp.linspace(t0, t1, num=3)

    sol = ivpsolve.solve_fixed_grid(
        vf,
        (u0,),
        grid=grid,
        parameters=params,
        solver=solver,
        output_scale=1.0,
    )
    data = sol.u + 0.1
    std = jnp.ones((sol.u.shape[0],))  # values irrelevant
    with testing.raises(TypeError, match="ilter"):
        _ = solution.log_marginal_likelihood(
            observation_std=std, u=data, posterior=sol.posterior, strategy=strategy
        )
