"""Tests for marginal log likelihood functionality (terminal values)."""
import diffeqzoo.ivps
import jax
import jax.numpy as jnp

from probdiffeq import ivpsolve, ivpsolvers, solution
from probdiffeq.backend import testing
from probdiffeq.statespace import recipes
from probdiffeq.strategies import filters, smoothers


@testing.fixture(name="problem", scope="module")
def fixture_problem():
    f, u0, (t0, _), f_args = diffeqzoo.ivps.lotka_volterra()
    t1 = 2.0  # Short time-intervals are sufficient for this test.

    @jax.jit
    def vf(x, *, t, p):  # pylint: disable=unused-argument
        return f(x, *p)

    return vf, u0, (t0, t1), f_args


@testing.case()
def case_impl_isotropic_factorisation():
    def iso_factory(ode_shape, num_derivatives):
        return recipes.ts0_iso(num_derivatives=num_derivatives)

    return iso_factory


@testing.case()  # this implies success of the scalar solver
def case_impl_blockdiag_factorisation():
    return recipes.ts0_blockdiag


@testing.case()
def case_impl_dense_factorisation():
    return recipes.ts0_dense


@testing.parametrize_with_cases("impl_factory", cases=".", prefix="case_impl")
@testing.case()
def case_sol_vary_the_statespace(problem, impl_factory):
    vf, u0, (t0, t1), params = problem
    recipe = impl_factory(num_derivatives=4, ode_shape=jnp.shape(u0))
    strategy, calibration = filters.filter(*recipe)
    solver = ivpsolvers.solver_calibrationfree(strategy, calibration)
    sol = ivpsolve.simulate_terminal_values(
        vf, (u0,), t0=t0, t1=t1, parameters=params, solver=solver, atol=1e-2, rtol=1e-2
    )
    return sol, strategy


_STRATEGY_FUNS = [filters.filter, smoothers.smoother_fixedpoint, smoothers.smoother]


@testing.parametrize("strategy_fun", _STRATEGY_FUNS)
@testing.case()
def case_sol_vary_the_strategy(problem, strategy_fun):
    vf, u0, (t0, t1), params = problem
    recipe = recipes.ts0_iso(num_derivatives=4)
    strategy, calibration = strategy_fun(*recipe)
    solver = ivpsolvers.solver_calibrationfree(strategy, calibration)
    sol = ivpsolve.simulate_terminal_values(
        vf, (u0,), t0=t0, t1=t1, parameters=params, solver=solver, atol=1e-2, rtol=1e-2
    )
    return sol, strategy


@testing.parametrize_with_cases("solution_and_strategy", cases=".", prefix="case_sol_")
def test_output_is_scalar_and_not_inf_and_not_nan(solution_and_strategy):
    """Test that terminal-value log-marginal-likelihood calls work with all strategies.

    See also: issue #477 (closed).
    """
    sol, strategy = solution_and_strategy

    data = sol.u + 0.1
    mll = solution.log_marginal_likelihood_terminal_values(
        observation_std=jnp.asarray(1e-2),
        u=data,
        posterior=sol.posterior,
        strategy=strategy,
    )
    assert mll.shape == ()
    assert not jnp.isnan(mll)
    assert not jnp.isinf(mll)


@testing.parametrize_with_cases("solution_and_strategy", cases=".", prefix="case_sol_")
def test_terminal_values_error_for_wrong_shapes(solution_and_strategy):
    sol, strategy = solution_and_strategy
    data = sol.u + 0.005

    # Non-scalar observation std
    with testing.raises(ValueError, match="expected"):
        _ = solution.log_marginal_likelihood_terminal_values(
            observation_std=jnp.ones((1,)),
            u=data[-1],
            posterior=sol.posterior,
            strategy=strategy,
        )

    # Data does not match u
    with testing.raises(ValueError, match="expected"):
        _ = solution.log_marginal_likelihood_terminal_values(
            observation_std=jnp.ones(()),
            u=data[None, :],
            posterior=sol.posterior,
            strategy=strategy,
        )
