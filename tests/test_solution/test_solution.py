"""Tests for IVP solvers."""
import jax
import jax.numpy as jnp

from probdiffeq import ivpsolve, solution, test_util
from probdiffeq.backend import testing
from probdiffeq.strategies import filters, smoothers


@testing.fixture(scope="function", name="solution_native_python_while_loop")
@testing.parametrize_with_cases("ode_problem", cases="..problem_cases", has_tag="nd")
def fixture_solution_native_python_while_loop(ode_problem, strategy_fn):
    solver = test_util.generate_solver(num_derivatives=1, strategy_factory=strategy_fn)
    sol = ivpsolve.solve_with_python_while_loop(
        ode_problem.vector_field,
        ode_problem.initial_values,
        t0=ode_problem.t0,
        t1=ode_problem.t1,
        parameters=ode_problem.args,
        solver=solver,
        atol=1e-1,
        rtol=1e-2,
    )
    return sol, solver


@testing.parametrize("strategy_fn", [filters.Filter], scope="function")
def test_solution_is_iterable(solution_native_python_while_loop):
    sol, _ = solution_native_python_while_loop
    assert isinstance(sol[0], type(sol))
    assert len(sol) == len(sol.t)


@testing.parametrize("strategy_fn", [filters.Filter], scope="function")
def test_getitem_raises_error_for_nonbatched_solutions(
    solution_native_python_while_loop,
):
    """__getitem__ only works for batched solutions."""
    sol, _ = solution_native_python_while_loop
    with testing.raises(ValueError):
        _ = sol[0][0]
    with testing.raises(ValueError):
        _ = sol[0, 0]


@testing.parametrize("strategy_fn", [filters.Filter], scope="function")
def test_loop_over_solution_is_possible(solution_native_python_while_loop):
    solution_full, _ = solution_native_python_while_loop

    i = 0
    for i, sol in zip(range(2 * len(solution_full)), solution_full):
        assert isinstance(sol, type(solution_full))

    assert i > 0
    assert i == len(solution_full) - 1


@testing.parametrize("strategy_fn", [filters.Filter], scope="function")
def test_marginal_nth_derivative_of_solution(solution_native_python_while_loop):
    """Assert that each $n$th derivative matches the quantity of interest's shape."""
    sol, _ = solution_native_python_while_loop

    # Assert that the marginals have the same shape as the qoi.
    for i in (0, 1):
        derivatives = sol.marginals.marginal_nth_derivative(i)
        assert derivatives.mean.shape == sol.u.shape

    # if the requested derivative is not in the state-space model, raise a ValueError
    with testing.raises(ValueError):
        sol.marginals.marginal_nth_derivative(100)


@testing.parametrize("strategy_fn", [filters.Filter], scope="function")
def test_offgrid_marginals_filter(solution_native_python_while_loop):
    """Assert that the offgrid-marginals are close to the boundary values."""
    sol, solver = solution_native_python_while_loop
    t0, t1 = sol.t[0], sol.t[-1]

    # Extrapolate from the left: close-to-left boundary must be similar,
    # but close-to-right boundary needs not be similar
    ts = jnp.linspace(t0 + 1e-4, t1 - 1e-4, num=4, endpoint=True)
    u, _ = solution.offgrid_marginals_searchsorted(ts=ts, solution=sol, solver=solver)
    assert jnp.allclose(u[0], sol.u[0], atol=1e-3, rtol=1e-3)
    assert not jnp.allclose(u[0], sol.u[1], atol=1e-3, rtol=1e-3)


@testing.parametrize("strategy_fn", [smoothers.Smoother], scope="function")
def test_offgrid_marginals_smoother(solution_native_python_while_loop):
    sol, solver = solution_native_python_while_loop
    t0, t1 = sol.t[0], sol.t[-1]

    # Extrapolate from the left: close-to-left boundary must be similar,
    # but close-to-right boundary must not be similar
    ts = jnp.linspace(t0 + 1e-4, t1 - 1e-4, num=4, endpoint=True)
    u, _ = solution.offgrid_marginals_searchsorted(ts=ts, solution=sol, solver=solver)
    assert jnp.allclose(u[0], sol.u[0], atol=1e-3, rtol=1e-3)
    assert jnp.allclose(u[-1], sol.u[-1], atol=1e-3, rtol=1e-3)


@testing.fixture(scope="function", name="solution_save_at")
@testing.parametrize_with_cases("ode_problem", cases="..problem_cases", has_tag=["nd"])
def fixture_solution_save_at(ode_problem):
    solver = test_util.generate_solver(strategy_factory=smoothers.FixedPointSmoother)

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


@testing.parametrize("shape", [(), (2,), (2, 2)], ids=["()", "(n,)", "(n,n)"])
def test_sample_shape(solution_save_at, shape):
    sol, solver = solution_save_at

    key = jax.random.PRNGKey(seed=15)
    # todo: remove "u" from this output?
    u, samples = sol.posterior.sample(key, shape=shape)
    assert u.shape == shape + sol.u.shape
    assert samples.shape == shape + sol.marginals.hidden_state.sample_shape

    # Todo: test values of the samples by checking a chi2 statistic
    #  in terms of the joint posterior. But this requires a joint_posterior()
    #  method, which is only future work I guess. So far we use the eye-test
    #  in the notebooks, which looks good.
