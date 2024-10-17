"""Tests for interaction with the solution object."""

from probdiffeq import ivpsolve, ivpsolvers, taylor
from probdiffeq.backend import functools, testing
from probdiffeq.backend import numpy as np
from probdiffeq.impl import impl


@testing.fixture(name="approximate_solution")
def fixture_approximate_solution(ssm):
    vf, u0, (t0, t1) = ssm.default_ode

    # Generate a solver
    output_scale = np.ones_like(impl.prototypes.output_scale())
    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), u0, num=1)
    ibm = ivpsolvers.prior_ibm(tcoeffs, output_scale=output_scale)

    ts0 = ivpsolvers.correction_ts0()
    strategy = ivpsolvers.strategy_filter(ibm, ts0)
    solver = ivpsolvers.solver_mle(strategy)
    adaptive_solver = ivpsolve.adaptive(solver, atol=1e-2, rtol=1e-2)

    init = solver.initial_condition()
    return ivpsolve.solve_adaptive_save_every_step(
        vf, init, t0=t0, t1=t1, dt0=0.1, adaptive_solver=adaptive_solver
    )


def test_getitem_possible_for_terminal_values(approximate_solution):
    solution_t1 = approximate_solution[-1]
    assert isinstance(solution_t1, type(approximate_solution))


@testing.parametrize("item", [-2, 0, slice(1, -1, 1)])
def test_getitem_impossible_for_nonterminal_values(approximate_solution, item):
    with testing.raises(ValueError, match="non-terminal"):
        _ = approximate_solution[item]


@testing.parametrize("item", [-1, -2, 0, slice(1, -1, 1)])
def test_getitem_impossible_at_single_time_for_any_item(approximate_solution, item):
    # Allowed slicing:
    # solution_t1 is not batched now, so further slicing should be impossible
    solution_t1 = approximate_solution[-1]

    with testing.raises(ValueError, match="not batched"):
        _ = solution_t1[item]


def test_iter_impossible(approximate_solution):
    with testing.raises(ValueError, match="not batched"):
        for _ in approximate_solution:
            pass


@testing.fixture(name="approximate_solution_batched")
def fixture_approximate_solution_batched(ssm):
    vf, (u0,), (t0, t1) = ssm.default_ode

    # Generate a solver
    save_at = np.linspace(t0, t1, endpoint=True, num=4)

    @functools.vmap
    def solve(init):
        tcoeffs = (init, vf(init, t=None))
        output_scale = np.ones_like(impl.prototypes.output_scale())
        ibm = ivpsolvers.prior_ibm(tcoeffs, output_scale=output_scale)

        ts0 = ivpsolvers.correction_ts0()
        strategy = ivpsolvers.strategy_filter(ibm, ts0)
        solver = ivpsolvers.solver_mle(strategy)
        adaptive_solver = ivpsolve.adaptive(solver, atol=1e-2, rtol=1e-2)

        initcond = solver.initial_condition()
        return ivpsolve.solve_adaptive_save_at(
            vf, initcond, save_at=save_at, adaptive_solver=adaptive_solver, dt0=0.1
        )

    return solve(np.stack((u0, u0 + 0.1, u0 + 0.2)))


def test_batched_getitem_possible(approximate_solution_batched):
    solution_type = type(approximate_solution_batched)
    for idx in (0, 1, 2):
        approximate_solution = approximate_solution_batched[idx]
        assert isinstance(approximate_solution, solution_type)
        assert np.allclose(approximate_solution.t, approximate_solution_batched.t[idx])
        assert np.allclose(approximate_solution.u, approximate_solution_batched.u[idx])


def test_batched_iter_possible(approximate_solution_batched):
    solution_type = type(approximate_solution_batched)
    for idx, approximate_solution in enumerate(approximate_solution_batched):
        assert isinstance(approximate_solution, solution_type)
        assert np.allclose(approximate_solution.t, approximate_solution_batched.t[idx])
        assert np.allclose(approximate_solution.u, approximate_solution_batched.u[idx])


def test_marginal_nth_derivative_of_solution(approximate_solution):
    """Assert that each $n$th derivative matches the quantity of interest's shape."""
    # Assert that the marginals have the same shape as the qoi.
    for i in (0, 1):
        marginals = approximate_solution.marginals
        derivatives = impl.stats.marginal_nth_derivative(marginals, i)
        assert derivatives.mean.shape == approximate_solution.u.shape

    # if the requested derivative is not in the state-space model, raise a ValueError
    with testing.raises(ValueError):
        _ = impl.stats.marginal_nth_derivative(approximate_solution.marginals, 100)
