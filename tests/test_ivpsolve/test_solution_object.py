"""Tests for interaction with the solution object."""
import jax
import jax.numpy as jnp

from probdiffeq import ivpsolve
from probdiffeq.backend import testing
from probdiffeq.impl import impl
from probdiffeq.solvers import calibrated
from probdiffeq.solvers.statespace import correction, extrapolation
from probdiffeq.solvers.strategies import filters
from probdiffeq.solvers.taylor import autodiff
from tests.setup import setup


@testing.fixture(name="approximate_solution")
def fixture_approximate_solution():
    vf, u0, (t0, t1) = setup.ode()

    # Generate a solver
    ibm = extrapolation.ibm_adaptive(num_derivatives=1)
    ts0 = correction.taylor_order_zero()
    strategy = filters.filter_adaptive(ibm, ts0)
    solver = calibrated.mle(strategy)

    output_scale = jnp.ones_like(impl.prototypes.output_scale())
    tcoeffs = autodiff.taylor_mode(lambda y: vf(y, t=t0), u0, num=1)
    return ivpsolve.solve_and_save_every_step(
        vf,
        tcoeffs,
        t0=t0,
        t1=t1,
        dt0=0.1,
        solver=solver,
        output_scale=output_scale,
        atol=1e-2,
        rtol=1e-2,
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
def fixture_approximate_solution_batched():
    vf, (u0,), (t0, t1) = setup.ode()

    # Generate a solver
    ibm = extrapolation.ibm_adaptive(num_derivatives=1)
    ts0 = correction.taylor_order_zero()
    strategy = filters.filter_adaptive(ibm, ts0)
    solver = calibrated.mle(strategy)

    output_scale = jnp.ones_like(impl.prototypes.output_scale())

    save_at = jnp.linspace(t0, t1, endpoint=True, num=4)

    @jax.vmap
    def solve(init):
        return ivpsolve.solve_and_save_at(
            vf,
            (init, vf(init, t=None)),
            save_at=save_at,
            solver=solver,
            dt0=0.1,
            output_scale=output_scale,
            atol=1e-2,
            rtol=1e-2,
        )

    return solve(jnp.stack((u0, u0 + 0.1, u0 + 0.2)))


def test_batched_getitem_possible(approximate_solution_batched):
    solution_type = type(approximate_solution_batched)
    for idx in (0, 1, 2):
        approximate_solution = approximate_solution_batched[idx]
        assert isinstance(approximate_solution, solution_type)
        assert jnp.allclose(approximate_solution.t, approximate_solution_batched.t[idx])
        assert jnp.allclose(approximate_solution.u, approximate_solution_batched.u[idx])


def test_batched_iter_possible(approximate_solution_batched):
    solution_type = type(approximate_solution_batched)
    for idx, approximate_solution in enumerate(approximate_solution_batched):
        assert isinstance(approximate_solution, solution_type)
        assert jnp.allclose(approximate_solution.t, approximate_solution_batched.t[idx])
        assert jnp.allclose(approximate_solution.u, approximate_solution_batched.u[idx])


def test_marginal_nth_derivative_of_solution(approximate_solution):
    """Assert that each $n$th derivative matches the quantity of interest's shape."""
    # Assert that the marginals have the same shape as the qoi.
    for i in (0, 1):
        marginals = approximate_solution.marginals
        derivatives = impl.hidden_model.marginal_nth_derivative(marginals, i)
        assert derivatives.mean.shape == approximate_solution.u.shape

    # if the requested derivative is not in the state-space model, raise a ValueError
    with testing.raises(ValueError):
        _ = impl.hidden_model.marginal_nth_derivative(
            approximate_solution.marginals, 100
        )
