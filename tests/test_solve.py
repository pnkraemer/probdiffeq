"""Tests for IVP solvers."""
import jax.numpy as jnp
import pytest
from jax.experimental.ode import odeint
from pytest_cases import parametrize_with_cases

from odefilter import ivpsolve


@parametrize_with_cases("vf, u0, t0, t1, p", cases=".ivp_cases", prefix="problem_")
@parametrize_with_cases(
    "solver, info_op", cases=".recipe_cases", prefix="solver_", has_tag=("solve",)
)
def test_solve(vf, u0, t0, t1, p, solver, info_op):
    ts = jnp.linspace(t0, t1, num=10)
    odeint_solution = odeint(
        lambda y, t, *par: vf(y, t=t, p=par), u0[0], ts, *p, atol=1e-6, rtol=1e-6
    )
    ts_reference, ys_reference = ts, odeint_solution

    solution = ivpsolve.solve(
        vf, u0, t0=t0, t1=t1, parameters=p, solver=solver, info_op=info_op
    )
    assert jnp.allclose(solution.t[-1], ts_reference[-1])
    assert jnp.allclose(solution.u[-1], ys_reference[-1], atol=1e-3, rtol=1e-3)

    # Some iter-ability-checks for the solution
    assert isinstance(solution[0], type(solution))
    assert len(solution) == len(solution.t)

    # __getitem__ only works for batched solutions.
    with pytest.raises(ValueError):
        _ = solution[0][0]
    with pytest.raises(ValueError):
        _ = solution[0, 0]

    for i, sol in zip(range(2 * len(solution)), solution):
        assert isinstance(sol, type(solution))

    assert i == len(solution) - 1


@parametrize_with_cases("vf, u0, t0, t1, p", cases=".ivp_cases", prefix="problem_")
@parametrize_with_cases(
    "solver, info_op", cases=".recipe_cases", prefix="solver_", has_tag=("solve",)
)
def test_dense_output(vf, u0, t0, t1, p, solver, info_op):
    solution = ivpsolve.solve(
        vf,
        u0,
        t0=t0,
        t1=t1,
        parameters=p,
        solver=solver,
        info_op=info_op,
        atol=1e-1,
        rtol=1e-1,
    )
    midpoint = (solution[0].t + solution[1].t) / 2
    dense = solver.dense_output(
        t=midpoint, state=solution[1], state_previous=solution[0]
    )

    assert isinstance(dense, type(solution))

    # Extrapolate from the left: close-to-left boundary must be similar,
    # but close-to-right boundary must not be similar
    # todo: this only applies to the filter! The smoother must be close to both!
    close_to_left = solver.dense_output(
        t=solution[0].t + 1e-4, state=solution[1], state_previous=solution[0]
    )
    close_to_right = solver.dense_output(
        t=solution[1].t - 1e-4, state=solution[1], state_previous=solution[0]
    )
    assert not jnp.allclose(close_to_right.u, solution[0].u, atol=1e-3, rtol=1e-3)
    assert jnp.allclose(close_to_left.u, solution[0].u, atol=1e-3, rtol=1e-3)

    ts = jnp.linspace(t0 + 1e-4, t1 - 1e-4, num=4, endpoint=True)
    dense = solver.dense_output_searchsorted(ts=ts, solution=solution)
    assert jnp.allclose(dense.t, ts)
    # check we correctly landed in the first interval
    assert jnp.allclose(dense.u[0], solution.u[0], atol=1e-3, rtol=1e-3)
    assert not jnp.allclose(dense.u[0], solution.u[1], atol=1e-3, rtol=1e-3)
