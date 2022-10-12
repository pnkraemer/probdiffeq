"""Tests for IVP solvers."""
import jax
import jax.numpy as jnp
from diffeqzoo import ivps
from jax.experimental.ode import odeint
from pytest_cases import case, parametrize_with_cases

from odefilter import ivpsolve, recipes

# All tests assume that we are dealing with a first-order, 2d-problem!


@case(tags=("checkpoint",))
def solver_dynamic_isotropic_fixpt_eks0():
    return recipes.dynamic_isotropic_fixpt_eks0(num_derivatives=2)


@case(tags=("terminal_value", "solve"))
def solver_dynamic_isotropic_eks0():
    return recipes.dynamic_isotropic_eks0(num_derivatives=2)


@case(tags=("terminal_value", "solve", "checkpoint"))
def solver_dynamic_isotropic_ekf0():
    return recipes.dynamic_isotropic_ekf0(num_derivatives=2)


@case(tags=("terminal_value", "solve", "checkpoint"))
def solver_dynamic_ekf1():
    return recipes.dynamic_ekf1(num_derivatives=2, ode_dimension=2)


@case
def problem_lotka():
    f, u0, tspan, f_args = ivps.lotka_volterra()

    @jax.jit
    def vf(x, t, *p):
        return f(x, *p)

    return vf, (u0,), *tspan, f_args


@parametrize_with_cases("vf, u0, t0, t1, p", cases=".", prefix="problem_")
@parametrize_with_cases("solver", cases=".", prefix="solver_", has_tag=("solve",))
def test_solve(vf, u0, t0, t1, p, solver):
    ts = jnp.linspace(t0, t1, num=10)
    odeint_solution = odeint(vf, u0[0], ts, *p, atol=1e-6, rtol=1e-6)
    ts_reference, ys_reference = ts, odeint_solution

    solution = ivpsolve.solve(
        vector_field=vf,
        initial_values=u0,
        t0=t0,
        t1=t1,
        parameters=p,
        solver=solver,
    )
    assert jnp.allclose(solution.t[-1], ts_reference[-1])
    assert jnp.allclose(solution.u[-1], ys_reference[-1], atol=1e-3, rtol=1e-3)


@parametrize_with_cases("vf, u0, t0, t1, p", cases=".", prefix="problem_")
@parametrize_with_cases(
    "solver", cases=".", prefix="solver_", has_tag=("terminal_value",)
)
def test_simulate_terminal_values(vf, u0, t0, t1, p, solver):
    odeint_solution = odeint(vf, u0[0], jnp.asarray([t0, t1]), *p, atol=1e-6, rtol=1e-6)
    ys_reference = odeint_solution[-1, :]

    solution = ivpsolve.simulate_terminal_values(
        vector_field=vf,
        initial_values=u0,
        t0=t0,
        t1=t1,
        parameters=p,
        solver=solver,
    )

    assert solution.t == t1
    assert jnp.allclose(solution.u, ys_reference, atol=1e-3, rtol=1e-3)


@parametrize_with_cases("vf, u0, t0, t1, p", cases=".", prefix="problem_")
@parametrize_with_cases("solver", cases=".", prefix="solver_", has_tag=("checkpoint",))
def test_simulate_checkpoints(vf, u0, t0, t1, p, solver):
    ts = jnp.linspace(t0, t1, num=10)

    odeint_solution = odeint(vf, u0[0], ts, *p, atol=1e-6, rtol=1e-6)
    ts_reference, ys_reference = ts, odeint_solution

    solution = ivpsolve.simulate_checkpoints(
        vector_field=vf,
        initial_values=u0,
        ts=ts,
        parameters=p,
        solver=solver,
    )
    assert jnp.allclose(solution.t, ts_reference)
    assert jnp.allclose(solution.u, ys_reference, atol=1e-3, rtol=1e-3)
