"""Tests for IVP solvers."""


import jax.numpy as jnp
import pytest_cases

from odefilter import ivpsolve, ivpsolvers, markov, problems


@pytest_cases.case
def problem_logistic():
    return problems.InitialValueProblem(
        f=lambda x: x * (1 - x),
        y0=0.5,
        t0=0.0,
        t1=10.0,
        p=(),
    )


@pytest_cases.case
def solver_ek0():
    return ivpsolvers.ek0(
        num_derivatives=2,
        step_control=ivpsolvers.pi_control(atol=1e-5, rtol=1e-7, error_order=3),
    )


@pytest_cases.parametrize_with_cases("problem", cases=".", prefix="problem_")
@pytest_cases.parametrize_with_cases("solver", cases=".", prefix="solver_")
def test_simulate_terminal_values(problem, solver):
    solver_alg, solver_params = solver
    solution = ivpsolve.simulate_terminal_values(
        f=problem.f,
        t0=problem.t0,
        t1=problem.t1,
        u0=problem.y0,
        solver=solver_alg,
        solver_params=solver_params,
    )
    assert solution.t == problem.t1

    mean, _ = solution.u
    assert jnp.allclose(mean[0], 1.0, atol=1e-3, rtol=1e-5)


# solver, solver_state = ek0()
# ivpsolve(f, u0, t0, t1, f_args, solver=solver, solver_params=solver_state)
#
#
#
#
# ivpsolve(f, u0, t0, t1, f_args, solver=ek0(init=taylor_mode()))
# ivpsolve(f, u0, t0, t1, f_args, solver=ek0())
#
# solver, state = ek0()
# solver, state = ek0(init=taylor_mode())
# solver.init == _TaylorMode
# state.init == _TaylorModeParams

#
# solver, params = ek0(step_selection=adaptive(step_controller=proportional_integral()))
# solver.step_selection.step_controller == _ProportionalIntegral
# params.step_selection.step_controller == _ProportionalIntegralParams
#
