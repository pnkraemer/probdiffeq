"""Test for scalar recipes."""

import diffeqzoo.ivps
import diffrax
import jax
import jax.numpy as jnp

from probdiffeq import ivpsolve, test_util
from probdiffeq.backend import testing
from probdiffeq.statespace import recipes
from probdiffeq.strategies import filters, smoothers


@testing.fixture(name="problem")
def fixture_problem():
    f, u0, (t0, t1), f_args = diffeqzoo.ivps.logistic()

    @jax.jit
    def vf(x, *, t, p):  # pylint: disable=unused-argument
        return f(x, *p)

    return vf, u0, (t0, t1), f_args


@testing.case()
def case_filter():
    return filters.filter


@testing.case()
def case_smoother():
    return smoothers.smoother


@testing.case()
def case_fixedpoint():
    return smoothers.smoother_fixedpoint


@testing.fixture(name="solution")
@testing.parametrize_with_cases("strategy", cases=".", prefix="case_")
def fixture_recipe_solution(problem, strategy):
    vf, u0, (t0, t1), f_args = problem

    problem_args = (vf, (u0,))
    problem_kwargs = {"t0": t0, "t1": t1, "parameters": f_args}
    impl_factory, output_scale = recipes.ts0_scalar, 1.0
    solver = test_util.generate_solver(
        num_derivatives=2, impl_factory=impl_factory, strategy_factory=strategy
    )
    adaptive_kwargs = {
        "solver": solver,
        "atol": 1e-2,
        "rtol": 1e-2,
        "output_scale": output_scale,
    }
    return ivpsolve.simulate_terminal_values(
        *problem_args, **problem_kwargs, **adaptive_kwargs
    )


@testing.fixture(name="diffrax_solution")
def fixture_diffrax_solution(problem):
    vf, u0, (t0, t1), f_args = problem

    # Solve the IVP
    @jax.jit
    def vf_diffrax(t, y, args):
        return vf(y, t=t, p=args)

    term = diffrax.ODETerm(vf_diffrax)
    solver = diffrax.Dopri5()
    solution_object = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=0.1,
        y0=u0,
        args=f_args,
        saveat=diffrax.SaveAt(dense=True),
        stepsize_controller=diffrax.PIDController(atol=1e-10, rtol=1e-10),
    )

    def solution(t):
        return solution_object.evaluate(t)

    return solution


def test_terminal_value_simulation_matches_diffrax(solution, diffrax_solution):
    expected = diffrax_solution(solution.t)
    received = solution.u

    assert jnp.allclose(received, expected, rtol=1e-2)
