"""The fixedpoint-smoother and smoother should yield identical results.

That is, at least in certain configurations.
"""
import diffeqzoo.ivps
import jax
import jax.numpy as jnp

from probdiffeq import ivpsolve, solution, test_util
from probdiffeq.backend import testing
from probdiffeq.strategies import smoothers


@testing.fixture(name="problem")
def fixture_problem():
    f, u0, (t0, t1), f_args = diffeqzoo.ivps.lotka_volterra()
    t1 = 4.0  # smaller time-span to decrease runtime

    @jax.jit
    def vf(x, *, t, p):  # pylint: disable=unused-argument
        return f(x, *p)

    return vf, jnp.atleast_1d(u0), (t0, t1), f_args


@testing.fixture(name="solver_setup")
def fixture_solver_setup(problem):
    vf, u0, (t0, t1), f_args = problem

    problem_args = (vf, (u0,))
    problem_kwargs = {"parameters": f_args, "rtol": 1e-2}
    return problem_args, problem_kwargs, (t0, t1)


@testing.fixture(name="solution_smoother")
def fixture_solution_smoother(solver_setup):
    args, kwargs, (t0, t1) = solver_setup
    solver = test_util.generate_solver(
        strategy_factory=smoothers.smoother, num_derivatives=2
    )
    return ivpsolve.solve_with_python_while_loop(
        *args, t0=t0, t1=t1, solver=solver, **kwargs
    )


def test_fixedpoint_smoother_equivalent_same_grid(solver_setup, solution_smoother):
    save_at = solution_smoother.t
    args, kwargs, _ = solver_setup
    solver = test_util.generate_solver(
        strategy_factory=smoothers.smoother_fixedpoint, num_derivatives=2
    )
    solution_fixedpoint = ivpsolve.solve_and_save_at(
        *args, save_at=save_at, solver=solver, **kwargs
    )
    assert testing.tree_all_allclose(solution_fixedpoint, solution_smoother)


def test_fixedpoint_smoother_equivalent_different_grid(solver_setup, solution_smoother):
    save_at = solution_smoother.t
    solver_smoother = test_util.generate_solver(
        strategy_factory=smoothers.smoother, num_derivatives=2
    )
    ts = jnp.linspace(save_at[0], save_at[-1], num=17, endpoint=True)
    u_interp, marginals_interp = solution.offgrid_marginals_searchsorted(
        ts=ts[1:-1], solution=solution_smoother, solver=solver_smoother
    )

    args, kwargs, _ = solver_setup
    solver_fixedpoint = test_util.generate_solver(
        strategy_factory=smoothers.smoother_fixedpoint, num_derivatives=2
    )
    solution_fixedpoint = ivpsolve.solve_and_save_at(
        *args, save_at=ts, solver=solver_fixedpoint, **kwargs
    )
    solution_fixedpoint = solution_fixedpoint[1:-1]

    assert testing.tree_all_allclose(solution_fixedpoint.u, u_interp)
    assert testing.marginals_allclose(marginals_interp, solution_fixedpoint.marginals)
