"""The fixedpoint-smoother and smoother should yield identical results.

That is, at least in certain configurations.
"""
import diffeqzoo.ivps
import jax
import jax.numpy as jnp

from probdiffeq import ivpsolve, solution, test_util
from probdiffeq.backend import testing
from probdiffeq.statespace import recipes
from probdiffeq.strategies import smoothers


@testing.case()
def case_isotropic_factorisation():
    def iso_factory(ode_shape, num_derivatives):
        return recipes.ts0_iso(num_derivatives=num_derivatives)

    return iso_factory


@testing.case()  # this implies success of the scalar solver
def case_blockdiag_factorisation():
    return recipes.ts0_blockdiag


@testing.case()
def case_dense_factorisation():
    return recipes.ts0_dense


@testing.fixture(name="problem")
def fixture_problem():
    f, u0, (t0, t1), f_args = diffeqzoo.ivps.lotka_volterra()
    t1 = 4.0  # smaller time-span to decrease runtime

    @jax.jit
    def vf(x, *, t, p):  # pylint: disable=unused-argument
        return f(x, *p)

    return vf, jnp.atleast_1d(u0), (t0, t1), f_args


@testing.fixture(name="solver_setup")
@testing.parametrize_with_cases("impl_factory", cases=".", prefix="case_")
def fixture_solver_setup(problem, impl_factory):
    vf, u0, (t0, t1), f_args = problem

    problem_args = (vf, (u0,))
    problem_kwargs = {"parameters": f_args, "rtol": 1e-2}

    def impl_factory_wrapped():
        return impl_factory(ode_shape=jnp.shape(u0), num_derivatives=2)

    return problem_args, problem_kwargs, (t0, t1), impl_factory_wrapped


@testing.fixture(name="solution_smoother")
def fixture_solution_smoother(solver_setup):
    args, kwargs, (t0, t1), impl_factory = solver_setup
    solver = test_util.generate_solver(
        strategy_factory=smoothers.smoother, impl_factory=impl_factory
    )
    return ivpsolve.solve_with_python_while_loop(
        *args, t0=t0, t1=t1, solver=solver, **kwargs
    )


def test_fixedpoint_smoother_equivalent_same_grid(solver_setup, solution_smoother):
    save_at = solution_smoother.t
    args, kwargs, _, impl_factory = solver_setup
    solver = test_util.generate_solver(
        strategy_factory=smoothers.smoother_fixedpoint, impl_factory=impl_factory
    )
    solution_fixedpoint = ivpsolve.solve_and_save_at(
        *args, save_at=save_at, solver=solver, **kwargs
    )
    assert testing.tree_all_allclose(solution_fixedpoint, solution_smoother)


def test_fixedpoint_smoother_equivalent_different_grid(solver_setup, solution_smoother):
    args, kwargs, _, impl_factory = solver_setup
    save_at = solution_smoother.t
    solver_smoother = test_util.generate_solver(
        strategy_factory=smoothers.smoother, impl_factory=impl_factory
    )
    ts = jnp.linspace(save_at[0], save_at[-1], num=17, endpoint=True)
    u_interp, marginals_interp = solution.offgrid_marginals_searchsorted(
        ts=ts[1:-1], solution=solution_smoother, solver=solver_smoother
    )

    solver_fixedpoint = test_util.generate_solver(
        strategy_factory=smoothers.smoother_fixedpoint, impl_factory=impl_factory
    )
    solution_fixedpoint = ivpsolve.solve_and_save_at(
        *args, save_at=ts, solver=solver_fixedpoint, **kwargs
    )
    solution_fixedpoint = solution_fixedpoint[1:-1]

    assert testing.tree_all_allclose(solution_fixedpoint.u, u_interp)
    assert testing.marginals_allclose(marginals_interp, solution_fixedpoint.marginals)
