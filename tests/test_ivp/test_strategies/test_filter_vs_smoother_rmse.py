"""The RMSE of the smoother should be (slightly) lower than the RMSE of the filter."""
import diffeqzoo.ivps
import diffrax
import jax
import jax.numpy as jnp

from probdiffeq import ivpsolve
from probdiffeq.backend import testing
from probdiffeq.solvers.statespace import recipes
from probdiffeq.solvers.strategies import filters, smoothers
from probdiffeq.util import test_util


@testing.case()
def case_isotropic_factorisation():
    return recipes.ts0_iso, 1.0


@testing.case()  # this implies success of the scalar solver
def case_blockdiag_factorisation():
    return recipes.ts0_blockdiag, jnp.ones((2,))


@testing.case()
def case_dense_factorisation():
    return recipes.ts0_dense, 1.0


@testing.fixture(name="problem")
def fixture_problem():
    f, u0, (t0, t1), f_args = diffeqzoo.ivps.lotka_volterra()
    t1 = 4.0  # smaller time-span to decrease runtime

    @jax.jit
    def vf(x, *, t, p):  # noqa: ARG001
        return f(x, *p)

    return vf, jnp.atleast_1d(u0), (t0, t1), f_args


@testing.fixture(name="solver_setup")
@testing.parametrize_with_cases("factorisation", cases=".", prefix="case_")
def fixture_solver_setup(problem, factorisation):
    vf, u0, (t0, t1), f_args = problem

    impl_factory, output_scale = factorisation
    grid = jnp.linspace(t0, t1, endpoint=True, num=10)
    problem_args = (vf, (u0,))
    problem_kwargs = {"grid": grid, "output_scale": output_scale, "parameters": f_args}

    def impl_factory_wrapped():
        return impl_factory(ode_shape=jnp.shape(u0), num_derivatives=4)

    return problem_args, problem_kwargs, impl_factory_wrapped


@testing.fixture(name="filter_solution")
def fixture_filter_solution(solver_setup):
    args, kwargs, impl_factory = solver_setup
    solver = test_util.generate_solver(
        strategy_factory=filters.filter, impl_factory=impl_factory
    )
    return ivpsolve.solve_fixed_grid(*args, solver=solver, **kwargs)


@testing.fixture(name="smoother_solution")
def fixture_smoother_solution(solver_setup):
    args, kwargs, impl_factory = solver_setup
    solver = test_util.generate_solver(
        strategy_factory=smoothers.smoother, impl_factory=impl_factory
    )
    return ivpsolve.solve_fixed_grid(*args, solver=solver, **kwargs)


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

    @jax.vmap
    def solution(t):
        return solution_object.evaluate(t)

    return solution


def test_compare_filter_smoother_rmse(
    filter_solution, smoother_solution, diffrax_solution
):
    assert jnp.allclose(filter_solution.t, smoother_solution.t)  # sanity check

    reference = diffrax_solution(filter_solution.t)
    filter_rmse = _rmse(filter_solution.u, reference)
    smoother_rmse = _rmse(smoother_solution.u, reference)

    # at least 10 percent difference to increase significance
    assert 1.1 * smoother_rmse < filter_rmse

    # The error should be small, otherwise the test makes little sense
    assert filter_rmse < 0.01


def _rmse(a, b):
    return jnp.linalg.norm((a - b) / b) / jnp.sqrt(b.size)
