"""The fixedpoint-smoother and smoother should yield identical results.

That is, when called with correct adaptive- and checkpoint-setups.
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

    return iso_factory, 1.0


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
    def vf(x, *, t, p):  # pylint: disable=unused-argument
        return f(x, *p)

    return vf, jnp.atleast_1d(u0), (t0, t1), f_args


@testing.fixture(name="solver_setup")
@testing.parametrize_with_cases("factorisation", cases=".", prefix="case_")
# Run with clipped and non-clipped controllers to cover both interpolation and clipping
# around the end(s) of time-intervals.
def fixture_solver_setup(problem, factorisation):
    vf, u0, (t0, t1), f_args = problem

    impl_factory, output_scale = factorisation
    args = (vf, (u0,))
    kwargs = {
        "parameters": f_args,
        "atol": 1e-3,
        "rtol": 1e-3,
        "output_scale": output_scale,
    }

    def impl_factory_wrapped():
        return impl_factory(ode_shape=jnp.shape(u0), num_derivatives=2)

    return args, kwargs, (t0, t1), impl_factory_wrapped


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

    # Re-generate the smoothing solver and compute the offgrid-marginals
    solver_smoother = test_util.generate_solver(
        strategy_factory=smoothers.smoother, impl_factory=impl_factory
    )
    ts = jnp.linspace(save_at[0], save_at[-1], num=7, endpoint=True)
    u_interp, marginals_interp = solution.offgrid_marginals_searchsorted(
        ts=ts[1:-1], solution=solution_smoother, solver=solver_smoother
    )

    # Generate a fixedpoint solver and solve (saving at the interpolation points)
    solver_fixedpoint = test_util.generate_solver(
        strategy_factory=smoothers.smoother_fixedpoint, impl_factory=impl_factory
    )
    solution_fixedpoint = ivpsolve.solve_and_save_at(
        *args, save_at=ts, solver=solver_fixedpoint, **kwargs
    )

    # Extract the interior points of the save_at solution
    # (because only there is the interpolated solution defined)
    u_fixedpoint = solution_fixedpoint.u[1:-1]
    marginals_fixedpoint = jax.tree_util.tree_map(
        lambda s: s[1:-1], solution_fixedpoint.marginals
    )

    # Compare QOI and marginals
    assert testing.tree_all_allclose(u_fixedpoint, u_interp)
    assert testing.marginals_allclose(marginals_fixedpoint, marginals_interp)
