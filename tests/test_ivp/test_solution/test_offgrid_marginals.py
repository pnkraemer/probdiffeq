"""Tests for IVP solvers."""
import diffeqzoo.ivps
import jax
import jax.numpy as jnp

from probdiffeq import ivpsolve
from probdiffeq.backend import testing
from probdiffeq.solvers import solution
from probdiffeq.solvers.statespace import recipes
from probdiffeq.solvers.strategies import filters, smoothers
from probdiffeq.util import test_util


@testing.fixture(name="problem")
def fixture_problem():
    f, u0, (t0, _), f_args = diffeqzoo.ivps.lotka_volterra()
    t1 = 2.0  # Short time-intervals are sufficient for this test.

    @jax.jit
    def vf(x, *, t, p):  # noqa: ARG001
        return f(x, *p)

    return vf, u0, (t0, t1), f_args


@testing.case()
def case_isotropic_factorisation():
    return recipes.ts0_iso, 2.0


@testing.case()  # this implies success of the scalar solver
def case_blockdiag_factorisation():
    return recipes.ts0_blockdiag, jnp.ones((2,)) * 2.0


@testing.case()
def case_dense_factorisation():
    return recipes.ts0_dense, 2.0


@testing.parametrize_with_cases("factorisation", cases=".", prefix="case_")
def test_filter_marginals_close_only_to_left_boundary(problem, factorisation):
    """Assert that the filter-marginals interpolate well close to the left boundary."""
    vf, u0, (t0, t1), f_args = problem
    impl_factory, output_scale = factorisation
    solver = test_util.generate_solver(
        num_derivatives=1,
        impl_factory=impl_factory,
        strategy_factory=filters.filter,
        ode_shape=jnp.shape(u0),
    )
    sol = ivpsolve.solve_with_python_while_loop(
        vf,
        (u0,),
        t0=t0,
        t1=t1,
        parameters=f_args,
        solver=solver,
        output_scale=output_scale,
        atol=1e-2,
        rtol=1e-2,
    )

    # Extrapolate from the left: close-to-left boundary must be similar,
    # but close-to-right boundary needs not be similar
    ts = jnp.linspace(t0 + 1e-4, t1 - 1e-4, num=4, endpoint=True)
    u, _ = solution.offgrid_marginals_searchsorted(ts=ts, solution=sol, solver=solver)
    assert jnp.allclose(u[0], sol.u[0], atol=1e-3, rtol=1e-3)
    assert not jnp.allclose(u[0], sol.u[1], atol=1e-3, rtol=1e-3)


@testing.parametrize_with_cases("factorisation", cases=".", prefix="case_")
def test_smoother_marginals_close_to_both_boundaries(problem, factorisation):
    """Assert that the smoother-marginals interpolate well close to the boundary."""
    vf, u0, (t0, t1), f_args = problem
    impl_factory, output_scale = factorisation
    solver = test_util.generate_solver(
        num_derivatives=1,
        impl_factory=impl_factory,
        strategy_factory=smoothers.smoother_adaptive,
        ode_shape=jnp.shape(u0),
    )
    sol = ivpsolve.solve_with_python_while_loop(
        vf,
        (u0,),
        t0=t0,
        t1=t1,
        parameters=f_args,
        solver=solver,
        output_scale=output_scale,
        atol=1e-2,
        rtol=1e-2,
    )
    # Extrapolate from the left: close-to-left boundary must be similar,
    # but close-to-right boundary must not be similar
    ts = jnp.linspace(t0 + 1e-4, t1 - 1e-4, num=4, endpoint=True)
    u, _ = solution.offgrid_marginals_searchsorted(ts=ts, solution=sol, solver=solver)
    assert jnp.allclose(u[0], sol.u[0], atol=1e-3, rtol=1e-3)
    assert jnp.allclose(u[-1], sol.u[-1], atol=1e-3, rtol=1e-3)
