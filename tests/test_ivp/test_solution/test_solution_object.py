"""Tests for IVP solvers."""
import diffeqzoo.ivps
import jax
import jax.numpy as jnp

from probdiffeq import ivpsolve, test_util
from probdiffeq.backend import testing
from probdiffeq.statespace import recipes


@testing.fixture(name="problem")
def fixture_problem():
    f, u0, (t0, _), f_args = diffeqzoo.ivps.lotka_volterra()
    t1 = 2.0  # Short time-intervals are sufficient for this test.

    @jax.jit
    def vf(x, *, t, p):  # pylint: disable=unused-argument
        return f(x, *p)

    return vf, u0, (t0, t1), f_args


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


@testing.fixture(name="approximate_solution")
@testing.parametrize_with_cases("impl_factory", cases=".", prefix="case_")
def fixture_approximate_solution(problem, impl_factory):
    vf, u0, (t0, t1), f_args = problem
    solver = test_util.generate_solver(
        num_derivatives=1, impl_factory=impl_factory, ode_shape=jnp.shape(u0)
    )
    sol = ivpsolve.solve_with_python_while_loop(
        vf,
        (u0,),
        t0=t0,
        t1=t1,
        parameters=f_args,
        solver=solver,
        atol=1e-2,
        rtol=1e-2,
    )
    return sol, solver


def test_marginal_nth_derivative_of_solution(approximate_solution):
    """Assert that each $n$th derivative matches the quantity of interest's shape."""
    sol, _ = approximate_solution

    # Assert that the marginals have the same shape as the qoi.
    for i in (0, 1):
        derivatives = sol.marginals.marginal_nth_derivative(i)
        assert derivatives.mean.shape == sol.u.shape

    # if the requested derivative is not in the state-space model, raise a ValueError
    with testing.raises(ValueError):
        sol.marginals.marginal_nth_derivative(100)
