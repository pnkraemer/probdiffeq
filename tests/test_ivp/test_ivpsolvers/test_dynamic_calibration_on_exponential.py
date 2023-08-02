"""Assert that the dynamic solver can successfully solve a linear function.

Specifically, we solve a linear function with exponentially increasing output-scale.
This is difficult for the MLE- and calibration-free solver,
but not for the dynamic solver.
"""
import jax
import jax.numpy as jnp

from probdiffeq import ivpsolve, test_util
from probdiffeq.backend import testing
from probdiffeq.ivpsolvers import calibrated
from probdiffeq.statespace import recipes


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
    return recipes.ts1_dense


@testing.fixture(name="problem")
def fixture_problem():
    @jax.jit
    def vf(x, *, t, p):  # pylint: disable=unused-argument
        return p * x

    p = 2.0
    t0, t1 = 0.0, 2.0
    return vf, jnp.ones((1,)), (t0, t1), p, lambda x: jnp.exp(p * x)


@testing.fixture(name="dynamic_solution_approximation_error")
@testing.parametrize_with_cases("impl_factory", cases=".", prefix="case_")
def fixture_approximation_error_low(problem, impl_factory):
    vf, u0, (t0, t1), f_args, solution = problem
    problem_args = (vf, (u0,))

    solver = test_util.generate_solver(
        # Problem setup chosen that when combined with ts1_dense,
        # only the dynamic solver can pass the test.
        # For some reason, zeroth-order solvers always do well.
        solver_factory=calibrated.dynamic,
        num_derivatives=1,
        impl_factory=impl_factory,
        ode_shape=(1,),
    )
    grid = jnp.linspace(t0, t1, num=20)
    solver_kwargs = {"grid": grid, "parameters": f_args, "solver": solver}
    approximation = ivpsolve.solve_fixed_grid(*problem_args, **solver_kwargs)

    return _rmse(approximation.u[-1], solution(t1))


def _rmse(a, b):
    return jnp.linalg.norm((a - b) / b) / jnp.sqrt(b.size)


def test_exponential_approximated_well(dynamic_solution_approximation_error):
    assert dynamic_solution_approximation_error < 0.1
