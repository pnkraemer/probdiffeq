"""Compare simulate_terminal_values to solve_and_save_every_step."""

import jax
import jax.numpy as jnp

from probdiffeq import ivpsolve
from probdiffeq.backend import testing
from probdiffeq.impl import impl
from probdiffeq.solvers import calibrated
from probdiffeq.solvers.strategies import correction, extrapolation
from probdiffeq.solvers.strategies import filters
from probdiffeq.solvers.taylor import autodiff
from tests.setup import setup


@testing.fixture(name="problem_args_kwargs")
def fixture_problem_args_kwargs():
    vf, u0, (t0, t1) = setup.ode()
    tcoeffs = autodiff.taylor_mode(lambda y: vf(y, t=t0), u0, num=2)

    return (vf, tcoeffs), {"t0": t0, "t1": t1}


@testing.fixture(name="solver_kwargs")
def fixture_solver_kwargs():
    # Generate a solver
    ibm = extrapolation.ibm_adaptive(num_derivatives=2)
    ts0 = correction.taylor_order_zero()
    strategy = filters.filter_adaptive(ibm, ts0)
    solver = calibrated.mle(strategy)

    output_scale = jnp.ones_like(impl.prototypes.output_scale())
    return {
        "solver": solver,
        "dt0": 0.1,
        "output_scale": output_scale,
        "atol": 1e-2,
        "rtol": 1e-2,
    }


@testing.fixture(name="solution_python_loop")
def fixture_solution_with_python_while_loop(problem_args_kwargs, solver_kwargs):
    args, kwargs = problem_args_kwargs
    return ivpsolve.solve_and_save_every_step(*args, **kwargs, **solver_kwargs)


@testing.fixture(name="simulation_terminal_values")
def fixture_simulation_terminal_values(problem_args_kwargs, solver_kwargs):
    args, kwargs = problem_args_kwargs
    return ivpsolve.simulate_terminal_values(*args, **kwargs, **solver_kwargs)


def test_terminal_values_identical(solution_python_loop, simulation_terminal_values):
    """The terminal values must be identical."""
    expected = jax.tree_util.tree_map(lambda s: s[-1], solution_python_loop)
    received = simulation_terminal_values
    assert testing.tree_all_allclose(received, expected)
