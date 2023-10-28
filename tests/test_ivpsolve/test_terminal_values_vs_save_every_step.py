"""Compare simulate_terminal_values to solve_and_save_every_step."""

import jax
import jax.numpy as jnp

from probdiffeq import adaptive, ivpsolve
from probdiffeq.backend import testing
from probdiffeq.impl import impl
from probdiffeq.solvers import calibrated
from probdiffeq.solvers.strategies import filters
from probdiffeq.solvers.strategies.components import corrections, priors
from probdiffeq.taylor import autodiff
from tests.setup import setup


@testing.fixture(name="problem_args_kwargs")
def fixture_problem_args_kwargs():
    vf, u0, (t0, t1) = setup.ode()

    # Generate a solver
    ibm = priors.ibm_adaptive(num_derivatives=2)
    ts0 = corrections.ts0()
    strategy = filters.filter_adaptive(ibm, ts0)
    solver = calibrated.mle(strategy)
    adaptive_solver = adaptive.adaptive(solver, atol=1e-2, rtol=1e-2)

    tcoeffs = autodiff.taylor_mode_scan(lambda y: vf(y, t=t0), u0, num=2)
    output_scale = jnp.ones_like(impl.prototypes.output_scale())
    init = solver.initial_condition(tcoeffs, output_scale=output_scale)

    args = (vf, init)
    kwargs = {"t0": t0, "t1": t1, "adaptive_solver": adaptive_solver, "dt0": 0.1}
    return args, kwargs


@testing.fixture(name="solution_python_loop")
def fixture_solution_with_python_while_loop(problem_args_kwargs):
    args, kwargs = problem_args_kwargs

    return ivpsolve.solve_and_save_every_step(*args, **kwargs)


@testing.fixture(name="simulation_terminal_values")
def fixture_simulation_terminal_values(problem_args_kwargs):
    args, kwargs = problem_args_kwargs
    return ivpsolve.simulate_terminal_values(*args, **kwargs)


def test_terminal_values_identical(solution_python_loop, simulation_terminal_values):
    """The terminal values must be identical."""
    expected = jax.tree_util.tree_map(lambda s: s[-1], solution_python_loop)
    received = simulation_terminal_values
    assert testing.tree_all_allclose(received, expected)
