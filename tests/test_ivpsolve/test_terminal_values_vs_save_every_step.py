"""Compare simulate_terminal_values to solve_adaptive_save_every_step."""

from probdiffeq import ivpsolve, ivpsolvers, taylor
from probdiffeq.backend import numpy as np
from probdiffeq.backend import testing, tree_util
from probdiffeq.impl import impl


@testing.fixture(name="problem_args_kwargs")
def fixture_problem_args_kwargs(ssm):
    vf, u0, (t0, t1) = ssm.default_ode

    # Generate a solver
    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), u0, num=2)
    output_scale = np.ones_like(impl.prototypes.output_scale())
    ibm = ivpsolvers.prior_ibm(tcoeffs, output_scale=output_scale)

    ts0 = ivpsolvers.correction_ts0()
    strategy = ivpsolvers.strategy_filter(ibm, ts0)
    solver = ivpsolvers.solver_mle(strategy)
    adaptive_solver = ivpsolve.adaptive(solver, atol=1e-2, rtol=1e-2)

    init = solver.initial_condition()

    args = (vf, init)
    kwargs = {"t0": t0, "t1": t1, "adaptive_solver": adaptive_solver, "dt0": 0.1}
    return args, kwargs


@testing.fixture(name="solution_python_loop")
def fixture_solution_with_python_while_loop(problem_args_kwargs):
    args, kwargs = problem_args_kwargs

    return ivpsolve.solve_adaptive_save_every_step(*args, **kwargs)


@testing.fixture(name="simulation_terminal_values")
def fixture_simulation_terminal_values(problem_args_kwargs):
    args, kwargs = problem_args_kwargs
    return ivpsolve.solve_adaptive_terminal_values(*args, **kwargs)


def test_terminal_values_identical(solution_python_loop, simulation_terminal_values):
    """The terminal values must be identical."""
    expected = tree_util.tree_map(lambda s: s[-1], solution_python_loop)
    received = simulation_terminal_values
    assert testing.tree_all_allclose(received, expected)
