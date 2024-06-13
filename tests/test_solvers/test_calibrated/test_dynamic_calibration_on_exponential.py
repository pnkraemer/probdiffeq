"""Assert that the dynamic solver can successfully solve a linear function.

Specifically, we solve a linear function with exponentially increasing output-scale.
This is difficult for the MLE- and calibration-free solver,
but not for the dynamic solver.
"""

from probdiffeq import ivpsolve
from probdiffeq.backend import linalg
from probdiffeq.backend import numpy as np
from probdiffeq.impl import impl
from probdiffeq.solvers import components, solvers, strategies
from tests.setup import setup


def test_exponential_approximated_well():
    vf, u0, (t0, t1), solution = setup.ode_affine()

    ibm = components.ibm_adaptive(num_derivatives=1)
    ts0 = components.correction_ts0()
    strategy = strategies.filter_adaptive(ibm, ts0)
    solver = solvers.dynamic(strategy)

    output_scale = np.ones_like(impl.prototypes.output_scale())
    init = solver.initial_condition((*u0, vf(*u0, t=t0)), output_scale=output_scale)

    problem_args = (vf, init)
    grid = np.linspace(t0, t1, num=20)
    solver_kwargs = {"grid": grid, "solver": solver}
    approximation = ivpsolve.solve_fixed_grid(*problem_args, **solver_kwargs)

    rmse = _rmse(approximation.u[-1], solution(t1))
    assert rmse < 0.1


def _rmse(a, b):
    return linalg.vector_norm((a - b) / b) / np.sqrt(b.size)
