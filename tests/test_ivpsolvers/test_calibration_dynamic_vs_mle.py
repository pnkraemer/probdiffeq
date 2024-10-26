"""Assert that the dynamic solver can successfully solve a linear function.

Specifically, we solve a linear function with exponentially increasing output-scale.
This is difficult for the MLE- and calibration-free solver,
but not for the dynamic solver.
"""

from probdiffeq import ivpsolve, ivpsolvers
from probdiffeq.backend import linalg, ode, testing
from probdiffeq.backend import numpy as np


@testing.parametrize("fact", ["dense", "isotropic", "blockdiag"])
def test_exponential_approximated_well(fact):
    vf, u0, (t0, t1) = ode.ivp_lotka_volterra()

    ibm, ssm = ivpsolvers.prior_ibm((*u0, vf(*u0, t=t0)), ssm_fact=fact)
    ts0 = ivpsolvers.correction_ts0(ssm=ssm)
    strategy = ivpsolvers.strategy_filter(ssm=ssm)
    solver = ivpsolvers.solver_dynamic(strategy, prior=ibm, correction=ts0, ssm=ssm)

    init = solver.initial_condition()

    problem_args = (vf, init)
    grid = np.linspace(t0, t1, num=20)
    solver_kwargs = {"grid": grid, "solver": solver, "ssm": ssm}
    approximation = ivpsolve.solve_fixed_grid(*problem_args, **solver_kwargs)

    solution = ode.odeint_dense(vf, u0, t0=t0, t1=t1, atol=1e-5, rtol=1e-5)
    rmse = _rmse(approximation.u[0][-1], solution(t1))
    assert rmse < 0.1


def _rmse(a, b):
    return linalg.vector_norm((a - b) / b) / np.sqrt(b.size)
