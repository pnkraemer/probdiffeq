"""Assert that the dynamic solver can successfully solve a linear function.

Specifically, we solve a linear function with exponentially increasing output-scale.
This is difficult for the MLE- and calibration-free solver,
but not for the dynamic solver.
"""

from probdiffeq import ivpsolve, probdiffeq
from probdiffeq.backend import functools, linalg, ode, testing, tree_util
from probdiffeq.backend import numpy as np


@testing.parametrize("fact", ["dense", "isotropic", "blockdiag"])
def test_exponential_approximated_well(fact):
    vf, u0, (t0, t1) = ode.ivp_lotka_volterra()

    tcoeffs = (*u0, vf(*u0, t=t0))
    init, ibm, ssm = probdiffeq.prior_wiener_integrated(tcoeffs, ssm_fact=fact)
    ts0 = probdiffeq.constraint_ode_ts0(ssm=ssm)
    strategy = probdiffeq.strategy_filter(ssm=ssm)
    solver = probdiffeq.solver_dynamic(
        vf, strategy=strategy, prior=ibm, constraint=ts0, ssm=ssm
    )

    grid = np.linspace(t0, t1, num=20)
    solve = ivpsolve.solve_fixed_grid(solver=solver)
    approximation = functools.jit(solve)(init, grid=grid)

    solution = ode.odeint_and_save_at(
        vf, u0, save_at=np.asarray([t0, t1]), atol=1e-5, rtol=1e-5
    )
    vmap_ravel = functools.vmap(lambda s: tree_util.ravel_pytree(s)[0])
    u = vmap_ravel(approximation.u.mean[0])
    sol = vmap_ravel(solution)
    rmse = _rmse(u[-1], sol[-1])
    assert rmse < 0.1


def _rmse(a, b):
    return linalg.vector_norm((a - b) / b) / np.sqrt(b.size)
