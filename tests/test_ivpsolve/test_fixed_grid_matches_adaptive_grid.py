"""Compare solve_fixed_grid to solve_adaptive_save_every_step."""

from probdiffeq import ivpsolve, probdiffeq, taylor
from probdiffeq.backend import containers, functools, ode, testing, tree_util
from probdiffeq.backend import numpy as np
from probdiffeq.backend.typing import Array
from probdiffeq.util import test_util


@testing.parametrize("fact", ["dense", "isotropic", "blockdiag"])
def test_fixed_grid_result_matches_adaptive_grid_result_when_reusing_grid(fact):
    vf, u0, (t0, t1) = ode.ivp_lotka_volterra()

    class Taylor(containers.NamedTuple):
        state: Array
        velocity: Array
        acceleration: Array

    tcoeffs = Taylor(*taylor.odejet_padded_scan(lambda y: vf(y, t=t0), u0, num=2))

    init, ibm, ssm = probdiffeq.prior_wiener_integrated(tcoeffs, ssm_fact=fact)
    ts0 = probdiffeq.constraint_ode_ts0(vf, ssm=ssm)
    strategy = probdiffeq.strategy_filter(ssm=ssm)
    solver = probdiffeq.solver_mle(strategy, prior=ibm, correction=ts0, ssm=ssm)
    errorest = probdiffeq.errorest_local_residual(prior=ibm, correction=ts0, ssm=ssm)

    solve = test_util.solve_adaptive_save_every_step(
        errorest=errorest, solver=solver, clip_dt=True
    )
    solution_adaptive = solve(init, t0=t0, t1=t1, atol=1e-2, rtol=1e-2)
    assert isinstance(solution_adaptive.u.mean, Taylor)

    grid_adaptive = solution_adaptive.t
    solve = ivpsolve.solve_fixed_grid(solver=solver)
    solution_fixed = functools.jit(solve)(init, grid=grid_adaptive)
    assert testing.allclose(solution_adaptive, solution_fixed)

    # Assert u and u_std have matching shapes (that was wrong before)
    u_shape = tree_util.tree_map(np.shape, solution_fixed.u.mean)
    u_std_shape = tree_util.tree_map(np.shape, solution_fixed.u.std)
    match = tree_util.tree_map(lambda a, b: a == b, u_shape, u_std_shape)
    assert tree_util.tree_all(match)
