"""Compare solve_fixed_grid to solve_adaptive_save_every_step."""

from probdiffeq import ivpsolve, probdiffeq
from probdiffeq.backend import func, ode, structs, testing, tree
from probdiffeq.backend.typing import Array
from probdiffeq.util import test_util


@testing.parametrize("fact", ["dense", "isotropic", "blockdiag"])
def test_fixed_grid_result_matches_adaptive_grid_result_when_reusing_grid(fact) -> None:
    vf, u0, (t0, t1) = ode.ivp_lotka_volterra()

    class Taylor(structs.NamedTuple):
        state: Array
        velocity: Array
        acceleration: Array

    vf = probdiffeq.ode(vf)
    jetexpand = probdiffeq.jetexpand_ode_padded_scan(num=2)
    coeffs, _ = jetexpand(vf, u0, t=t0)
    tcoeffs = Taylor(*coeffs)

    ssm = probdiffeq.state_space_model(ssm_fact=fact)
    init, iwp = probdiffeq.prior_wiener_integrated(tcoeffs, ssm=ssm)
    ts0 = probdiffeq.constraint_ode_ts0(vf, ssm=ssm)
    strategy = probdiffeq.strategy_filter()
    solver = probdiffeq.solver_mle(strategy=strategy, prior=iwp, constraint=ts0)
    error = probdiffeq.error_residual_std(constraint=ts0, prior=iwp)

    solve = test_util.solve_adaptive_save_every_step(
        error=error, solver=solver, clip_dt=True
    )
    solution_adaptive = solve(init, t0=t0, t1=t1, atol=1e-2, rtol=1e-2)
    assert isinstance(solution_adaptive.u.mean, Taylor)

    grid_adaptive = solution_adaptive.t
    solve = ivpsolve.solve_fixed_grid(solver=solver)
    solution_fixed = func.jit(solve)(init, grid=grid_adaptive)
    assert testing.allclose(solution_adaptive, solution_fixed)

    # Assert u and u_std have matching shapes (that was wrong before)
    _, u_shape = tree.tree_flatten_depth_one(solution_fixed.u.mean)
    _, u_std_shape = tree.tree_flatten_depth_one(solution_fixed.u.std)
    assert u_shape == u_std_shape
