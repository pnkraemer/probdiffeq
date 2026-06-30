"""Tests for off-grid marginal interpolation."""

from probdiffeq import ivpsolve, probdiffeq
from probdiffeq.backend import func, np, ode, structs, testing, tree
from probdiffeq.util import test_util


@testing.case
def case_ssm_dense():
    """Construct a dense SSM."""
    return probdiffeq.state_space_model_dense()


@testing.case
def case_ssm_blockdiag():
    """Construct a block-diagonal SSM."""
    return probdiffeq.state_space_model_blockdiag()


@testing.case
def case_ssm_isotropic():
    """Construct an isotropic SSM."""
    return probdiffeq.state_space_model_isotropic()


@structs.dataclass
class StrategySetup:
    """Test-config for offgrid marginal tests."""

    save_every_step: probdiffeq.MarkovStrategy
    save_at: probdiffeq.MarkovStrategy


@testing.case
def case_strategy_filter() -> StrategySetup:
    """Construct a setup for filtering."""
    filter_ = probdiffeq.strategy_filter()
    return StrategySetup(save_every_step=filter_, save_at=filter_)


@testing.case
def case_strategy_smoother() -> StrategySetup:
    """Construct a setup for smoothing."""
    fixedinterval = probdiffeq.strategy_smoother_fixedinterval()
    fixedpoint = probdiffeq.strategy_smoother_fixedpoint()
    return StrategySetup(save_every_step=fixedinterval, save_at=fixedpoint)


@testing.parametrize_with_cases("ssm", cases=".", prefix="case_ssm_")
@testing.parametrize_with_cases("strategy", cases=".", prefix="case_strategy_")
def test_save_at_result_matches_interpolated_adaptive_result(ssm, strategy) -> None:
    """Test that the save_at result matches the interpolation (using a filter)."""
    vf, u0, (t0, t1) = ode.ivp_lotka_volterra()

    # Generate a solver
    vf = probdiffeq.ode(vf)
    jetexpand = probdiffeq.jetexpand_ode_padded_scan(num=2)
    tcoeffs, _ = jetexpand(vf, u0, t=t0)
    iwp = ssm.prior_wiener_integrated(tcoeffs)
    ts0 = ssm.constraint_ode_ts0(vf)
    error = probdiffeq.error_residual_std(constraint=ts0)

    # Compute an adaptive solution and interpolate
    ts = np.linspace(t0, t1, num=15, endpoint=True)
    solver = probdiffeq.solver(strategy=strategy.save_every_step, constraint=ts0)
    solve = test_util.solve_adaptive_save_every_step(error=error, solver=solver)
    save_every = solve(iwp, t0=t0, t1=t1, dt0=0.1, atol=1e-2, rtol=1e-2)
    offgrid = func.vmap(lambda s: solver.offgrid_marginals(s, solution=save_every))
    u_interpolated = func.jit(offgrid)(ts[1:-1])

    # Compute a save-at solution and remove the edge-points (because offgrid_marginals can't compute them)
    solver = probdiffeq.solver(strategy=strategy.save_at, constraint=ts0)
    solve = ivpsolve.solve_adaptive_save_at(error=error, solver=solver)
    save_at = func.jit(solve)(iwp, atol=1e-2, rtol=1e-2, save_at=ts, dt0=0.1)
    u_save_at = tree.tree_map(lambda s: s[1:-1], save_at.u)

    # Assert similarity
    assert testing.allclose(u_interpolated.mean, u_save_at.mean)
    assert testing.allclose(u_interpolated.std, u_save_at.std)

    mv1 = u_interpolated.to_multivariate_normal()
    mv2 = u_save_at.to_multivariate_normal()
    assert testing.allclose(mv1, mv2)
