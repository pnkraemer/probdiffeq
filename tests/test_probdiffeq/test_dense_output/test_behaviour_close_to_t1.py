"""Test that filtering/smoothing close to t=t1 is correct."""

from probdiffeq import ivpsolve, probdiffeq
from probdiffeq.backend import func, np, ode, structs, testing, tree
from probdiffeq.util import test_util


@testing.case
def case_ssm_dense():
    """Construct a dense SSM."""
    return probdiffeq.state_space_model_dense()


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


@testing.case
def case_solver_standardsolver():
    """Construct a setup for solvers."""
    return probdiffeq.solver


@testing.case
def case_solver_mle():
    """Construct a setup for solvers."""
    return probdiffeq.solver_mle


@testing.case
def case_solver_dynamic():
    """Construct a setup for solvers."""
    return probdiffeq.solver_dynamic


@testing.parametrize_with_cases("ssm", cases=".", prefix="case_ssm_")
@testing.parametrize_with_cases("strategy", cases=".", prefix="case_strategy_")
@testing.parametrize_with_cases("solver_factory", cases=".", prefix="case_solver_")
@testing.parametrize("tol, num", [(1e-1, 30)])  # hand-tuned
def test_behaviour_at_t1_is_correct(ssm, strategy, solver_factory, tol, num) -> None:
    """Test that filtering/smoothing close to t=t1 is correct."""
    vf, u0, (t0, t1) = ode.ivp_lotka_volterra()

    # Generate a solver
    vf = probdiffeq.ode(vf)
    jetexpand = probdiffeq.jetexpand_ode_padded_scan(num=2)
    tcoeffs, _ = jetexpand(vf, u0, t=t0)
    iwp = ssm.prior_wiener_integrated(tcoeffs)
    ts0 = ssm.constraint_ode_ts1(vf)
    error = probdiffeq.error_residual_std(constraint=ts0)

    # Get the computation-time-grid
    solver = solver_factory(strategy=strategy.save_every_step, constraint=ts0)
    solve = test_util.solve_adaptive_save_every_step(error=error, solver=solver)
    solution_every = solve(iwp, atol=tol, rtol=tol, t0=t0, t1=t1)

    # Generate a save-at solver
    solver = solver_factory(strategy=strategy.save_at, constraint=ts0)
    solve = func.jit(ivpsolve.solve_adaptive_save_at(error=error, solver=solver))

    # Solve 2x: once on 'ts', once on 'ts' except the last T points
    ts = np.linspace(solution_every.t[0], solution_every.t[-2], endpoint=True, num=num)
    T = 1
    solution_full = solve(iwp, atol=tol, rtol=tol, save_at=ts)
    solution_cut = solve(iwp, atol=tol, rtol=tol, save_at=ts[:-T])

    # Now, the cut solution should be identical to the full solution except the last T points
    # For the filter, this is relatively trivial. For the smoother, this check verifies
    # that the marginals close to t1 are actually informed by the "overstepped" state.
    # Recall that the fwd-interpolation in smoothers does not use that info.
    (m1, C1) = solution_full.u.to_multivariate_normal()
    m1, C1 = tree.tree_map(lambda s: s[:-T], (m1, C1))
    (m2, C2) = solution_cut.u.to_multivariate_normal()
    assert testing.allclose(m1, m2)
    assert testing.allclose(C1, C2)
