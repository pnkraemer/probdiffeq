"""Tests for filter interpolation."""

from probdiffeq import ivpsolve, probdiffeq
from probdiffeq.backend import func, np, ode, testing, tree


@testing.case
def case_ssm_dense():
    """Construct a dense SSM."""
    return probdiffeq.state_space_model_dense()


@testing.parametrize_with_cases("ssm", cases=".", prefix="case_ssm_")
def test_filter_highest_derivative_extrapolates_constant(ssm) -> None:
    """Test that the save_at result matches the interpolation (using a filter)."""
    vf, u0, (t0, t1) = ode.ivp_lotka_volterra()

    # Generate a solver
    vf = probdiffeq.ode(vf)
    jetexpand = probdiffeq.jetexpand_ode_padded_scan(num=2)
    tcoeffs, _ = jetexpand(vf, u0, t=t0)
    iwp = ssm.prior_wiener_integrated(tcoeffs)
    ts0 = ssm.constraint_ode_ts0(vf)
    error = probdiffeq.error_residual_std(constraint=ts0)

    # Compute a save-at solution
    ts = np.linspace(t0, t1 / 2.0, num=75, endpoint=True)
    strategy = probdiffeq.strategy_filter()
    solver = probdiffeq.solver(strategy=strategy, constraint=ts0)
    solve = ivpsolve.solve_adaptive_save_at(error=error, solver=solver)
    solution = func.jit(solve)(iwp, atol=1e-1, rtol=1e-1, save_at=ts)

    # Flatten the pytree into an (T, d) array
    fx = func.vmap(lambda s: tree.ravel_pytree(s)[0])(solution.u.mean[-1])

    # If 'fx' is truly a step function, then its increments should be mostly zero
    # To assert this, we fine-tune the tolerances and input grids so that the first
    # and last couple entries are zero.
    increments = np.diff(fx, axis=0)
    assert testing.allclose(increments[-15:], np.zeros((15, 2)))
    assert testing.allclose(increments[:5], np.zeros((5, 2)))
