"""Tests for log-marginal-likelihood functionality."""

from probdiffeq import ivpsolve, probdiffeq
from probdiffeq.backend import func, np, ode, testing, tree


@testing.fixture(name="solution")
@testing.parametrize(
    "ssm_factory",
    [
        probdiffeq.state_space_model_isotropic,
        probdiffeq.state_space_model_blockdiag,
        probdiffeq.state_space_model_dense,
    ],
)
def fixture_solution(ssm_factory):
    """Solve the Lotka-Volterra IVP and set up the timeseries LML loss and data."""
    vf, (u0,), (t0, t1) = ode.ivp_lotka_volterra()

    vf = probdiffeq.ode(vf)
    jetexpand = probdiffeq.jetexpand_ode_padded_scan(num=2)
    tcoeffs, _ = jetexpand(vf, (u0,), t=t0)
    ssm = ssm_factory()
    iwp = ssm.prior_wiener_integrated(tcoeffs)

    ts0 = ssm.constraint_ode_ts0(vf)
    strategy = probdiffeq.strategy_smoother_fixedpoint()

    solver = probdiffeq.solver(strategy=strategy, constraint=ts0)
    error = probdiffeq.error_residual_std(constraint=ts0)

    save_at = np.linspace(t0, t1, endpoint=True, num=13)
    solve = ivpsolve.solve_adaptive_save_at(error=error, solver=solver)
    sol = func.jit(solve)(iwp, save_at=save_at, atol=1e-2, rtol=1e-2)

    loss = probdiffeq.loss_lml_timeseries()
    data = sol.u.mean[0]
    std = (
        np.ones_like(save_at)
        if isinstance(ssm, probdiffeq.state_space_model_isotropic)
        else tree.tree_map(np.ones_like, data)
    )
    return sol, loss, data, std


def test_lml_is_scalar_and_finite(solution) -> None:
    """Assert that the timeseries LML is a finite scalar."""
    sol, loss, data, std = solution

    lml = func.jit(loss)(data, posterior=sol.solution_full, std=std)

    assert lml.shape == ()
    assert not np.isnan(lml)
    assert not np.isinf(lml)


def test_that_function_raises_error_for_wrong_number_of_timesteps(solution) -> None:
    """Test that the log-marginal-likelihood function complains about the wrong shape.

    Specifically, about receiving fewer standard-deviations than data-points.
    """
    sol, loss, data, std = solution
    std = tree.tree_map(lambda s: s[:-1], std)

    with testing.raises(ValueError, match="container differs"):
        _ = loss(data, posterior=sol.solution_full, std=std)


def test_raises_error_if_terminal_values_were_intended(solution) -> None:
    """Test that the log-marginal-likelihood function complains when called incorrectly.

    Specifically, raise an error when calling log_marginal_likelihood even though
    log_marginal_likelihood_terminal_values was meant.
    """
    sol, loss, data, std = solution

    # Call with marginals to pretend we're a filter
    with testing.raises(TypeError, match="datatype"):
        _ = loss(data, posterior=sol.u, std=std)
