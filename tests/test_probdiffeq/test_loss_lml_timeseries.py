"""Tests for log-marginal-likelihood functionality."""

from probdiffeq import diffeqjet, ivpsolve, probdiffeq
from probdiffeq.backend import func, np, ode, testing, tree


@testing.fixture(name="solution")
@testing.parametrize("fact", ["isotropic", "blockdiag", "dense"])
def fixture_solution(fact):
    vf, (u0,), (t0, t1) = ode.ivp_lotka_volterra()

    tcoeffs = diffeqjet.odejet_padded_scan(lambda y: vf(y, t=t0), (u0,), num=2)
    init, ssm = probdiffeq.ssm_taylor(tcoeffs, ssm_fact=fact)
    iwp = probdiffeq.prior_wiener_integrated(ssm=ssm)

    ts0 = probdiffeq.constraint_ode_ts0(vf, ssm=ssm)
    strategy = probdiffeq.strategy_smoother_fixedpoint(ssm=ssm)

    solver = probdiffeq.solver(strategy=strategy, prior=iwp, constraint=ts0, ssm=ssm)
    error = probdiffeq.error_residual_std(constraint=ts0, prior=iwp, ssm=ssm)

    save_at = np.linspace(t0, t1, endpoint=True, num=13)
    solve = ivpsolve.solve_adaptive_save_at(error=error, solver=solver)
    sol = func.jit(solve)(init, save_at=save_at, atol=1e-2, rtol=1e-2)

    loss = probdiffeq.loss_lml_timeseries(ssm=ssm)
    data = sol.u.mean[0]
    std = (
        tree.tree_map(np.ones_like, data)
        if fact in ["dense", "blockdiag"]
        else np.ones_like(save_at)
    )
    return sol, loss, data, std


def test_output_is_a_scalar(solution) -> None:
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
