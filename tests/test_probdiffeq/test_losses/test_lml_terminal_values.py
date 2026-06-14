"""Tests for marginal log likelihood functionality (terminal values)."""

from probdiffeq import ivpsolve, probdiffeq
from probdiffeq.backend import func, np, ode, testing, tree


@testing.case()
def case_strategy_filter():
    """Use the filter strategy."""
    return probdiffeq.strategy_filter


@testing.case()
def case_strategy_smoother_fixedinterval():
    """Use the fixed interval smoother strategy."""
    return probdiffeq.strategy_smoother_fixedinterval


@testing.case()
def case_strategy_smoother_fixedpoint():
    """Use the fixed point smoother strategy."""
    return probdiffeq.strategy_smoother_fixedpoint


@testing.fixture(name="solution_and_loss_and_data")
@testing.parametrize_with_cases("strategy_func", cases=".", prefix="case_strategy_")
@testing.parametrize(
    "ssm_factory",
    [
        probdiffeq.state_space_model_dense,
        probdiffeq.state_space_model_isotropic,
        probdiffeq.state_space_model_blockdiag,
    ],
)
def fixture_solution_and_loss_and_data(strategy_func, ssm_factory):
    """Solve the Lotka-Volterra IVP and set up the terminal-values LML loss and data."""
    vf, (u0,), (t0, t1) = ode.ivp_lotka_volterra()

    vf = probdiffeq.ode(vf)
    jetexpand = probdiffeq.jetexpand_ode_padded_scan(num=4)
    tcoeffs, _ = jetexpand(vf, (u0,), t=t0)
    ssm = ssm_factory()
    iwp = ssm.prior_wiener_integrated(tcoeffs)
    ts0 = ssm.constraint_ode_ts0(vf)
    strategy = strategy_func()
    solver = probdiffeq.solver(strategy=strategy, constraint=ts0)
    error = probdiffeq.error_residual_std(constraint=ts0)
    solve = ivpsolve.solve_adaptive_terminal_values(solver=solver, error=error)
    sol = func.jit(solve)(iwp, t0=t0, t1=t1, atol=1e-2, rtol=1e-2)

    loss = probdiffeq.loss_lml_terminal_values()
    data = sol.u.mean[0]
    std = (
        np.ones(())
        if isinstance(ssm, probdiffeq.state_space_model_isotropic)
        else tree.tree_map(np.ones_like, data)
    )
    return sol, loss, data, std


def test_lml_is_scalar_and_finite(solution_and_loss_and_data) -> None:
    """Assert that the terminal-values LML is a finite scalar."""
    solution, loss, data, std = solution_and_loss_and_data

    mll = func.jit(loss)(data, std=std, marginals=solution.u)

    assert mll.shape == ()
    assert not np.isnan(mll)
    assert not np.isinf(mll)


def test_raise_error_if_std_shape_is_wrong(solution_and_loss_and_data) -> None:
    """Assert that a std with the wrong container structure raises a ValueError."""
    solution, loss, data, std = solution_and_loss_and_data

    std = tree.tree_map(lambda s: s[None], std)
    with testing.raises(ValueError, match="container differs"):
        _ = loss(data, std=std, marginals=solution.u)
