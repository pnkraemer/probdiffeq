"""Tests for marginal log likelihood functionality (terminal values)."""

from probdiffeq import diffeqjet, ivpsolve, probdiffeq
from probdiffeq.backend import func, np, ode, testing, tree


@testing.case()
def case_strategy_filter():
    return probdiffeq.strategy_filter


@testing.case()
def case_strategy_smoother_fixedinterval():
    return probdiffeq.strategy_smoother_fixedinterval


@testing.case()
def case_strategy_smoother_fixedpoint():
    return probdiffeq.strategy_smoother_fixedpoint


@testing.fixture(name="solution_and_loss_and_data")
@testing.parametrize_with_cases("strategy_func", cases=".", prefix="case_strategy_")
@testing.parametrize("fact", ["dense", "isotropic", "blockdiag"])
def fixture_solution_and_loss_and_data(strategy_func, fact):
    vf, (u0,), (t0, t1) = ode.ivp_lotka_volterra()

    tcoeffs = diffeqjet.odejet_padded_scan(lambda y: vf(y, t=t0), (u0,), num=4)
    init, ssm = probdiffeq.ssm_taylor(tcoeffs, ssm_fact=fact)
    iwp = probdiffeq.prior_wiener_integrated(ssm=ssm)
    ts0 = probdiffeq.constraint_ode_ts0(vf, ssm=ssm)
    strategy = strategy_func(ssm=ssm)
    solver = probdiffeq.solver(strategy=strategy, prior=iwp, constraint=ts0, ssm=ssm)
    error = probdiffeq.error_residual_std(constraint=ts0, prior=iwp, ssm=ssm)
    solve = ivpsolve.solve_adaptive_terminal_values(solver=solver, error=error)
    sol = func.jit(solve)(init, t0=t0, t1=t1, atol=1e-2, rtol=1e-2)

    loss = probdiffeq.loss_lml_terminal_values(ssm=ssm)
    data = sol.u.mean[0]
    std = (
        tree.tree_map(np.ones_like, data)
        if fact in ["dense", "blockdiag"]
        else np.ones(())
    )
    return sol, loss, data, std


def test_output_is_scalar(solution_and_loss_and_data) -> None:
    solution, loss, data, std = solution_and_loss_and_data

    mll = func.jit(loss)(data, std=std, marginals=solution.u)

    assert mll.shape == ()
    assert not np.isnan(mll)
    assert not np.isinf(mll)


def test_raise_error_if_std_shape_is_wrong(solution_and_loss_and_data) -> None:
    solution, loss, data, std = solution_and_loss_and_data

    std = tree.tree_map(lambda s: s[None], std)
    with testing.raises(ValueError, match="container differs"):
        _ = loss(data, std=std, marginals=solution.u)
