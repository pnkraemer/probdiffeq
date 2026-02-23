"""Tests for marginal log likelihood functionality (terminal values)."""

from probdiffeq import ivpsolve, probdiffeq, taylor
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

    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), (u0,), num=4)
    init, ibm, ssm = probdiffeq.prior_wiener_integrated(tcoeffs, ssm_fact=fact)
    ts0 = probdiffeq.constraint_ode_ts0(vf, ssm=ssm)
    strategy = strategy_func(ssm=ssm)
    solver = probdiffeq.solver(strategy=strategy, prior=ibm, constraint=ts0, ssm=ssm)
    error = probdiffeq.error_residual_std(constraint=ts0, prior=ibm, ssm=ssm)
    solve = ivpsolve.solve_adaptive_terminal_values(solver=solver, error=error)
    sol = func.jit(solve)(init, t0=t0, t1=t1, atol=1e-2, rtol=1e-2)

    def linop(tcoeffs):
        return {"output": 2 * tcoeffs[0]["u"]}

    # TODO: multiple LMLs...
    loss = probdiffeq.loss_lml_terminal_values(ssm=ssm)
    loss = probdiffeq.loss_lml_terminal_values_linop(ssm=ssm)
    data = linop(sol.u.mean)
    return sol, loss, data


def test_output_is_scalar_and_not_inf_and_not_nan(solution_and_loss_and_data):
    """Test that terminal-value log-marginal-likelihood calls work with all strategies.

    See also: issue #477 (closed).
    """
    solution, loss, data = solution_and_loss_and_data

    std = tree.tree_map(np.ones_like, data)
    mll = func.jit(loss)(data, std, marginals=solution.u.marginals)

    assert mll.shape == ()
    assert not np.isnan(mll)
    assert not np.isinf(mll)


def test_raise_error_if_std_shape_differs_from_u_shape(solution_and_loss_and_data):
    solution, loss, data = solution_and_loss_and_data

    std = tree.tree_map(lambda s: 1.0, data)  # not the correct pytree

    with testing.raises(ValueError, match="container differs"):
        _ = loss(data, std, marginals=solution.u.marginals)
