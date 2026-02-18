"""Assert that the base adaptive solver is accurate."""

from probdiffeq import ivpsolve, probdiffeq, taylor
from probdiffeq.backend import func, np, ode, structs, testing, tree
from probdiffeq.backend.typing import Callable


@testing.fixture(name="ivp")
def ivp_lotka_volterra():
    return ode.ivp_lotka_volterra()


@structs.dataclass
class Factory:
    """A solver factory.

    This data structure ensures that we don't test
    the product space of all configurations, whose size
    grows too quickly.
    """

    strategy: Callable = probdiffeq.strategy_filter
    solver: Callable = probdiffeq.solver
    constraint: Callable = probdiffeq.constraint_ode_ts0
    error: Callable = probdiffeq.error_residual_std


@testing.case
def case_factory_strategy_filter():
    return Factory(strategy=probdiffeq.strategy_filter)


@testing.case
def case_factory_strategy_smoother_fixedpoint():
    return Factory(strategy=probdiffeq.strategy_smoother_fixedpoint)


@testing.case
def case_factory_solver_solver():
    return Factory(solver=probdiffeq.solver)


@testing.case
def case_factory_solver_mle():
    return Factory(solver=probdiffeq.solver_mle)


@testing.case
def case_factory_solver_dynamic_without_relinearization():
    dynamic = func.partial(
        probdiffeq.solver_dynamic, re_linearize_after_calibration=False
    )
    return Factory(solver=dynamic)


@testing.case
def case_factory_solver_dynamic_with_relinearization():
    dynamic = func.partial(
        probdiffeq.solver_dynamic, re_linearize_after_calibration=True
    )
    return Factory(solver=dynamic)


@testing.case
def case_factory_constraint_ode_ts0():
    return Factory(constraint=probdiffeq.constraint_ode_ts0)


@testing.case
@testing.parametrize("jacfun", [func.jacfwd, func.jacrev])
def case_factory_constraint_ode_ts1_materialize(jacfun):
    jacobian = probdiffeq.jacobian_materialize(jacfun=jacfun)
    constraint = func.partial(probdiffeq.constraint_ode_ts1, jacobian=jacobian)
    return Factory(constraint=constraint)


@testing.case
def case_factory_constraint_ode_ts1_hutchinson_fwd():
    jacobian = probdiffeq.jacobian_hutchinson_fwd()
    constraint = func.partial(probdiffeq.constraint_ode_ts1, jacobian=jacobian)
    return Factory(constraint=constraint)


@testing.case
def case_factory_constraint_ode_ts1_hutchinson_rev():
    jacobian = probdiffeq.jacobian_hutchinson_rev()
    constraint = func.partial(probdiffeq.constraint_ode_ts1, jacobian=jacobian)
    return Factory(constraint=constraint)


@testing.case
def case_factory_constraint_root_ts1(ivp):
    vf, _u0, (_t0, _t1) = ivp

    jacobian = probdiffeq.jacobian_materialize()

    def root(u, du, /, *, t):
        return tree.tree_map(lambda a, b: a - b, du, vf(u, t=t))

    constraint_fn = func.partial(probdiffeq.constraint_root_ts1, jacobian=jacobian)

    def constraint(vf, **kwargs):
        try:
            del vf  # no vector fields, we use the root instead
            return constraint_fn(root, **kwargs)
        except NotImplementedError:
            reason = "This linearisation is not implemented"
            reason += ", likely due to the selected state-space factorisation."
            testing.skip(reason)

    return Factory(constraint=constraint)


@testing.case
def case_factory_constraint_ode_slr0():
    def constraint(*args, **kwargs):
        try:
            return probdiffeq.constraint_ode_slr0(*args, **kwargs)
        except NotImplementedError:
            reason = "This linearisation is not implemented"
            reason += ", likely due to the selected state-space factorisation."
            testing.skip(reason)

    return Factory(constraint=constraint)


@testing.case
def case_factory_constraint_ode_slr1():
    def constraint(*args, **kwargs):
        try:
            return probdiffeq.constraint_ode_slr1(*args, **kwargs)
        except NotImplementedError:
            reason = "This linearisation is not implemented"
            reason += ", likely due to the selected state-space factorisation."
            testing.skip(reason)

    return Factory(constraint=constraint)


@testing.case
def case_factory_error_residual_std_cached():
    error = func.partial(probdiffeq.error_residual_std, re_linearize_before_error=True)
    return Factory(error=error)


@testing.case
def case_factory_error_residual_std_not_cached():
    error = func.partial(probdiffeq.error_residual_std, re_linearize_before_error=False)
    return Factory(error=error)


@testing.parametrize("ssm_fact", ["dense", "isotropic", "blockdiag"])
@testing.parametrize_with_cases("factory", ".", prefix="case_factory_")
def test_output_matches_reference(ivp, ssm_fact, factory: Factory):
    vf, u0, (t0, t1) = ivp

    # Build a solver
    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), u0, num=4)
    init, iwp, ssm = probdiffeq.prior_wiener_integrated(tcoeffs, ssm_fact=ssm_fact)
    strategy = factory.strategy(ssm=ssm)
    constraint = factory.constraint(vf, ssm=ssm)
    solver = factory.solver(
        strategy=strategy, prior=iwp, constraint=constraint, ssm=ssm
    )
    error = factory.error(prior=iwp, ssm=ssm, constraint=constraint)

    # Compute the PN solution
    save_at = np.linspace(t0, t1, endpoint=True, num=7)
    solve = ivpsolve.solve_adaptive_save_at(solver=solver, error=error)
    received = func.jit(solve)(init, save_at=save_at, atol=1e-3, rtol=1e-3)

    # Compute a reference solution
    expected = ode.odeint_and_save_at(vf, u0, save_at=save_at, atol=1e-7, rtol=1e-7)

    # The results should be very similar
    assert testing.allclose(received.u.mean[0], expected)

    # Assert u and u_std have matching shapes (that was wrong before)
    u_shape = tree.tree_map(np.shape, received.u.mean)
    u_std_shape = tree.tree_map(np.shape, received.u.std)
    match = tree.tree_map(lambda a, b: a == b, u_shape, u_std_shape)
    assert tree.tree_all(match)
