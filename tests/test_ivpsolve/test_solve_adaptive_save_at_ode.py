"""Assert that the base adaptive solver is accurate."""

from probdiffeq import ivpsolve, probdiffeq
from probdiffeq.backend import func, np, ode, structs, testing, tree
from probdiffeq.backend.typing import Callable


@testing.fixture(name="ivp")
def ivp_lotka_volterra():
    vf, (u0,), (t0, t1) = ode.ivp_lotka_volterra()
    vf = probdiffeq.ode(vf)
    return vf, (u0,), (t0, t1)


@structs.dataclass
class Factory:
    """A solver factory.

    This data structure ensures that we don't test
    the product space of all configurations, whose size
    would grow too quickly.

    Instead, we carry defaults for each parameter
    and make each case only vary one of the parameters.
    """

    prior: Callable = probdiffeq.prior_wiener_integrated
    strategy: Callable = probdiffeq.strategy_filter
    solver: Callable = probdiffeq.solver
    jacobian: Callable = probdiffeq.jacobian_materialize

    # ts1 default because it uses more backend functions (eg Jacobians)
    # so the tests gain relevance
    constraint: Callable = probdiffeq.constraint_ode_ts1
    error: Callable = probdiffeq.error_residual_std


@testing.case
def case_factory_prior_wiener_integrated():
    return Factory(prior=probdiffeq.prior_wiener_integrated)


@testing.case
def case_factory_prior_ioup():
    def prior(*args, **kwargs):
        try:

            def linop(u, /):
                return tree.tree_map(lambda s: 0.01 * np.flip(s), u)

            return probdiffeq.prior_ornstein_uhlenbeck_integrated(
                linop, *args, **kwargs
            )
        except NotImplementedError:
            reason = "This prior is not implemented"
            reason += ", likely due to the selected state-space factorisation."
            testing.skip(reason)

    return Factory(prior=prior)


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
def case_factory_constraint_ode_ts1():
    return Factory(constraint=probdiffeq.constraint_ode_ts1)


@testing.case
@testing.parametrize("jacfun", [func.jacfwd, func.jacrev])
def case_factory_jacobian_materialize(jacfun):
    jacobian = probdiffeq.jacobian_materialize(jacfun=jacfun)
    return Factory(jacobian=jacobian)


@testing.case
def case_factory_jacobian_hutchinson_fwd():
    jacobian = probdiffeq.jacobian_hutchinson_fwd()
    return Factory(jacobian=jacobian)


@testing.case
def case_factory_jacobian_hutchinson_rev():
    jacobian = probdiffeq.jacobian_hutchinson_rev()
    return Factory(jacobian=jacobian)


@testing.case
def case_factory_constraint_root_ts1(ivp):
    vf, _u0, (_t0, _t1) = ivp

    # Always materialize to stabilise blockdiagonal/isotropic TS1
    jacobian = probdiffeq.jacobian_materialize()

    def root(u, du, /, *, t):
        return tree.tree_map(lambda a, b: a - b, du, vf(u=(u,), t=t))

    root = probdiffeq.implicit(root, jacobian=jacobian)

    def constraint(vf, **kwargs):
        try:
            del vf  # no vector fields, we use the root instead
            return probdiffeq.constraint(root, **kwargs)
        except NotImplementedError:
            reason = "This linearisation is not implemented"
            reason += ", likely due to the selected state-space factorisation."
            testing.skip(reason)

    return Factory(constraint=constraint)


@testing.case
def case_factory_error_state_std_cached():
    error = func.partial(probdiffeq.error_state_std, re_linearize_before_error=True)
    return Factory(error=error)


@testing.case
def case_factory_error_state_std_not_cached():
    error = func.partial(probdiffeq.error_state_std, re_linearize_before_error=True)
    return Factory(error=error)


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
def test_output_matches_reference(ivp, ssm_fact, factory: Factory) -> None:
    vf, u0, (t0, t1) = ivp

    ssm = probdiffeq.state_space_model(ssm_fact=ssm_fact)

    # Build a solver
    initialize = probdiffeq.jetexpand_ode_padded_scan(num=4)

    tcoeffs = initialize(vf, u0, t=t0)
    init, prior = factory.prior(tcoeffs, ssm=ssm)
    strategy = factory.strategy(ssm=ssm)
    constraint = factory.constraint(vf, ssm=ssm)
    solver = factory.solver(
        strategy=strategy, prior=prior, constraint=constraint, ssm=ssm
    )
    # not all constraints have shape (d,):
    error_norm = probdiffeq.error_norm_rms_then_scale()
    error = factory.error(
        prior=prior, ssm=ssm, constraint=constraint, error_norm=error_norm
    )

    # Compute the PN solution
    save_at = np.linspace(t0, t1, endpoint=True, num=7)
    solve = ivpsolve.solve_adaptive_save_at(solver=solver, error=error)
    received = func.jit(solve)(init, save_at=save_at, atol=1e-4, rtol=1e-4)

    # Compute a reference solution
    expected = ode.odeint_and_save_at(vf, u0, save_at=save_at, atol=1e-7, rtol=1e-7)

    # The results should be very similar
    assert testing.allclose(received.u.mean[0], expected)

    # Assert u and u_std have matching treedefs (that was wrong before)
    # but the shapes of the leaves may be different, e.g. STDs in isotropic
    # models are always scalar
    _, meandef = tree.tree_flatten_depth_one(received.u.mean)
    _, stddef = tree.tree_flatten_depth_one(received.u.std)
    match = tree.tree_map(lambda a, b: a == b, meandef, stddef)
    assert tree.tree_all(match)
