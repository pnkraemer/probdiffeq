"""Assert that the base adaptive solver is accurate."""

from probdiffeq import ivpsolve, probdiffeq, taylor
from probdiffeq.backend import func, np, ode, testing, tree


@testing.case
def case_strategy_filter():
    return probdiffeq.strategy_filter


@testing.case
def case_strategy_smoother_fixedpoint():
    return probdiffeq.strategy_smoother_fixedpoint


@testing.case
def case_solver_solver():
    return probdiffeq.solver


@testing.case
def case_solver_mle():
    return probdiffeq.solver_mle


@testing.case
def case_solver_dynamic_without_relinearization():
    return func.partial(probdiffeq.solver_dynamic, re_linearize_after_calibration=False)


@testing.case
def case_solver_dynamic():
    return probdiffeq.solver_dynamic


@testing.case
def case_constraint_ode_ts0():
    return probdiffeq.constraint_ode_ts0


@testing.case
@testing.parametrize("jacfun", [func.jacfwd, func.jacrev])
def case_constraint_ode_ts1_materialize(jacfun):
    jacobian = probdiffeq.jacobian_materialize(jacfun=jacfun)
    return func.partial(probdiffeq.constraint_ode_ts1, jacobian=jacobian)


@testing.case
def case_constraint_ode_ts1_hutchinson_fwd():
    jacobian = probdiffeq.jacobian_hutchinson_fwd()
    return func.partial(probdiffeq.constraint_ode_ts1, jacobian=jacobian)


@testing.case
def case_constraint_ode_ts1_hutchinson_rev():
    jacobian = probdiffeq.jacobian_hutchinson_rev()
    return func.partial(probdiffeq.constraint_ode_ts1, jacobian=jacobian)


@testing.case
def case_constraint_root_ts1():
    jacobian = probdiffeq.jacobian_materialize()

    def root(vf, u, du):
        return tree.tree_map(lambda a, b: a - b, du, vf(u))

    constraint_fn = func.partial(
        probdiffeq.constraint_root_ts1, root, jacobian=jacobian
    )

    def constraint(*args, **kwargs):
        try:
            return constraint_fn(*args, **kwargs)
        except NotImplementedError:
            reason = "This linearisation is not implemented"
            reason += ", likely due to the selected state-space factorisation."
            testing.skip(reason)

    return constraint


@testing.case
def case_constraint_ode_slr0():
    def constraint(*args, **kwargs):
        try:
            return probdiffeq.constraint_ode_slr0(*args, **kwargs)
        except NotImplementedError:
            reason = "This linearisation is not implemented"
            reason += ", likely due to the selected state-space factorisation."
            testing.skip(reason)

    return constraint


@testing.case
def case_constraint_ode_slr1():
    def constraint(*args, **kwargs):
        try:
            return probdiffeq.constraint_ode_slr1(*args, **kwargs)
        except NotImplementedError:
            reason = "This linearisation is not implemented"
            reason += ", likely due to the selected state-space factorisation."
            testing.skip(reason)

    return constraint


@testing.case
def case_errorest_local_residual_cached():
    def residual_wrapper(vector_field, constraint, **kw):
        del vector_field
        del constraint
        return probdiffeq.errorest_local_residual_cached(**kw)

    return residual_wrapper


@testing.case
def case_errorest_local_residual():
    return probdiffeq.errorest_local_residual


@testing.parametrize("fact", ["dense", "isotropic", "blockdiag"])
@testing.parametrize_with_cases("strategy_factory", ".", prefix="case_strategy_")
@testing.parametrize_with_cases("solver_factory", ".", prefix="case_solver_")
@testing.parametrize_with_cases("constraint_factory", ".", prefix="case_constraint_")
@testing.parametrize_with_cases("errorest_factory", ".", prefix="case_errorest_")
def test_output_matches_reference(
    fact, solver_factory, constraint_factory, strategy_factory, errorest_factory
):
    vf, u0, (t0, t1) = ode.ivp_lotka_volterra()

    # Build a solver
    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), u0, num=4)
    init, iwp, ssm = probdiffeq.prior_wiener_integrated(tcoeffs, ssm_fact=fact)
    strategy = strategy_factory(ssm=ssm)
    constraint = constraint_factory(ssm=ssm)
    solver = solver_factory(
        vf, strategy=strategy, prior=iwp, constraint=constraint, ssm=ssm
    )
    errorest = errorest_factory(
        prior=iwp, ssm=ssm, vector_field=vf, constraint=constraint
    )

    # Compute the PN solution
    save_at = np.linspace(t0, t1, endpoint=True, num=7)
    solve = ivpsolve.solve_adaptive_save_at(solver=solver, errorest=errorest)
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
