"""Tests for sampling behaviour."""

from probdiffeq import ivpsolve, probdiffeq
from probdiffeq.backend import func, np, testing


def test_solution_matches_odesolve():
    y0 = [np.asarray([0.99, 0.01, 0.0])]
    solution_ode = solve_ode(y0, num=3)
    solution_dae = solve_dae(y0, num=3)
    assert testing.allclose(solution_ode.u.mean[0], solution_dae.u.mean[0])


@func.partial(func.jit, static_argnames=("num",))
def solve_ode(inits, num):
    """Solve the SIR model as an ODE to serve as a reference solution."""

    @probdiffeq.ode
    def vf_ode(y, /, *, t):
        del t
        beta, gamma = 2.0, 0.5  # infection and recovery rates
        S, I, _R = y  # noqa: E741 ("I" is a good variable name in an SIR model)

        f0 = -beta * S * I
        f1 = beta * S * I - gamma * I
        f2 = gamma * I

        return np.stack([f0, f1, f2])

    jetexpand = probdiffeq.jetexpand_ode_padded_scan(num=num)
    tcoeffs, _ = jetexpand(vf_ode, inits, t=0.0)

    ssm = probdiffeq.state_space_model(ssm_fact="dense")
    init, iwp = probdiffeq.prior_wiener_integrated(tcoeffs, ssm=ssm)
    ts0 = probdiffeq.constraint_ode_ts0(vf_ode, ssm=ssm)
    strategy = probdiffeq.strategy_filter(ssm=ssm)
    solver = probdiffeq.solver(strategy=strategy, prior=iwp, constraint=ts0, ssm=ssm)
    error = probdiffeq.error_state_std(constraint=ts0, prior=iwp, ssm=ssm)
    solve = ivpsolve.solve_adaptive_save_at(solver=solver, error=error)

    save_at = np.linspace(0.0, 5.0, endpoint=True, num=10)
    return func.jit(solve)(init, save_at=save_at, atol=1e-6, rtol=1e-6)


@func.partial(func.jit, static_argnames=("num",))
def solve_dae(inits, num):
    """Solve the SIR model as a DAE."""

    @func.partial(probdiffeq.jet_lift, lift_by=num)
    @probdiffeq.residual_state
    def algebraic(u, /, *, t):
        del t
        N = 1.0  # total population
        return u[0] + u[1] + u[2] - N

    @func.partial(probdiffeq.jet_lift, lift_by=num - 1)
    @probdiffeq.residual_state_velocity
    def differential(u, du, /, *, t):
        del t
        beta, gamma = 2.0, 0.5
        S, I, _R = u  # noqa: E741 ("I" is a good variable name in an SIR model)

        f0 = -beta * S * I
        f1 = beta * S * I - gamma * I

        F1 = du[0] - f0
        F2 = du[1] - f1

        return np.stack([F1, F2])

    dae = probdiffeq.dae_system(differential=differential, algebraic=algebraic)

    jetexpand = probdiffeq.jetexpand_dae_nlstsq(num=num)
    tcoeffs, _ = jetexpand(dae, inits, t=0.0)

    ssm = probdiffeq.state_space_model(ssm_fact="dense")
    init, iwp = probdiffeq.prior_wiener_integrated(tcoeffs, ssm=ssm)
    ts0 = probdiffeq.constraint_dae(dae, ssm=ssm)
    strategy = probdiffeq.strategy_filter(ssm=ssm)
    solver = probdiffeq.solver(strategy=strategy, prior=iwp, constraint=ts0, ssm=ssm)
    error = probdiffeq.error_state_std(constraint=ts0, prior=iwp, ssm=ssm)
    solve = ivpsolve.solve_adaptive_save_at(solver=solver, error=error)

    save_at = np.linspace(0.0, 5.0, endpoint=True, num=10)
    return func.jit(solve)(init, save_at=save_at, atol=1e-6, rtol=1e-6)
