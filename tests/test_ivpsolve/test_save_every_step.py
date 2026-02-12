"""Assert that solve_with_python_loop is accurate."""

from probdiffeq import ivpsolve, probdiffeq, taylor
from probdiffeq.backend import numpy as np
from probdiffeq.backend import ode, testing, tree_util


@testing.parametrize("fact", ["dense", "isotropic", "blockdiag"])
@testing.parametrize(
    "strategy", [probdiffeq.strategy_filter, probdiffeq.strategy_smoother]
)
def test_python_loop_output_matches_reference(fact, strategy):
    ivp = ode.ivp_lotka_volterra()

    received = python_loop_solution(ivp, fact=fact, strategy_fun=strategy)
    expected = reference_solution(ivp, received.t)
    assert testing.allclose(received.u.mean[0], expected, rtol=1e-2)

    # Assert u and u_std have matching shapes (that was wrong before)
    u_shape = tree_util.tree_map(np.shape, received.u.mean)
    u_std_shape = tree_util.tree_map(np.shape, received.u.std)
    match = tree_util.tree_map(lambda a, b: a == b, u_shape, u_std_shape)
    assert tree_util.tree_all(match)


def python_loop_solution(ivp, *, fact, strategy_fun):
    vf, u0, (t0, t1) = ivp

    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), u0, num=4)
    init, transition, ssm = probdiffeq.prior_wiener_integrated(tcoeffs, ssm_fact=fact)

    ts0 = probdiffeq.correction_ts0(vf, ssm=ssm)
    strategy = strategy_fun(ssm=ssm)
    solver = probdiffeq.solver_mle(strategy, prior=transition, correction=ts0, ssm=ssm)

    # Adaptive solvers need an error estimate
    errorest = probdiffeq.errorest_schober_bosch(
        prior=transition, ssm=ssm, correction=ts0, atol=1e-2, rtol=1e-2
    )
    return ivpsolve.solve_adaptive_save_every_step(
        init, t0=t0, t1=t1, solver=solver, dt0=0.1, errorest=errorest
    )


def reference_solution(ivp, ts):
    vf, u0, (t0, t1) = ivp
    return ode.odeint_and_save_at(vf, u0, save_at=ts, atol=1e-10, rtol=1e-10)
