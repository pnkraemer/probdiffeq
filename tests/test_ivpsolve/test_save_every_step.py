"""Assert that solve_with_python_loop is accurate."""

from probdiffeq import ivpsolve, ivpsolvers, taylor
from probdiffeq.backend import numpy as np
from probdiffeq.backend import ode, testing


@testing.parametrize("fact", ["dense", "isotropic", "blockdiag"])
def test_python_loop_output_matches_reference(fact):
    ivp = ode.ivp_lotka_volterra()

    received = python_loop_solution(ivp, fact=fact)
    expected = reference_solution(ivp, received.t)
    # print(ivp[1])
    # print(received.marginals.mean)
    # print(received.u)
    # assert False
    print(received.u[0])
    # print(received.u[1])
    # print(received.u[2])
    # print(received.u[3])
    print(expected)
    assert np.allclose(received.u[0], expected, rtol=1e-2)


def python_loop_solution(ivp, *, fact):
    vf, u0, (t0, t1) = ivp

    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), u0, num=4)
    ibm, ssm = ivpsolvers.prior_ibm(tcoeffs, ssm_fact=fact)

    ts0 = ivpsolvers.correction_ts0(ssm=ssm)
    strategy = ivpsolvers.strategy_filter(ibm, ts0, ssm=ssm)
    solver = ivpsolvers.solver_mle(strategy, ssm=ssm)
    control = ivpsolve.control_proportional_integral(clip=True)
    adaptive_solver = ivpsolve.adaptive(
        solver, atol=1e-2, rtol=1e-2, control=control, ssm=ssm
    )

    dt0 = ivpsolve.dt0_adaptive(
        vf, u0, t0=t0, atol=1e-2, rtol=1e-2, error_contraction_rate=5
    )

    init = solver.initial_condition()

    args = (vf, init)
    kwargs = {
        "t0": t0,
        "t1": t1,
        "adaptive_solver": adaptive_solver,
        "dt0": dt0,
        "ssm": ssm,
    }
    return ivpsolve.solve_adaptive_save_every_step(*args, **kwargs)


def reference_solution(ivp, ts):
    vf, u0, (t0, t1) = ivp
    sol = ode.odeint_dense(vf, u0, t0=t0, t1=t1, atol=1e-10, rtol=1e-10)
    return sol(ts)
