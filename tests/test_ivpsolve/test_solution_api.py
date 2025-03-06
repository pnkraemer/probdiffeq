"""Tests for interaction with the solution API."""

from probdiffeq import ivpsolve, ivpsolvers, taylor
from probdiffeq.backend import containers, ode, testing
from probdiffeq.backend.typing import Array


class Taylor(containers.NamedTuple):
    """A non-standard Taylor-coefficient data structure."""

    state: Array
    velocity: Array
    acceleration: Array


@testing.fixture(name="pn_solution")
@testing.parametrize("fact", ["dense", "blockdiag", "isotropic"])
def fixture_pn_solution(fact):
    vf, u0, (t0, t1) = ode.ivp_lotka_volterra()

    # Generate a solver
    tcoeffs = Taylor(*taylor.odejet_padded_scan(lambda y: vf(y, t=t0), u0, num=2))
    ibm, ssm = ivpsolvers.prior_ibm(tcoeffs, ssm_fact=fact)

    ts0 = ivpsolvers.correction_ts0(ssm=ssm)
    strategy = ivpsolvers.strategy_filter(ssm=ssm)
    solver = ivpsolvers.solver_mle(strategy, prior=ibm, correction=ts0, ssm=ssm)
    asolver = ivpsolvers.adaptive(solver, atol=1e-2, rtol=1e-2, ssm=ssm)

    init = solver.initial_condition()
    return ivpsolve.solve_adaptive_save_every_step(
        vf, init, t0=t0, t1=t1, dt0=0.1, adaptive_solver=asolver, ssm=ssm
    )


def test_u_inherits_data_structure(pn_solution):
    assert isinstance(pn_solution.u, Taylor)


def test_u_std_inherits_data_structure(pn_solution):
    assert isinstance(pn_solution.u_std, Taylor)
