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
    init, ibm, ssm = ivpsolvers.prior_wiener_integrated(tcoeffs, ssm_fact=fact)

    ts0 = ivpsolvers.correction_ts0(vf, ssm=ssm)
    strategy = ivpsolvers.strategy_filter(ssm=ssm)
    solver = ivpsolvers.solver_mle(strategy, prior=ibm, correction=ts0, ssm=ssm)
    errorest = ivpsolvers.errorest_schober(
        prior=ibm, correction=ts0, atol=1e-2, rtol=1e-2, ssm=ssm
    )
    return ivpsolve.solve_adaptive_save_every_step(
        init, t0=t0, t1=t1, solver=solver, errorest=errorest
    )


def test_u_inherits_data_structure(pn_solution):
    assert isinstance(pn_solution.u.mean, Taylor)


def test_u_std_inherits_data_structure(pn_solution):
    assert isinstance(pn_solution.u.std, Taylor)
