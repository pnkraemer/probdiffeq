"""Tests for interaction with the solution API."""

from probdiffeq import ivpsolve, probdiffeq, taylor
from probdiffeq.backend import containers, functools, ode, testing
from probdiffeq.backend import numpy as np
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
    init, ibm, ssm = probdiffeq.prior_wiener_integrated(tcoeffs, ssm_fact=fact)

    ts0 = probdiffeq.constraint_ode_ts0(vf, ssm=ssm)
    strategy = probdiffeq.strategy_filter(ssm=ssm)
    solver = probdiffeq.solver_mle(strategy, prior=ibm, correction=ts0, ssm=ssm)
    errorest = probdiffeq.errorest_local_residual(prior=ibm, correction=ts0, ssm=ssm)
    solve = ivpsolve.solve_adaptive_save_at(solver=solver, errorest=errorest)
    save_at = np.linspace(t0, t1, endpoint=True, num=5)
    return functools.jit(solve)(init, save_at=save_at, atol=1e-2, rtol=1e-2)


def test_u_inherits_data_structure(pn_solution):
    assert isinstance(pn_solution.u.mean, Taylor)


def test_u_std_inherits_data_structure(pn_solution):
    assert isinstance(pn_solution.u.std, Taylor)
