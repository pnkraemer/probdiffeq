"""Tests for interaction with the solution API."""

from probdiffeq import diffeqjet, ivpsolve, probdiffeq
from probdiffeq.backend import func, np, ode, structs, testing
from probdiffeq.backend.typing import Array


class Taylor(structs.NamedTuple):
    """A non-standard Taylor-coefficient data structure."""

    state: Array
    velocity: Array
    acceleration: Array


@testing.fixture(name="pn_solution")
@testing.parametrize("fact", ["dense", "blockdiag", "isotropic"])
def fixture_pn_solution(fact):
    vf, u0, (t0, t1) = ode.ivp_lotka_volterra()

    # Generate a solver
    tcoeffs = Taylor(*diffeqjet.odejet_padded_scan(lambda y: vf(y, t=t0), u0, num=2))
    init, ssm = probdiffeq.ssm_taylor(tcoeffs, ssm_fact=fact)
    iwp = probdiffeq.prior_wiener_integrated(ssm=ssm)

    ts0 = probdiffeq.constraint_ode_ts0(vf, ssm=ssm)
    strategy = probdiffeq.strategy_filter(ssm=ssm)
    solver = probdiffeq.solver_mle(
        strategy=strategy, prior=iwp, constraint=ts0, ssm=ssm
    )
    error = probdiffeq.error_residual_std(constraint=ts0, prior=iwp, ssm=ssm)
    solve = ivpsolve.solve_adaptive_save_at(solver=solver, error=error)
    save_at = np.linspace(t0, t1, endpoint=True, num=5)
    return func.jit(solve)(init, save_at=save_at, atol=1e-2, rtol=1e-2)


def test_u_inherits_data_structure(pn_solution) -> None:
    assert isinstance(pn_solution.u.mean, Taylor)


def test_u_std_inherits_data_structure(pn_solution) -> None:
    assert isinstance(pn_solution.u.std, Taylor)
