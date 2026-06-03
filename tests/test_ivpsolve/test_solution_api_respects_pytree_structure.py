"""Tests for interaction with the solution API."""

from probdiffeq import ivpsolve, probdiffeq
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
    vf = probdiffeq.ode_function(vf)
    jetexpand = probdiffeq.jetexpand_ode_padded_scan(num=2)
    coeffs, _ = jetexpand(vf, u0, t=t0)
    tcoeffs = Taylor(*coeffs)
    ssm = probdiffeq.state_space_model(ssm_fact=fact)
    init, iwp = probdiffeq.prior_wiener_integrated(tcoeffs, ssm=ssm)

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
