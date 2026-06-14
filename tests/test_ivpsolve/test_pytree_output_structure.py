"""Tests for pytree output structure."""

from probdiffeq import ivpsolve, probdiffeq
from probdiffeq.backend import func, np, ode, structs, testing
from probdiffeq.backend.typing import Array


class Taylor(structs.NamedTuple):
    """A non-standard Taylor-coefficient data structure."""

    state: Array
    velocity: Array
    acceleration: Array


@testing.fixture(name="pn_solution")
@testing.parametrize(
    "ssm_factory",
    [
        probdiffeq.state_space_model_dense,
        probdiffeq.state_space_model_blockdiag,
        probdiffeq.state_space_model_isotropic,
    ],
)
def fixture_pn_solution(ssm_factory):
    """Solve the Lotka-Volterra IVP with a custom Taylor pytree and return the solution."""
    vf, u0, (t0, t1) = ode.ivp_lotka_volterra()

    # Generate a solver
    vf = probdiffeq.ode(vf)
    jetexpand = probdiffeq.jetexpand_ode_padded_scan(num=2)
    coeffs, _ = jetexpand(vf, u0, t=t0)
    tcoeffs = Taylor(*coeffs)
    ssm = ssm_factory()
    iwp = ssm.prior_wiener_integrated(tcoeffs)

    ts0 = ssm.constraint_ode_ts0(vf)
    strategy = probdiffeq.strategy_filter()
    solver = probdiffeq.solver_mle(strategy=strategy, constraint=ts0)
    error = probdiffeq.error_residual_std(constraint=ts0)
    solve = ivpsolve.solve_adaptive_save_at(solver=solver, error=error)
    save_at = np.linspace(t0, t1, endpoint=True, num=5)
    return func.jit(solve)(iwp, save_at=save_at, atol=1e-2, rtol=1e-2)


def test_u_mean_and_std_inherit_data_structure(pn_solution) -> None:
    """Assert that u.mean and u.std both inherit the custom Taylor pytree structure."""
    assert isinstance(pn_solution.u.mean, Taylor)
    assert isinstance(pn_solution.u.std, Taylor)
