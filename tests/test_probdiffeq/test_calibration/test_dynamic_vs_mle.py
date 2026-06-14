"""Assert that the dynamic solver can successfully solve a linear function.

Specifically, we solve a linear function with exponentially increasing output-scale.
This is difficult for the MLE- and calibration-free solver,
but not for the dynamic solver.
"""

from probdiffeq import ivpsolve, probdiffeq
from probdiffeq.backend import func, linalg, np, ode, testing, tree


@testing.parametrize(
    "ssm_factory",
    [
        probdiffeq.state_space_model_dense,
        probdiffeq.state_space_model_isotropic,
        probdiffeq.state_space_model_blockdiag,
    ],
)
def test_dynamic_solver_tracks_exponential_output_scale(ssm_factory) -> None:
    """Assert that the dynamic solver achieves low RMSE on a problem with exponentially increasing output scale."""
    vf, u0, (t0, t1) = ode.ivp_lotka_volterra()

    tcoeffs = (*u0, vf(*u0, t=t0))
    ssm = ssm_factory()
    iwp = ssm.prior_wiener_integrated(tcoeffs)
    vf = probdiffeq.ode(vf)
    ts0 = ssm.constraint_ode_ts0(vf)
    strategy = probdiffeq.strategy_filter()
    solver = probdiffeq.solver_dynamic(strategy=strategy, constraint=ts0)

    grid = np.linspace(t0, t1, num=20)
    solve = ivpsolve.solve_fixed_grid(solver=solver)
    approximation = func.jit(solve)(iwp, grid=grid)

    solution = ode.odeint_and_save_at(
        vf, u0, save_at=np.asarray([t0, t1]), atol=1e-5, rtol=1e-5
    )
    vmap_ravel = func.vmap(lambda s: tree.ravel_pytree(s)[0])
    u = vmap_ravel(approximation.u.mean[0])
    sol = vmap_ravel(solution)
    rmse = _rmse(u[-1], sol[-1])
    assert rmse < 0.1


def _rmse(a, b):
    return linalg.vector_norm((a - b) / b) / np.sqrt(b.size)
