"""Compare solve_fixed_grid to solve_adaptive_save_every_step."""

from probdiffeq import ivpsolve, ivpsolvers, taylor
from probdiffeq.backend import containers, ode, testing
from probdiffeq.backend.typing import Array


@testing.parametrize("fact", ["dense", "isotropic", "blockdiag"])
def test_fixed_grid_result_matches_adaptive_grid_result(fact):
    vf, u0, (t0, t1) = ode.ivp_lotka_volterra()

    class Taylor(containers.NamedTuple):
        state: Array
        velocity: Array
        acceleration: Array

    tcoeffs = Taylor(*taylor.odejet_padded_scan(lambda y: vf(y, t=t0), u0, num=2))

    ibm, ssm = ivpsolvers.prior_ibm(tcoeffs, ssm_fact=fact)
    ts0 = ivpsolvers.correction_ts0(ssm=ssm)
    strategy = ivpsolvers.strategy_filter(ssm=ssm)
    solver = ivpsolvers.solver_mle(strategy, prior=ibm, correction=ts0, ssm=ssm)

    control = ivpsolvers.control_integral(clip=True)  # Any clipped controller will do.
    asolver = ivpsolvers.adaptive(
        solver, ssm=ssm, atol=1e-2, rtol=1e-2, control=control
    )

    init = solver.initial_condition()
    args = (vf, init)

    adaptive_kwargs = {
        "t0": t0,
        "t1": t1,
        "dt0": 0.1,
        "adaptive_solver": asolver,
        "ssm": ssm,
    }
    solution_adaptive = ivpsolve.solve_adaptive_save_every_step(
        *args, **adaptive_kwargs
    )
    assert isinstance(solution_adaptive.u, Taylor)

    grid_adaptive = solution_adaptive.t
    fixed_kwargs = {"grid": grid_adaptive, "solver": solver, "ssm": ssm}
    solution_fixed = ivpsolve.solve_fixed_grid(*args, **fixed_kwargs)
    assert testing.tree_all_allclose(solution_adaptive, solution_fixed)
