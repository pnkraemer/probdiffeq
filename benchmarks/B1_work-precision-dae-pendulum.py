"""Walltime | Simple pendulum DAE (index-3)."""

import functools
import statistics
from collections.abc import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import tqdm

from probdiffeq import ivpsolve, probdiffeq
from probdiffeq.util import benchmark_util

# Fail this notebook on NaN detection (to catch those in the CI)
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_enable_x64", True)


def main(start=3.0, stop=12.0, step=0.5, repeats=2, time_span=(0.0, 1.0)) -> None:
    """Run the script."""
    jax.config.update("jax_enable_x64", True)

    tolerances = benchmark_util.setup_tolerances(start=start, stop=stop, step=step)
    timeit_fun = benchmark_util.setup_timeit(repeats=repeats)

    algorithms = {
        "index-3 | Jet(4)": solver_index3(num_derivatives=4, time_span=time_span),
        "index-1 | Jet(4)": solver_index1(num_derivatives=4, time_span=time_span),
    }

    reference = solver_index1(num_derivatives=5, time_span=time_span)(
        0.1 * tolerances[-1]
    )
    rmse_fun = rmse_relative(reference)

    results = {}
    pbar = tqdm.tqdm(algorithms.items())
    for label, algo in pbar:
        pbar.set_description(label)
        param_to_wp = workprec(algo, precision_fun=rmse_fun, work_fun=timeit_fun)
        results[label] = param_to_wp(tolerances)

    _fig, ax = plt.subplots(ncols=2, figsize=(13, 5))

    for label, wp in results.items():
        wdw = 3

        precision, y = wp["precision"], wp["work_mean"]
        x, _ = precision.T
        x = jnp.exp(jnp.convolve(jnp.log(x), jnp.ones((wdw,)) / wdw, mode="valid"))
        y = jnp.exp(jnp.convolve(jnp.log(y), jnp.ones((wdw,)) / wdw, mode="valid"))
        ax[0].loglog(x, y, label=label)

        x, violation = precision.T
        eps = jnp.finfo(x.dtype).eps
        ax[1].loglog(tolerances, eps + jnp.abs(violation), "-", label=label)

    ax[0].set_title("Work-precision diagram")
    ax[0].set_xlabel("Precision (relative RMSE)")
    ax[0].set_ylabel("Work (avg. wall time)")
    ax[0].grid(linestyle="dotted", which="both")
    ax[0].legend(fontsize="small")

    ax[1].set_title("Constraint violation")
    ax[1].set_xlabel("Tolerance (user input)")
    ax[1].set_ylabel("Position constraint violation $|x^2 + y^2 - 1|$")
    ax[1].grid(linestyle="dotted", which="both")
    ax[1].legend(fontsize="small")

    plt.tight_layout()
    plt.show()


def solver_index1(*, num_derivatives: int, time_span) -> Callable:
    """Construct a first-order DAE solver for the pendulum (index-1)."""

    @functools.partial(probdiffeq.jet_lift, lift_by=0)
    @probdiffeq.residual_state_velocity
    def dynamics(u, du, /, *, t):
        del t
        x, y, vx, vy, lam = u
        dx, dy, dvx, dvy, _ = du
        return jnp.array([dx - vx, dy - vy, dvx + x * lam, dvy + y * lam + 9.81])

    @functools.partial(probdiffeq.jet_lift, lift_by=0)
    @probdiffeq.residual_state
    def constraint(u, /, *, t):
        del t
        x, y, vx, vy, lam = u
        # dx, dy, dvx, dvy, dlam = du
        # gdot = x * vx + y * vy
        g = x**2 + y**2 - 1.0
        # gddot = dvx * x + dvy * y + vx**2 + vy**2
        return jnp.array([g])

    residual = probdiffeq.residual_from_stack(dynamics, constraint)

    x0, y0, vx0, vy0, lam0 = _y0_first_order_dae()

    assert jnp.isclose(x0**2 + y0**2 - 1.0, 0.0)  # position constraint
    assert jnp.isclose(x0 * vx0 + y0 * vy0, 0.0)  # velocity constraint

    @jax.jit
    def param_to_solution(tol):
        t0, t1 = time_span
        inits = _y0_first_order_dae()

        nlstsq = probdiffeq.lstsq_constrained_gauss_newton(tol=1e-12, maxiter=10)
        jetexpand = probdiffeq.jetexpand_residual(nlstsq=nlstsq, num=num_derivatives)
        tcoeffs, _ = jetexpand(residual, (inits,), t=t0)

        ssm = probdiffeq.state_space_model()
        output_scale = jnp.array([1.0, 1.0, 1.0, 1.0, 9.81 + 1.0])
        is_exact = jnp.asarray([True, True, True, True, False])
        init, iwp = probdiffeq.prior_wiener_integrated(
            tcoeffs,
            output_scale=output_scale,
            ssm=ssm,
            is_exact=[is_exact for _ in range(len(tcoeffs))],
        )

        taylor_point = probdiffeq.taylor_point_maximum_a_posteriori(nlstsq=nlstsq)
        jet = probdiffeq.constraint_residual(
            residual, taylor_point=taylor_point, ssm=ssm
        )
        strategy = probdiffeq.strategy_filter()
        solver = probdiffeq.solver_dynamic(strategy=strategy, prior=iwp, constraint=jet)
        error = probdiffeq.error_state_std(constraint=jet, prior=iwp)

        solve = ivpsolve.solve_adaptive_terminal_values(solver=solver, error=error)
        solution = solve(init, t0=t0, t1=t1, atol=1e-3 * tol, rtol=tol)

        xy = solution.u.mean[0][:2]
        vxvy = solution.u.mean[0][2:4]
        return jnp.concatenate([xy, vxvy])

    return param_to_solution


def solver_index3(*, num_derivatives: int, time_span) -> Callable:
    """Construct a first-order DAE solver for the pendulum (index-3)."""

    @functools.partial(probdiffeq.jet_lift, lift_by=0)
    @probdiffeq.residual_state_velocity
    def dynamics(u, du, /, *, t):
        del t
        x, y, vx, vy, lam = u
        dx, dy, dvx, dvy, _ = du
        return jnp.array([dx - vx, dy - vy, dvx + x * lam, dvy + y * lam + 9.81])

    @functools.partial(probdiffeq.jet_lift, lift_by=0)
    @probdiffeq.residual_state
    def constraint(u, /, *, t):
        del t
        return jnp.array([u[0] ** 2 + u[1] ** 2 - 1.0])  # <-- was wrapped in a list []

    residual = probdiffeq.residual_from_stack(dynamics, constraint)

    @jax.jit
    def param_to_solution(tol):
        t0, t1 = time_span
        inits = _y0_first_order_dae()

        nlstsq = probdiffeq.lstsq_constrained_gauss_newton(tol=1e-12, maxiter=10)
        jetexpand = probdiffeq.jetexpand_residual(nlstsq=nlstsq, num=num_derivatives)
        tcoeffs, _ = jetexpand(residual, (inits,), t=t0)

        ssm = probdiffeq.state_space_model()

        output_scale = jnp.array([1.0, 1.0, 1.0, 1.0, 9.81 + 1.0])
        is_exact = jnp.asarray([True, True, True, True, False])
        init, iwp = probdiffeq.prior_wiener_integrated(
            tcoeffs,
            output_scale=output_scale,
            ssm=ssm,
            is_exact=[is_exact for _ in range(len(tcoeffs))],
        )

        taylor_point = probdiffeq.taylor_point_maximum_a_posteriori(nlstsq=nlstsq)
        jet = probdiffeq.constraint_residual(
            residual, taylor_point=taylor_point, ssm=ssm
        )
        strategy = probdiffeq.strategy_filter()

        solver = probdiffeq.solver_dynamic(strategy=strategy, prior=iwp, constraint=jet)
        error = probdiffeq.error_state_std(constraint=jet, prior=iwp)

        solve = ivpsolve.solve_adaptive_terminal_values(solver=solver, error=error)
        solution = solve(init, t0=t0, t1=t1, atol=1e-3 * tol, rtol=tol)

        xy = solution.u.mean[0][:2]
        vxvy = solution.u.mean[0][2:4]
        return jnp.concatenate([xy, vxvy])

    return param_to_solution


@jax.jit
def _y0_first_order_dae():
    """First-order DAE initial state [x, y, vx, vy, λ]."""
    lam0 = 1.0**2 + 0.0**2 - 9.81 * (-1.0)
    return jnp.array([0.0, -1.0, 1.0, 0.0, lam0])


def rmse_relative(expected: jax.Array) -> Callable:
    """Compute the relative RMSE and the constraint violation."""
    expected = jnp.asarray(expected)

    def rmse(received):
        received = jnp.asarray(received)
        error_absolute = jnp.abs(expected - received)
        error_relative = error_absolute / (1e-5 + jnp.abs(expected))
        rmse_val = jnp.linalg.norm(error_relative) / jnp.sqrt(error_relative.size)

        # Constraint violation from the position components
        x, y = received[0], received[1]
        violation = x**2 + y**2 - 1.0
        return rmse_val, violation

    return rmse


def workprec(fun, *, precision_fun: Callable, work_fun: Callable) -> Callable:
    """Turn a parameter-to-solution function into parameter-to-workprecision."""

    def parameter_list_to_workprecision(list_of_args, /):
        works_mean = []
        works_std = []
        precisions = []
        for arg in list_of_args:
            precision = precision_fun(fun(arg).block_until_ready())
            work = work_fun(lambda: fun(arg).block_until_ready())  # noqa: B023

            precisions.append(precision)
            works_mean.append(statistics.mean(work))
            works_std.append(statistics.stdev(work))
        return {
            "work_mean": jnp.asarray(works_mean),
            "work_std": jnp.asarray(works_std),
            "precision": jnp.asarray(precisions),
        }

    return parameter_list_to_workprecision


main()
