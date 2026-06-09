"""Walltime | Simple pendulum DAE (index-3)."""

import functools
import statistics
from collections.abc import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import tqdm

from probdiffeq import ivpsolve, probdiffeq
from probdiffeq.util import benchmark_util

# Fail this notebook on NaN detection (to catch those in the CI)
jax.config.update("jax_debug_nans", True)

_G = 9.81


def main(start=3.0, stop=10.0, step=0.5, repeats=2, time_span=(0.0, 1.0)) -> None:
    """Run the script."""
    jax.config.update("jax_enable_x64", True)

    # Simulate once to plot the state
    ts, ys = solve_ivp_once(t_span=time_span, tol=1e-10, method="Radau")

    _fig, ax = plt.subplots(nrows=2, figsize=(5, 4), sharex=True)
    ax[0].set_title("Pendulum solution")
    ax[0].plot(ts, ys[:, 0], label="x")
    ax[0].plot(ts, ys[:, 1], label="y")
    ax[0].set_ylabel("Position")
    ax[0].legend(fontsize="small")
    ax[1].plot(ts, ys[:, 2], label="$v_x$")
    ax[1].plot(ts, ys[:, 3], label="$v_y$")
    ax[1].set_ylabel("Velocity")
    ax[1].set_xlabel("Time $t$")
    ax[1].legend(fontsize="small")
    plt.tight_layout()
    plt.show()

    tolerances = benchmark_util.setup_tolerances(start=start, stop=stop, step=step)
    timeit_fun = benchmark_util.setup_timeit(repeats=repeats)

    algorithms = {
        "DAE | Jet(3)": solver_residual(num_derivatives=3, time_span=time_span),
        "DAE | Jet(4)": solver_residual(num_derivatives=4, time_span=time_span),
        "DAE | Jet(5)": solver_residual(num_derivatives=5, time_span=time_span),
    }

    reference = solver_scipy(method="Radau", time_span=time_span)(0.1 * tolerances[-1])
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


def solve_ivp_once(*, t_span, method, tol):
    """Compute a reference solution for plotting."""
    solution = scipy.integrate.solve_ivp(
        _pendulum_vf,
        y0=_y0_first_order(),
        t_span=t_span,
        dense_output=True,
        atol=1e-3 * tol,
        rtol=tol,
        method=method,
    )
    ts = np.linspace(*t_span, num=200)
    return ts, solution.sol(ts).T


def solver_residual(*, num_derivatives: int, time_span) -> Callable:
    """Construct a second-order DAE solver for the pendulum."""

    @functools.partial(probdiffeq.jet_lift, lift_by=num_derivatives - 1)
    @probdiffeq.residual_state_velocity_acceleration
    def dynamics(u, du, ddu, /, *, t):
        del t
        del du
        x, y, lam = u
        ax, ay = ddu[0], ddu[1]
        return jnp.array([ax + x * lam, ay + _G + y * lam])

    @functools.partial(probdiffeq.jet_lift, lift_by=num_derivatives)
    @probdiffeq.residual_state
    def constraint(u, /, *, t):
        del t
        return jnp.array([u[0] ** 2 + u[1] ** 2 - 1.0])

    residual = probdiffeq.residual_from_stack(dynamics, constraint)

    @jax.jit
    def param_to_solution(tol):
        t0, t1 = time_span
        inits = _y0_second_order()

        nlstsq = probdiffeq.lstsq_constrained_gauss_newton(
            maxiter=10, tol=jnp.finfo(inits[0].dtype).eps ** 0.5
        )
        jetexpand = probdiffeq.jetexpand_residual(nlstsq=nlstsq, num=num_derivatives)
        tcoeffs, _ = jetexpand(residual, inits, t=t0)

        ssm = probdiffeq.state_space_model()

        # Scale accounts for λ being ~10x larger than positions
        output_scale = jnp.array([1.0, 1.0, _G + 1.0])
        init, iwp = probdiffeq.prior_wiener_integrated(
            tcoeffs, output_scale=output_scale, ssm=ssm
        )

        taylor_point = probdiffeq.taylor_point_maximum_a_posteriori(nlstsq)
        jet = probdiffeq.constraint_residual(
            residual, taylor_point=taylor_point, ssm=ssm
        )
        strategy = probdiffeq.strategy_filter()

        solver = probdiffeq.solver_dynamic(strategy=strategy, prior=iwp, constraint=jet)
        error = probdiffeq.error_state_std(constraint=jet, prior=iwp)

        solve = ivpsolve.solve_adaptive_terminal_values(solver=solver, error=error)
        solution = solve(init, t0=t0, t1=t1, atol=1e-3 * tol, rtol=tol)

        # Return [x, y, vx, vy] — mean[0] is position, mean[1] is velocity
        xy = solution.u.mean[0][:2]
        vxvy = solution.u.mean[1][:2]
        return jax.block_until_ready(jnp.concatenate([xy, vxvy]))

    return param_to_solution


def solver_scipy(*, method: str, time_span) -> Callable:
    """Construct a reference solver via SciPy (not plotted, for error computation only)."""

    def param_to_solution(tol):
        solution = scipy.integrate.solve_ivp(
            _pendulum_vf,
            y0=_y0_first_order(),
            t_span=time_span,
            t_eval=[time_span[-1]],
            atol=1e-3 * tol,
            rtol=tol,
            method=method,
        )
        return jnp.asarray(solution.y[:, -1])

    return param_to_solution


def _pendulum_vf(t, u):
    """Pendulum ODE with λ eliminated analytically (for SciPy reference)."""
    del t
    x, y, vx, vy = u
    lam = vx**2 + vy**2 - _G * y
    return [vx, vy, -x * lam, -_G - y * lam]


def _y0_first_order():
    """First-order initial state [x, y, vx, vy] for SciPy."""
    return [0.0, -1.0, 1.0, 0.0]


def _y0_second_order():
    """Second-order initial state as (u0, du0) for the probdiffeq DAE solver."""
    lam0 = 1.0**2 + 0.0**2 - _G * (-1.0)  # vx0^2 + vy0^2 - g*y0
    u0 = jnp.array([0.0, -1.0, lam0])
    du0 = jnp.array([1.0, 0.0, 0.0])  # vx0=1, vy0=0, dlam/dt|0 = -3g*vy0 = 0
    return [u0, du0]


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
