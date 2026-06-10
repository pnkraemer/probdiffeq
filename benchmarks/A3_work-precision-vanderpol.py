"""Walltime | stiff van-der-Pol."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numba
import numpy as np
import scipy.integrate
import tqdm

from probdiffeq import ivpsolve, probdiffeq
from probdiffeq.util import benchmark_util

# Fail this notebook on NaN detection (to catch those in the CI)
jax.config.update("jax_debug_nans", True)


def main(start=4.0, stop=10.0, step=0.25, repeats=2) -> None:
    """Run the script."""
    # Set up all the configs
    jax.config.update("jax_enable_x64", True)

    # Simulate once to plot the state
    ts, ys = solve_ivp_once()

    _fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(ts, ys)
    ax.set_ylim((-6, 6))
    ax.set_title("Van-der-Pol problem")
    ax.set_xlabel("Time")
    ax.set_ylabel("State")
    plt.tight_layout()
    plt.show()

    # Read configuration from command line
    tolerances = benchmark_util.setup_tolerances(start=start, stop=stop, step=step)
    timeit_fun = benchmark_util.setup_timeit(repeats=repeats)

    # Assemble algorithms
    algorithms = {
        "TS1(4)": solver_probdiffeq(num_derivatives=4),
        "TS1(8)": solver_probdiffeq(num_derivatives=8),
        "SciPy('LSODA')": solver_scipy(method="LSODA"),
    }

    # Compute a reference solution
    reference = solver_scipy(method="LSODA")(0.1 * tolerances[-1])
    precision_fun = benchmark_util.rmse_absolute(reference)

    # Compute all work-precision diagrams
    results = {}
    pbar = tqdm.tqdm(algorithms.items())
    for label, algo in pbar:
        pbar.set_description(label)
        param_to_wp = benchmark_util.workprec(
            algo, precision_fun=precision_fun, timeit_fun=timeit_fun
        )
        results[label] = param_to_wp(tolerances)

    _fig, ax = plt.subplots(figsize=(8, 5))

    for label, wp in results.items():
        wdw = 3  # window
        x, y = wp["precision"], wp["work_mean"]
        x = jnp.exp(jnp.convolve(jnp.log(x), jnp.ones((wdw,)) / wdw, mode="valid"))
        y = jnp.exp(jnp.convolve(jnp.log(y), jnp.ones((wdw,)) / wdw, mode="valid"))
        ax.loglog(x, y, label=label)

    ax.set_title("Work-precision diagram")
    ax.set_xlabel("Precision (relative RMSE)")
    ax.set_ylabel("Work (avg. wall time)")
    ax.grid(linestyle="dotted", which="both")
    ax.legend(fontsize="small")

    plt.tight_layout()
    plt.show()


def solve_ivp_once():
    """Compute plotting-values for the IVP."""

    @numba.jit(nopython=True)
    def vf_scipy(_t, u):
        """Van-der-Pol dynamics as a first-order differential equation."""
        return np.asarray([u[1], 1e5 * ((1.0 - u[0] ** 2) * u[1] - u[0])])

    u0 = np.concatenate((np.atleast_1d(2.0), np.atleast_1d(0.0)))
    time_span = np.asarray((0.0, 6.3))

    tol = 1e-12
    solution = scipy.integrate.solve_ivp(
        vf_scipy, y0=u0, t_span=time_span, atol=1e-3 * tol, rtol=tol, method="LSODA"
    )
    return solution.t, solution.y.T


def solver_probdiffeq(*, num_derivatives: int) -> Callable:
    """Construct a solver that wraps ProbDiffEq's solution routines."""

    @probdiffeq.ode_second_order
    def vf_probdiffeq(u, du, *, t):  # noqa: ARG001
        """Van-der-Pol dynamics as a second-order differential equation."""
        return 1e5 * ((1.0 - u**2) * du - u)

    @probdiffeq.residual_position_velocity_acceleration
    def residual(u, du, ddu, /, *, t):
        """Evaluate a residual to solve the 2nd-order problem directly."""
        return ddu - vf_probdiffeq(u, du, t=t)

    t0, t1 = 0.0, 3.0
    u0, du0 = (jnp.asarray(2.0), jnp.asarray(0.0))
    t0, t1 = (0.0, 6.3)

    @jax.jit
    def param_to_solution(tol):
        # Build a solver
        jetexpand = probdiffeq.jetexpand_ode_padded_scan(num=num_derivatives - 1)
        tcoeffs, _ = jetexpand(vf_probdiffeq, (u0, du0), t=t0)

        ssm = probdiffeq.state_space_model_dense()
        init, iwp = ssm.prior_wiener_integrated(tcoeffs)
        ts = ssm.constraint_residual(residual)
        strategy = probdiffeq.strategy_filter()

        solver = probdiffeq.solver_dynamic(strategy=strategy, prior=iwp, constraint=ts)
        error = probdiffeq.error_state_std(constraint=ts, prior=iwp)

        control = ivpsolve.control_proportional_integral()
        solve = ivpsolve.solve_adaptive_terminal_values(
            solver=solver, error=error, control=control
        )
        solution = solve(init, t0=t0, t1=t1, atol=1e-3 * tol, rtol=tol)
        return jax.block_until_ready(solution.u.mean[0])

    return param_to_solution


def solver_scipy(method: str) -> Callable:
    """Construct a solver that wraps SciPy's solution routines."""

    @numba.jit(nopython=True)
    def vf_scipy(_t, u):
        """Van-der-Pol dynamics as a first-order differential equation."""
        return np.asarray([u[1], 1e5 * ((1.0 - u[0] ** 2) * u[1] - u[0])])

    u0 = np.concatenate((np.atleast_1d(2.0), np.atleast_1d(0.0)))
    time_span = np.asarray((0.0, 6.3))

    def param_to_solution(tol):
        solution = scipy.integrate.solve_ivp(
            vf_scipy,
            y0=u0,
            t_span=time_span,
            t_eval=time_span,
            atol=1e-3 * tol,
            rtol=tol,
            method=method,
        )
        return jnp.asarray(solution.y[0, -1])

    return param_to_solution


main()
