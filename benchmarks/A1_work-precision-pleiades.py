"""Walltime | Pleiades."""

from collections.abc import Callable

import diffrax
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


def main(start=3.0, stop=11.0, step=1.0, repeats=1) -> None:
    """Run the script."""
    # Set up all the configs
    jax.config.update("jax_enable_x64", True)

    # Simulate once to plot the state
    _ts, ys = solve_ivp_once()

    _fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(ys[:, :7], ys[:, 7:14], linestyle="solid", marker="None")
    ax.plot(ys[0, :7], ys[0, 7:14], linestyle="None", marker=".", markersize=4)
    ax.plot(ys[-1, :7], ys[-1, 7:14], linestyle="None", marker="*", markersize=8)

    ax.set_title("Pleiades problem")
    ax.set_xlabel("Time")
    ax.set_ylabel("State")
    plt.tight_layout()
    plt.show()

    # Read configuration from command line
    tolerances = benchmark_util.setup_tolerances(start=start, stop=stop, step=step)

    # Compute a reference solution
    reference = solver_scipy(
        method="LSODA", use_numba=False, precision_fun=lambda x: x
    )(1e-14)
    precision_fun = benchmark_util.rmse_absolute(reference)

    # Assemble algorithms
    algorithms = {
        r"ProbDiffEq: TS0($5$)": solver_probdiffeq(
            num_derivatives=5, precision_fun=precision_fun
        ),
        r"ProbDiffEq: TS0($8$)": solver_probdiffeq(
            num_derivatives=8, precision_fun=precision_fun
        ),
        "SciPy: 'RK45'": solver_scipy(
            method="RK45", use_numba=False, precision_fun=precision_fun
        ),
        "SciPy: 'DOP853'": solver_scipy(
            method="DOP853", use_numba=False, precision_fun=precision_fun
        ),
        "SciPy: 'RK45' (+numba)": solver_scipy(
            method="RK45", use_numba=True, precision_fun=precision_fun
        ),
        "SciPy: 'DOP853' (+numba)": solver_scipy(
            method="DOP853", use_numba=True, precision_fun=precision_fun
        ),
        "Diffrax: Tsit5()": solver_diffrax(
            solver=diffrax.Tsit5(), precision_fun=precision_fun
        ),
        "Diffrax: Dopri8()": solver_diffrax(
            solver=diffrax.Dopri8(), precision_fun=precision_fun
        ),
    }

    # Compute all work-precision diagrams
    _fig, ax = plt.subplots(figsize=(8, 3))
    pbar = tqdm.tqdm(algorithms.items())
    for label, algo in pbar:
        pbar.set_description(label)
        param_to_wp = benchmark_util.workprec(algo, num_timing_calls=repeats)
        wp = param_to_wp(tolerances)
        ax.loglog(wp.precision.mean(axis=-1), wp.work.mean(axis=-1), ".-", label=label)

    ax.set_title("Work-precision diagram")
    ax.set_xlabel("Precision (relative RMSE)")
    ax.set_ylabel("Work (avg. wall time)")
    ax.grid(linestyle="dotted", which="both")
    ax.legend(
        fontsize="small", loc="center left", frameon=False, bbox_to_anchor=(1, 0.5)
    )

    plt.tight_layout()
    plt.show()


def solve_ivp_once():
    """Compute plotting-values for the IVP."""
    # fmt: off
    u0 = np.asarray(
        [
            3.0,  3.0, -1.0, -3.00, 2.0, -2.00,  2.0,
            3.0, -3.0,  2.0,  0.00, 0.0, -4.00,  4.0,
            0.0,  0.0,  0.0,  0.00, 0.0,  1.75, -1.5,
            0.0,  0.0,  0.0, -1.25, 1.0,  0.00,  0.0,
        ]
    )
    # fmt: on

    def vf_scipy(_t, u):
        """Pleiades problem."""
        x = u[0:7]  # x
        y = u[7:14]  # y
        xi, xj = x[:, None], x[None, :]
        yi, yj = y[:, None], y[None, :]
        rij = ((xi - xj) ** 2 + (yi - yj) ** 2) ** (3 / 2)
        mj = np.arange(1, 8)[None, :]

        # Explicitly avoid dividing by zero for scipy's solver
        # The JAX solvers divide by zero and turn the NaNs to zeros.
        rij = np.where(rij == 0.0, 1.0, rij)
        ddx = np.sum((mj * (xj - xi) / rij), axis=1)
        ddy = np.sum((mj * (yj - yi) / rij), axis=1)
        return np.concatenate((u[14:21], u[21:28], ddx, ddy))

    time_span = np.asarray([0.0, 3.0])

    tol = 1e-12
    solution = scipy.integrate.solve_ivp(
        vf_scipy, y0=u0, t_span=time_span, atol=1e-3 * tol, rtol=tol, method="LSODA"
    )

    return solution.t, solution.y.T


def solver_probdiffeq(*, num_derivatives: int, precision_fun) -> Callable:
    """Construct a solver that wraps ProbDiffEq's solution routines."""
    # fmt: off
    u0 = jnp.asarray(
        [
            3.0,  3.0, -1.0, -3.00, 2.0, -2.00,  2.0,
            3.0, -3.0,  2.0,  0.00, 0.0, -4.00,  4.0,
        ]
    )
    du0 = jnp.asarray(
        [
            0.0,  0.0,  0.0,  0.00, 0.0,  1.75, -1.5,
            0.0,  0.0,  0.0, -1.25, 1.0,  0.00,  0.0,
        ]
    )
    # fmt: on

    @probdiffeq.ode_order_two
    def vf_probdiffeq(u, du, /, *, t):  # noqa: ARG001
        """Pleiades problem."""
        x = u[0:7]  # x
        y = u[7:14]  # y
        xi, xj = x[:, None], x[None, :]
        yi, yj = y[:, None], y[None, :]
        rij = ((xi - xj) ** 2 + (yi - yj) ** 2) ** (3 / 2)
        mj = jnp.arange(1, 8)[None, :]
        ddx = jnp.sum(jnp.nan_to_num(mj * (xj - xi) / rij), axis=1)
        ddy = jnp.sum(jnp.nan_to_num(mj * (yj - yi) / rij), axis=1)
        return jnp.concatenate((ddx, ddy))

    t0, t1 = 0.0, 3.0

    @jax.jit
    def param_to_solution(tol):
        # Build a solver
        jetexpand = probdiffeq.jetexpand_ode_unroll(num=num_derivatives - 1)
        tcoeffs, _ = jetexpand(vf_probdiffeq, (u0, du0), t=t0)

        ssm = probdiffeq.state_space_model_isotropic()
        iwp = ssm.prior_wiener_integrated(tcoeffs)
        ts = ssm.constraint_ode_ts0(vf_probdiffeq)
        strategy = probdiffeq.strategy_filter()
        solver = probdiffeq.solver_dynamic(strategy=strategy, constraint=ts)
        error = probdiffeq.error_residual_std(constraint=ts)

        control = ivpsolve.control_proportional_integral()
        solve = ivpsolve.solve_adaptive_terminal_values(
            solver=solver, error=error, control=control, clip_dt=True
        )

        # Solve
        dt0 = ivpsolve.dt0(vf_probdiffeq, (u0, du0), t=t0)
        solution = solve(iwp, t0=t0, t1=t1, dt0=dt0, atol=1e-3 * tol, rtol=tol)

        # Return the terminal value
        return precision_fun(solution.u.mean[0])

    return param_to_solution


def solver_diffrax(*, solver, precision_fun) -> Callable:
    """Construct a solver that wraps Diffrax' solution routines."""
    # fmt: off
    u0 = jnp.asarray(
        [
            3.0,  3.0, -1.0, -3.00, 2.0, -2.00,  2.0,
            3.0, -3.0,  2.0,  0.00, 0.0, -4.00,  4.0,
            0.0,  0.0,  0.0,  0.00, 0.0,  1.75, -1.5,
            0.0,  0.0,  0.0, -1.25, 1.0,  0.00,  0.0,
        ]
    )
    # fmt: on

    @diffrax.ODETerm
    @jax.jit
    def vf_diffrax(_t, u, _args):
        """Pleiades problem."""
        x = u[0:7]  # x
        y = u[7:14]  # y
        xi, xj = x[:, None], x[None, :]
        yi, yj = y[:, None], y[None, :]
        rij = ((xi - xj) ** 2 + (yi - yj) ** 2) ** (3 / 2)
        mj = jnp.arange(1, 8)[None, :]
        ddx = jnp.sum(jnp.nan_to_num(mj * (xj - xi) / rij), axis=1)
        ddy = jnp.sum(jnp.nan_to_num(mj * (yj - yi) / rij), axis=1)
        return jnp.concatenate((u[14:21], u[21:28], ddx, ddy))

    t0, t1 = 0.0, 3.0

    @jax.jit
    def param_to_solution(tol):
        controller = diffrax.PIDController(atol=1e-3 * tol, rtol=tol)
        saveat = diffrax.SaveAt(t0=False, t1=True, ts=None)
        solution = diffrax.diffeqsolve(
            vf_diffrax,
            y0=u0,
            t0=t0,
            t1=t1,
            saveat=saveat,
            stepsize_controller=controller,
            dt0=None,
            max_steps=10_000,
            solver=solver,
        )
        return precision_fun(solution.ys[0, :14])

    return param_to_solution


def solver_scipy(*, method: str, use_numba: bool, precision_fun) -> Callable:
    """Construct a solver that wraps SciPy's solution routines."""
    # fmt: off
    u0 = np.asarray(
        [
            3.0,  3.0, -1.0, -3.00, 2.0, -2.00,  2.0,
            3.0, -3.0,  2.0,  0.00, 0.0, -4.00,  4.0,
            0.0,  0.0,  0.0,  0.00, 0.0,  1.75, -1.5,
            0.0,  0.0,  0.0, -1.25, 1.0,  0.00,  0.0,
        ]
    )
    # fmt: on

    def vf_scipy(_t, u):
        """Pleiades problem."""
        x = u[0:7]  # x
        y = u[7:14]  # y
        xi, xj = x[:, None], x[None, :]
        yi, yj = y[:, None], y[None, :]
        rij = ((xi - xj) ** 2 + (yi - yj) ** 2) ** (3 / 2)
        mj = np.arange(1, 8)[None, :]

        # Explicitly avoid dividing by zero for scipy's solver
        # The JAX solvers divide by zero and turn the NaNs to zeros.
        rij = np.where(rij == 0.0, 1.0, rij)
        ddx = np.sum((mj * (xj - xi) / rij), axis=1)
        ddy = np.sum((mj * (yj - yi) / rij), axis=1)
        return np.concatenate((u[14:21], u[21:28], ddx, ddy))

    if use_numba:
        vf_scipy = numba.jit(nopython=True)(vf_scipy)

    time_span = np.asarray([0.0, 3.0])

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
        return precision_fun(jnp.asarray(solution.y[:14, -1]))

    return param_to_solution


main()
