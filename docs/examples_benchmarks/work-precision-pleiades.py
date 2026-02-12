# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # WP diagram: Pleiades

# +
"""Pleiades work-precision diagram."""

import functools
import statistics
import timeit
from collections.abc import Callable

import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numba
import numpy as np
import scipy.integrate
import tqdm

from probdiffeq import ivpsolve, probdiffeq, taylor


def main(start=3.0, stop=11.0, step=1.0, repeats=2, use_diffrax: bool = False):
    """Run the script."""
    # Set up all the configs
    jax.config.update("jax_enable_x64", True)

    # Simulate once to plot the state
    ts, ys = solve_ivp_once()

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(ys[:, :7], ys[:, 7:14], linestyle="solid", marker="None")
    ax.plot(ys[0, :7], ys[0, 7:14], linestyle="None", marker=".", markersize=4)
    ax.plot(ys[-1, :7], ys[-1, 7:14], linestyle="None", marker="*", markersize=8)

    ax.set_title("Pleiades problem")
    ax.set_xlabel("Time")
    ax.set_ylabel("State")
    plt.tight_layout()
    plt.show()

    # Read configuration from command line
    tolerances = setup_tolerances(start=start, stop=stop, step=step)
    timeit_fun = setup_timeit(repeats=repeats)

    # Assemble algorithms
    algorithms = {
        r"ProbDiffEq: TS0($5$)": solver_probdiffeq(
            num_derivatives=5, correction_fun=probdiffeq.correction_ts0
        ),
        r"ProbDiffEq: TS0($8$)": solver_probdiffeq(
            num_derivatives=8, correction_fun=probdiffeq.correction_ts0
        ),
        "SciPy: 'RK45'": solver_scipy(method="RK45", use_numba=False),
        "SciPy: 'DOP853'": solver_scipy(method="DOP853", use_numba=False),
        "SciPy: 'RK45' (+numba)": solver_scipy(method="RK45", use_numba=True),
        "SciPy: 'DOP853' (+numba)": solver_scipy(method="DOP853", use_numba=True),
        "Diffrax: Tsit5()": solver_diffrax(solver=diffrax.Tsit5()),
        "Diffrax: Dopri8()": solver_diffrax(solver=diffrax.Dopri8()),
    }

    if use_diffrax:
        # TODO: this is a temporary fix because Diffrax doesn't work with JAX >= 0.7.0
        # Revisit in the near future.
        algorithms["Diffrax: Kvaerno3()"] = solver_diffrax(solver=diffrax.Kvaerno3())
        algorithms["Diffrax: Kvaerno5()"] = solver_diffrax(solver=diffrax.Kvaerno5())
    else:
        print("\nSkipped Diffrax.\n")

    # Compute a reference solution
    reference = solver_scipy(method="LSODA", use_numba=False)(1e-14)
    precision_fun = rmse_absolute(reference)

    # Compute all work-precision diagrams
    results = {}
    for label, algo in tqdm.tqdm(algorithms.items()):
        param_to_wp = workprec(algo, precision_fun=precision_fun, timeit_fun=timeit_fun)
        results[label] = param_to_wp(tolerances)

    fig, ax = plt.subplots(figsize=(7, 3))
    for label, wp in results.items():
        ax.loglog(wp["precision"], wp["work_mean"], label=label)

    ax.set_title("Work-precision diagram")
    ax.set_xlabel("Precision (relative RMSE)")
    ax.set_ylabel("Work (avg. wall time)")
    ax.grid(linestyle="dotted", which="both")
    ax.legend(fontsize="small", loc="center left", bbox_to_anchor=(1, 0.5))

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


def setup_tolerances(*, start: float, stop: float, step: float) -> jax.Array:
    """Choose vector of tolerances from the command-line arguments."""
    return 0.1 ** jnp.arange(start, stop, step=step)


def setup_timeit(*, repeats: int) -> Callable:
    """Construct a timeit-function from the command-line arguments."""

    def timer(fun, /):
        return list(timeit.repeat(fun, number=1, repeat=repeats))

    return timer


def solver_probdiffeq(*, num_derivatives: int, correction_fun) -> Callable:
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

    @jax.jit
    def vf_probdiffeq(u, du, *, t):  # noqa: ARG001
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

    # Build a solver
    vf_auto = functools.partial(vf_probdiffeq, t=t0)
    tcoeffs = taylor.odejet_padded_scan(vf_auto, (u0, du0), num=num_derivatives - 1)

    init, ibm, ssm = probdiffeq.prior_wiener_integrated(tcoeffs, ssm_fact="isotropic")
    ts0_or_ts1 = correction_fun(vf_probdiffeq, ssm=ssm, ode_order=2)
    strategy = probdiffeq.strategy_filter(ssm=ssm)
    solver = probdiffeq.solver_dynamic(
        strategy, prior=ibm, correction=ts0_or_ts1, ssm=ssm
    )
    errorest = probdiffeq.errorest_schober_bosch(
        correction=ts0_or_ts1, prior=ibm, ssm=ssm
    )

    control = ivpsolve.control_proportional_integral()
    solve = ivpsolve.solve_adaptive_terminal_values(
        solver=solver, errorest=errorest, control=control
    )

    # Solve
    dt0 = ivpsolve.dt0(vf_auto, (u0, du0))

    @jax.jit
    def param_to_solution(tol):
        solution = solve(init, t0=t0, t1=t1, dt0=dt0, atol=1e-3 * tol, rtol=tol)

        # Return the terminal value
        return jax.block_until_ready(solution.u.mean[0])

    return param_to_solution


def solver_diffrax(*, solver) -> Callable:
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
        return jax.block_until_ready(solution.ys[0, :14])

    return param_to_solution


def solver_scipy(*, method: str, use_numba: bool) -> Callable:
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
        return jnp.asarray(solution.y[:14, -1])

    return param_to_solution


def rmse_absolute(expected: jax.Array) -> Callable:
    """Compute the absolute RMSE."""
    expected = jnp.asarray(expected)

    def rmse(received):
        received = jnp.asarray(received)
        error_absolute = jnp.abs(expected - received)
        return jnp.linalg.norm(error_absolute) / jnp.sqrt(error_absolute.size)

    return rmse


def workprec(fun, *, precision_fun: Callable, timeit_fun: Callable) -> Callable:
    """Turn a parameter-to-solution function into parameter-to-workprecision."""

    def parameter_list_to_workprecision(list_of_args, /):
        works_mean = []
        works_std = []
        precisions = []
        for arg in list_of_args:
            precision = precision_fun(fun(arg).block_until_ready())
            times = timeit_fun(lambda: fun(arg).block_until_ready())  # noqa: B023

            precisions.append(precision)
            works_mean.append(statistics.mean(times))
            works_std.append(statistics.stdev(times))
        return {
            "work_mean": jnp.asarray(works_mean),
            "work_std": jnp.asarray(works_std),
            "precision": jnp.asarray(precisions),
        }

    return parameter_list_to_workprecision


main()
