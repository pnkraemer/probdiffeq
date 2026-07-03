"""Walltime | Lotka-Volterra."""

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

# High accuracy requires double precision
jax.config.update("jax_enable_x64", True)


def main(start=3.0, stop=12.0, step=1.0, repeats=1) -> None:
    """Run the script."""
    # Simulate once to plot the state
    ts, ys = solve_ivp_once()

    _fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(ts, ys)
    ax.set_title("Lotka-Volterra problem")
    ax.set_xlabel("Time")
    ax.set_ylabel("State")
    plt.tight_layout()
    plt.show()

    # Read configuration from command line
    tolerances = benchmark_util.setup_tolerances(start=start, stop=stop, step=step)
    timeit_fun = benchmark_util.setup_timeit(repeats=repeats)

    # Assemble algorithms
    ts0_iso = solver_probdiffeq(5, constraint_order=0, implementation="isotropic")
    ts0_iso_jet = solver_probdiffeq_jet(
        5, constraint_order=0, implementation="isotropic"
    )
    ts0_bd = solver_probdiffeq(5, constraint_order=0, implementation="blockdiag")
    ts0_bd_jet = solver_probdiffeq_jet(
        5, constraint_order=0, implementation="blockdiag"
    )
    algorithms = {
        r"TS0(5, isotropic)": ts0_iso,
        r"JetTS0(5, isotropic)": ts0_iso_jet,
        r"TS0(5, blockdiag)": ts0_bd,
        r"JetTS0(5, blockdiag)": ts0_bd_jet,
        "SciPy: 'RK45'": solver_scipy(method="RK45"),
    }

    # Compute a reference solution
    reference = solver_scipy(method="BDF")(1e-13)
    precision_fun = benchmark_util.rmse_relative(reference)

    # Compute all work-precision diagrams
    results = {}
    pbar = tqdm.tqdm(algorithms.items())
    for label, algo in pbar:
        pbar.set_description(label)
        param_to_wp = benchmark_util.workprec(
            algo, precision_fun=precision_fun, timeit_fun=timeit_fun
        )
        results[label] = param_to_wp(tolerances)

    _fig, ax = plt.subplots(figsize=(8, 5), dpi=120, constrained_layout=True)
    for label, wp in results.items():
        ax.loglog(wp["precision"], wp["work_mean"], label=label)

    ax.set_ylabel("Work (avg. wall time)")
    ax.set_title("Work-precision diagram")
    ax.set_xlabel("Precision (relative RMSE)")
    ax.grid(linestyle="dotted", which="both")
    ax.legend(fontsize="small")

    plt.show()


def solve_ivp_once():
    """Compute plotting-values for the IVP."""

    def vf_scipy(_t, y):
        """Lotka--Volterra dynamics."""
        dy1 = 0.5 * y[0] - 0.05 * y[0] * y[1]
        dy2 = -0.5 * y[1] + 0.05 * y[0] * y[1]
        return np.asarray([dy1, dy2])

    u0 = jnp.asarray((20.0, 20.0))
    time_span = np.asarray([0.0, 50.0])
    tol = 1e-12
    solution = scipy.integrate.solve_ivp(
        vf_scipy, y0=u0, t_span=time_span, atol=1e-3 * tol, rtol=tol, method="LSODA"
    )
    return solution.t, solution.y.T


def solver_probdiffeq(
    num_derivatives: int, implementation, constraint_order: int
) -> Callable:
    """Construct a solver that wraps ProbDiffEq's solution routines."""

    @probdiffeq.ode
    def vf_probdiffeq(y, /, *, t):  # noqa: ARG001
        """Lotka--Volterra dynamics."""
        dy1 = 0.5 * y[0] - 0.05 * y[0] * y[1]
        dy2 = -0.5 * y[1] + 0.05 * y[0] * y[1]
        return jnp.asarray([dy1, dy2])

    u0 = jnp.asarray((20.0, 20.0))
    t0, t1 = (0.0, 50.0)

    jetexpand = probdiffeq.jetexpand_ode_padded_scan(num=num_derivatives)

    @jax.jit
    def param_to_solution(tol):
        tcoeffs, _ = jetexpand(vf_probdiffeq, (u0,), t=t0)

        ssm = state_space_model(implementation)
        iwp = ssm.prior_wiener_integrated(tcoeffs)
        strategy = probdiffeq.strategy_filter()
        if constraint_order == 0:
            ts = ssm.constraint_ode_ts0(vf_probdiffeq)
        else:
            ts = ssm.constraint_ode_ts1(vf_probdiffeq)
        solver = probdiffeq.solver(strategy=strategy, constraint=ts)
        error = probdiffeq.error_state_std(constraint=ts)

        control = ivpsolve.control_proportional_integral()
        solve = ivpsolve.solve_adaptive_terminal_values(
            error=error, solver=solver, control=control
        )

        # Build a solver
        solution = solve(iwp, t0=t0, t1=t1, atol=1e-2 * tol, rtol=tol)

        # Return the terminal value
        return jax.block_until_ready(solution.u.mean[0])

    def state_space_model(implementation):
        match implementation:
            case "blockdiag":
                return probdiffeq.state_space_model_blockdiag()
            case "dense":
                return probdiffeq.state_space_model_dense()
            case "isotropic":
                return probdiffeq.state_space_model_isotropic()

    return param_to_solution


def solver_probdiffeq_jet(
    num_derivatives: int, implementation, constraint_order: int
) -> Callable:
    """Construct a solver that wraps ProbDiffEq's solution routines."""

    @probdiffeq.ode
    def vf_probdiffeq(y, /, *, t):  # noqa: ARG001
        """Lotka--Volterra dynamics."""
        dy1 = 0.5 * y[0] - 0.05 * y[0] * y[1]
        dy2 = -0.5 * y[1] + 0.05 * y[0] * y[1]
        return jnp.asarray([dy1, dy2])

    u0 = jnp.asarray((20.0, 20.0))
    t0, t1 = (0.0, 50.0)

    vf_probdiffeq = vf_probdiffeq.jet_lift(lift_by=num_derivatives - 1)

    @jax.jit
    def param_to_solution(tol):

        ssm = state_space_model(implementation)
        iwp = ssm.prior_wiener_integrated([u0], diffuse_derivatives=num_derivatives)
        strategy = probdiffeq.strategy_filter()
        if constraint_order == 0:
            ts = ssm.constraint_ode_ts0(vf_probdiffeq)
        else:
            ts = ssm.constraint_ode_ts1(vf_probdiffeq)
        solver = probdiffeq.solver(strategy=strategy, constraint=ts, constraint_init=ts)
        error = probdiffeq.error_state_std(constraint=ts)

        control = ivpsolve.control_proportional_integral()
        solve = ivpsolve.solve_adaptive_terminal_values(
            error=error, solver=solver, control=control
        )

        # Build a solver
        solution = solve(iwp, t0=t0, t1=t1, atol=1e-2 * tol, rtol=tol)

        # Return the terminal value
        return jax.block_until_ready(solution.u.mean[0])

    def state_space_model(implementation):
        match implementation:
            case "blockdiag":
                return probdiffeq.state_space_model_blockdiag()
            case "dense":
                return probdiffeq.state_space_model_dense()
            case "isotropic":
                return probdiffeq.state_space_model_isotropic()

    return param_to_solution


def solver_scipy(*, method: str) -> Callable:
    """Construct a solver that wraps SciPy's solution routines."""

    def vf_scipy(_t, y):
        """Lotka--Volterra dynamics."""
        dy1 = 0.5 * y[0] - 0.05 * y[0] * y[1]
        dy2 = -0.5 * y[1] + 0.05 * y[0] * y[1]
        return np.asarray([dy1, dy2])

    u0 = jnp.asarray((20.0, 20.0))
    time_span = np.asarray([0.0, 50.0])

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
        return jnp.asarray(solution.y[:, -1])

    return param_to_solution


main()
