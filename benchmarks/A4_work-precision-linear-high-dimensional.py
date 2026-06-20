"""Walltime | Linear ODE with many components."""

from collections.abc import Callable

import diffrax
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


def main(start=3.0, stop=4.0, step=1.0, repeats=2) -> None:
    """Run the script."""
    # Set up all the configs
    jax.config.update("jax_enable_x64", True)

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
    ts0_bd = solver_probdiffeq(5, constraint_order=0, implementation="blockdiag")
    ts1_bd = solver_probdiffeq(5, constraint_order=1, implementation="blockdiag")
    ts1_mf = solver_probdiffeq(5, constraint_order=1, implementation="matfree")
    algorithms = {
        r"ProbDiffEq: TS1($5$, matfree)": ts1_mf
        # r"ProbDiffEq: TS0($5$, isotropic)": ts0_iso,
        # r"ProbDiffEq: TS0($5$, blockdiag)": ts0_bd,
        # r"ProbDiffEq: TS1($5$, blockdiag)": ts1_bd,
        # "Diffrax: Tsit5()": solver_diffrax(solver=diffrax.Tsit5()),
        # "SciPy: 'RK45'": solver_scipy(method="RK45"),
    }

    # Compute a reference solution
    reference = solver_scipy(method="LSODA")(1e-13)
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

    _fig, ax = plt.subplots(figsize=(7, 3))
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

    def vf_scipy(_t, y):
        """Lotka--Volterra dynamics."""
        return 1.01 * y

    u0 = np.ones((11,))
    time_span = np.asarray([0.0, 1.0])
    tol = 1e-12
    solution = scipy.integrate.solve_ivp(
        vf_scipy, y0=u0, t_span=time_span, atol=1e-3 * tol, rtol=tol, method="LSODA"
    )
    return solution.t, solution.y.T


def solver_probdiffeq(
    num_derivatives: int, implementation, constraint_order: int
) -> Callable:
    """Construct a solver that wraps ProbDiffEq's solution routines."""
    # raise RuntimeError(
    #     "Understand why in the two branches, one solver works well and the other doesnt. Then make the less invasive branch work."
    # )

    @probdiffeq.ode
    def vf_probdiffeq(y, /, *, t):
        """Lotka--Volterra dynamics."""
        del t
        return 1.01 * y

    u0 = jnp.ones((11,), dtype=float)
    t0, t1 = (0.0, 1.0)

    @jax.jit
    def param_to_solution(tol):
        jetexpand = probdiffeq.jetexpand_ode_unroll(num=num_derivatives)
        tcoeffs, _ = jetexpand(vf_probdiffeq, (u0,), t=t0)

        ssm = state_space_model(implementation)

        iwp = ssm.prior_wiener_integrated(tcoeffs)
        strategy = probdiffeq.strategy_filter()
        if constraint_order == 0:
            ts = ssm.constraint_ode_ts0(vf_probdiffeq)
        elif constraint_order == 1:
            ts = ssm.constraint_ode_ts1(vf_probdiffeq)
        else:
            raise ValueError

        solver = probdiffeq.solver(strategy=strategy, constraint=ts)
        error = probdiffeq.error_residual_std(constraint=ts)

        control = ivpsolve.control_proportional_integral()
        solve = ivpsolve.solve_adaptive_terminal_values(
            error=error, solver=solver, control=control
        )
        dt0 = ivpsolve.dt0(vf_probdiffeq, (u0,), t=t0)

        # Build a solver
        solution = solve(iwp, t0=t0, t1=t1, dt0=dt0, atol=1e-2 * tol, rtol=tol)

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
            case "matfree":
                key = jax.random.PRNGKey(1)
                num_probes = (num_derivatives + 1) * 2
                return probdiffeq.state_space_model_matfree(
                    key=key, num_probes=num_probes
                )
            case _:
                raise ValueError

    return param_to_solution


def solver_diffrax(*, solver) -> Callable:
    """Construct a solver that wraps Diffrax' solution routines."""

    @diffrax.ODETerm
    @jax.jit
    def vf_diffrax(_t, y, _args):
        """Lotka--Volterra dynamics."""
        return 1.01 * y

    u0 = jnp.ones((11,))
    t0, t1 = (0.0, 1.0)

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
        return jax.block_until_ready(solution.ys[0, :])

    return param_to_solution


def solver_scipy(*, method: str) -> Callable:
    """Construct a solver that wraps SciPy's solution routines."""

    def vf_scipy(_t, y):
        """Lotka--Volterra dynamics."""
        return 1.01 * y

    u0 = np.ones((11,))
    time_span = np.asarray([0.0, 1.0])

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
