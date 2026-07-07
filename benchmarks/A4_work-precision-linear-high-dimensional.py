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

DIMENSION = 100
"""The dimension of the ODE problem.

Large enough to exclude all O(d^3) methods.
"""

SCALE = 1.5
"""The 'scale' in u' = scale *  u."""


def main(start=3.0, stop=10.0, step=0.5, repeats=2) -> None:
    """Run the script."""
    # Double precision for high precision simulations
    jax.config.update("jax_enable_x64", True)

    # Read configuration from command line
    tolerances = benchmark_util.setup_tolerances(start=start, stop=stop, step=step)

    # Compute a reference solution
    reference = solver_scipy(method="LSODA", precision_fun=lambda x: x)(1e-13)
    precision_fun = benchmark_util.rmse_relative(reference)

    # Assemble algorithms
    ts0_iso = solver_probdiffeq(
        4, constraint_order=0, implementation="isotropic", precision_fun=precision_fun
    )
    ts0_bd = solver_probdiffeq(
        4, constraint_order=0, implementation="blockdiag", precision_fun=precision_fun
    )
    ts1_bd = solver_probdiffeq(
        4, constraint_order=1, implementation="blockdiag", precision_fun=precision_fun
    )
    ts1_mf = solver_probdiffeq(
        4, constraint_order=1, implementation="matfree", precision_fun=precision_fun
    )
    algorithms = {
        r"TS1, matfree": ts1_mf,
        r"TS1, blockdiag": ts1_bd,
        r"TS0, blockdiag": ts0_bd,
        r"TS0, isotropic": ts0_iso,
        "Diffrax: Tsit5()": solver_diffrax(
            solver=diffrax.Tsit5(), precision_fun=precision_fun
        ),
        "SciPy: 'RK45'": solver_scipy(method="RK45", precision_fun=precision_fun),
    }

    # Compute all work-precision diagrams
    _fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    pbar = tqdm.tqdm(algorithms.items())
    for label, algo in pbar:
        pbar.set_description(label)
        param_to_wp = benchmark_util.workprec(algo, num_timing_calls=repeats)
        wp = param_to_wp(tolerances)

        linestyle = (
            "dashed" if "iffrax" in label.lower() or "ipy" in label.lower() else "solid"
        )
        ax.loglog(
            wp.precision.mean(axis=-1),
            wp.work.mean(axis=-1),
            marker=".",
            label=label,
            linestyle=linestyle,
        )

    ax.set_title("Work-precision diagram")
    ax.set_xlabel("Precision (relative RMSE)")
    ax.set_ylabel("Work (avg. wall time)")
    ax.grid(linestyle="dotted", which="both")
    ax.legend(loc="center left", frameon=False, bbox_to_anchor=(1, 0.5))

    plt.show()


def solve_ivp_once():
    """Compute plotting-values for the IVP."""

    def vf_scipy(_t, y):
        """Lotka--Volterra dynamics."""
        return SCALE * y

    u0 = np.ones((DIMENSION,))
    time_span = np.asarray([0.0, 1.0])
    tol = 1e-12
    solution = scipy.integrate.solve_ivp(
        vf_scipy, y0=u0, t_span=time_span, atol=1e-3 * tol, rtol=tol, method="LSODA"
    )
    return solution.t, solution.y.T


def solver_probdiffeq(
    num_derivatives: int, implementation, constraint_order: int, precision_fun
) -> Callable:
    """Construct a solver that wraps ProbDiffEq's solution routines."""

    @probdiffeq.ode
    def vf_probdiffeq(y, /, *, t):
        """Lotka--Volterra dynamics."""
        del t
        return SCALE * y

    u0 = jnp.ones((DIMENSION,), dtype=float)
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
        error = probdiffeq.error_state_std(constraint=ts)
        solve = ivpsolve.solve_adaptive_terminal_values(error=error, solver=solver)

        # Build a solver
        solution = solve(iwp, t0=t0, t1=t1, atol=1e-3 * tol, rtol=tol)

        # Return the terminal value
        return precision_fun(solution.u.mean[0])

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
                num_ensembles = (num_derivatives + 1) * 2
                return probdiffeq.state_space_model_matfree(
                    key=key, num_ensembles=num_ensembles
                )
            case _:
                raise ValueError

    return param_to_solution


def solver_diffrax(*, solver, precision_fun) -> Callable:
    """Construct a solver that wraps Diffrax' solution routines."""

    @diffrax.ODETerm
    @jax.jit
    def vf_diffrax(_t, y, _args):
        """Lotka--Volterra dynamics."""
        return SCALE * y

    u0 = jnp.ones((DIMENSION,))
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
        return precision_fun(solution.ys[0, :])

    return param_to_solution


def solver_scipy(*, method: str, precision_fun) -> Callable:
    """Construct a solver that wraps SciPy's solution routines."""

    def vf_scipy(_t, y):
        """Lotka--Volterra dynamics."""
        return SCALE * y

    u0 = np.ones((DIMENSION,))
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
        return precision_fun(jnp.asarray(solution.y[:, -1]))

    return param_to_solution


main()
