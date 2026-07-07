"""Walltime | Burgers PDE."""

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

N = 50
"""Number of spatial grid points.

Small enough to include dense methods,
but large enough to penalise their O(d^3) complexity.
"""

NU = 0.01
"""Diffusion coefficient.

The larger, the stiffer the problem.
"""


def main(start=3.0, stop=8.0, step=1.0, repeats=1) -> None:
    """Run the script."""
    jax.config.update("jax_enable_x64", True)

    # Visualise the dynamics
    ts, ys = solve_ivp_once()

    x = np.linspace(0.0, 1.0, N + 1, endpoint=True)[1:-1]
    _fig, ax = plt.subplots(figsize=(5, 3))
    pcm = ax.pcolormesh(x, ts, ys, cmap="coolwarm", vmin=-0.5, vmax=0.5)
    _fig.colorbar(pcm, ax=ax)
    ax.set_title("Burgers PDE")
    ax.set_xlabel("Space")
    ax.set_ylabel("Time")
    plt.tight_layout()
    plt.show()

    # Read configuration from command line
    tolerances = 0.1 ** jnp.arange(start, stop, step=step)

    # Compute a reference solution
    reference = solver_scipy(method="LSODA", precision_fun=lambda x: x)(1e-13)
    precision_fun = benchmark_util.rmse_absolute(reference)

    # Assemble algorithms
    algorithms = {
        r"TS1($3$, dense)": solver_dense(
            num_derivatives=3, precision_fun=precision_fun
        ),
        r"TS1($3$, blockdiag)": solver_blockdiag(
            num_derivatives=3, precision_fun=precision_fun
        ),
        r"TS1($3$, matfree)": solver_matfree(
            num_derivatives=3, precision_fun=precision_fun
        ),
    }

    # Compute all work-precision diagrams
    _fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    pbar = tqdm.tqdm(algorithms.items())
    for label, algo in pbar:
        pbar.set_description(label)
        param_to_wp = benchmark_util.workprec(algo, num_timing_calls=repeats)
        wp = param_to_wp(tolerances)

        ax.loglog(wp.precision, wp.work.mean(axis=-1), ".-", label=label)

    ax.set_title("Work-precision diagram")
    ax.set_xlabel("Precision (absolute RMSE)")
    ax.set_ylabel("Work (avg. wall time)")
    ax.grid(linestyle="dotted", which="both")
    ax.legend(loc="center left", frameon=False, bbox_to_anchor=(1, 0.5))
    plt.show()


def solve_ivp_once():
    """Compute plotting values for the Burgers PDE."""

    def vf_scipy(_t, u):
        """Viscous Burgers equation, zero Dirichlet BC, conservative advection."""
        dx = 1.0 / N
        u_bc = np.pad(u, 1)  # zero Dirichlet ghosts
        u_left = u_bc[:-2]
        u_right = u_bc[2:]
        flux = u_bc**2 / 2.0

        fluxterm = (flux[2:] - flux[:-2]) / (2.0 * dx)
        laplacian = (u_right - 2.0 * u + u_left) / dx**2
        return -fluxterm + NU * laplacian

    x = np.linspace(0.0, 1.0, N + 1, endpoint=True)[1:-1]
    u0 = jnp.sin(3 * jnp.pi * x) ** 3 * (1 - x) ** 1.5
    time_span = np.asarray([0.0, 1.0])
    t_eval = np.linspace(0.0, 1.0, 200)

    tol = 1e-9
    solution = scipy.integrate.solve_ivp(
        vf_scipy,
        y0=u0,
        t_span=time_span,
        t_eval=t_eval,
        atol=1e-3 * tol,
        rtol=tol,
        method="LSODA",
    )
    return solution.t, solution.y.T


def solver_blockdiag(*, num_derivatives: int, precision_fun) -> Callable:
    """Construct a solver that wraps ProbDiffEq's block-diagonal routines."""

    @probdiffeq.ode
    def vf(u, /, *, t):  # noqa: ARG001
        """Viscous Burgers equation."""
        dx = 1.0 / N
        u_bc = jnp.pad(u, 1)  # zero Dirichlet ghosts
        u_left = u_bc[:-2]
        u_right = u_bc[2:]
        flux = u_bc**2 / 2.0

        fluxterm = (flux[2:] - flux[:-2]) / (2.0 * dx)
        laplacian = (u_right - 2.0 * u + u_left) / dx**2
        return -fluxterm + NU * laplacian

    x = jnp.linspace(0.0, 1.0, N + 1, endpoint=True)[1:-1]
    u0 = jnp.sin(3 * jnp.pi * x) ** 3 * (1 - x) ** 1.5
    t0, t1 = 0.0, 1.0

    ssm = probdiffeq.state_space_model_blockdiag()

    @jax.jit
    def param_to_solution(tol):
        jetexpand = probdiffeq.jetexpand_ode_unroll(num=num_derivatives)
        tcoeffs, _ = jetexpand(vf, (u0,), t=t0)

        iwp = ssm.prior_wiener_integrated(tcoeffs)
        ts1 = ssm.constraint_ode_ts1(vf)
        strategy = probdiffeq.strategy_filter()
        solver = probdiffeq.solver(strategy=strategy, constraint=ts1)
        error = probdiffeq.error_state_std(constraint=ts1)
        control = ivpsolve.control_proportional_integral()
        solve_fn = ivpsolve.solve_adaptive_terminal_values(
            solver=solver, error=error, control=control, clip_dt=True
        )

        solution = solve_fn(iwp, t0=t0, t1=t1, atol=1e-3 * tol, rtol=tol)
        return precision_fun(solution.u.mean[0])

    return param_to_solution


def solver_matfree(*, num_derivatives: int, precision_fun) -> Callable:
    """Construct a solver that wraps ProbDiffEq's matrix-free routines."""

    @probdiffeq.ode
    def vf(u, /, *, t):  # noqa: ARG001
        """Viscous Burgers equation."""
        dx = 1.0 / N
        u_bc = jnp.pad(u, 1)  # zero Dirichlet ghosts
        u_left = u_bc[:-2]
        u_right = u_bc[2:]
        flux = u_bc**2 / 2.0

        fluxterm = (flux[2:] - flux[:-2]) / (2.0 * dx)
        laplacian = (u_right - 2.0 * u + u_left) / dx**2
        return -fluxterm + NU * laplacian

    x = jnp.linspace(0.0, 1.0, N + 1, endpoint=True)[1:-1]
    u0 = jnp.sin(3 * jnp.pi * x) ** 3 * (1 - x) ** 1.5
    t0, t1 = 0.0, 1.0

    key = jax.random.PRNGKey(1)
    num_ensembles = (num_derivatives + 1) * 2
    ssm = probdiffeq.state_space_model_matfree(key=key, num_ensembles=num_ensembles)

    @jax.jit
    def param_to_solution(tol):
        jetexpand = probdiffeq.jetexpand_ode_unroll(num=num_derivatives)
        tcoeffs, _ = jetexpand(vf, (u0,), t=t0)

        iwp = ssm.prior_wiener_integrated(tcoeffs)
        ts1 = ssm.constraint_ode_ts1(vf)
        strategy = probdiffeq.strategy_filter()
        solver = probdiffeq.solver(strategy=strategy, constraint=ts1)
        error = probdiffeq.error_state_std(constraint=ts1)
        control = ivpsolve.control_proportional_integral()
        solve_fn = ivpsolve.solve_adaptive_terminal_values(
            solver=solver, error=error, control=control, clip_dt=True
        )

        solution = solve_fn(iwp, t0=t0, t1=t1, atol=1e-3 * tol, rtol=tol)
        return precision_fun(solution.u.mean[0])

    return param_to_solution


def solver_dense(*, num_derivatives: int, precision_fun) -> Callable:
    """Construct a solver that wraps ProbDiffEq's matrix-free routines."""

    @probdiffeq.ode
    def vf(u, /, *, t):  # noqa: ARG001
        """Viscous Burgers equation."""
        dx = 1.0 / N
        u_bc = jnp.pad(u, 1)  # zero Dirichlet ghosts
        u_left = u_bc[:-2]
        u_right = u_bc[2:]
        flux = u_bc**2 / 2.0

        fluxterm = (flux[2:] - flux[:-2]) / (2.0 * dx)
        laplacian = (u_right - 2.0 * u + u_left) / dx**2
        return -fluxterm + NU * laplacian

    x = jnp.linspace(0.0, 1.0, N + 1, endpoint=True)[1:-1]
    u0 = jnp.sin(3 * jnp.pi * x) ** 3 * (1 - x) ** 1.5
    t0, t1 = 0.0, 1.0

    ssm = probdiffeq.state_space_model_dense()

    @jax.jit
    def param_to_solution(tol):
        jetexpand = probdiffeq.jetexpand_ode_unroll(num=num_derivatives)
        tcoeffs, _ = jetexpand(vf, (u0,), t=t0)

        iwp = ssm.prior_wiener_integrated(tcoeffs)
        ts1 = ssm.constraint_ode_ts1(vf)
        strategy = probdiffeq.strategy_filter()
        solver = probdiffeq.solver(strategy=strategy, constraint=ts1)
        error = probdiffeq.error_state_std(constraint=ts1)
        control = ivpsolve.control_proportional_integral()
        solve_fn = ivpsolve.solve_adaptive_terminal_values(
            solver=solver, error=error, control=control, clip_dt=True
        )

        solution = solve_fn(iwp, t0=t0, t1=t1, atol=1e-3 * tol, rtol=tol)

        return precision_fun(solution.u.mean[0])

    return param_to_solution


def solver_scipy(*, method: str, precision_fun) -> Callable:
    """Construct a solver that wraps SciPy's solution routines."""

    def vf_scipy(_t, u):
        """Viscous Burgers equation, zero Dirichlet BC, conservative advection."""
        dx = 1.0 / N
        u_bc = np.pad(u, 1)  # zero Dirichlet ghosts
        u_left = u_bc[:-2]
        u_right = u_bc[2:]
        flux = u_bc**2 / 2.0

        fluxterm = (flux[2:] - flux[:-2]) / (2.0 * dx)
        laplacian = (u_right - 2.0 * u + u_left) / dx**2
        return -fluxterm + NU * laplacian

    x = np.linspace(0.0, 1.0, N + 1, endpoint=True)[1:-1]
    u0 = jnp.sin(3 * jnp.pi * x) ** 3 * (1 - x) ** 1.5
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
