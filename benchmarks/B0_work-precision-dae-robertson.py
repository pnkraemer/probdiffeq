"""Walltime | Robertson DAE."""

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


def main(start=3.0, stop=10.0, step=0.5, repeats=2, time_span=(1e-6, 1e5)) -> None:
    """Run the script."""
    # Set up all the configs
    jax.config.update("jax_enable_x64", True)

    # Simulate once to plot the state
    t0, t1 = time_span
    save_at = jnp.exp(jnp.linspace(jnp.log(t0), jnp.log(t1), num=100))
    ts, ys = solve_ivp_once(save_at=save_at, tol=1e-10, method="LSODA")

    _fig, ax = plt.subplots(nrows=3, figsize=(5, 5), sharex=True)
    ax[0].set_title("Robertson solution")
    ax[0].semilogx(ts, 1e-10 + ys[:, 0])
    ax[1].semilogx(ts, 1e-10 + ys[:, 1])
    ax[2].semilogx(ts, 1e-10 + ys[:, 2])

    ax[0].set_ylabel("State $y_1$")
    ax[1].set_ylabel("State $y_2$")
    ax[2].set_ylabel("State $y_3$")
    ax[2].set_xlabel("Time $t$")
    ax[0].set_xlim((t0, t1))
    plt.tight_layout()
    plt.show()

    # Read configuration from command line
    tolerances = benchmark_util.setup_tolerances(start=start, stop=stop, step=step)

    # Compute a reference solution
    reference_solver = solver_scipy(
        method="Radau", time_span=time_span, precision_fun=lambda x: x
    )
    reference = reference_solver(0.1 * tolerances[-1])
    rmse_fun = benchmark_util.rmse_relative(reference)

    def precision_fun(received):
        rmse = rmse_fun(received)
        eps = jnp.finfo(received.dtype).eps
        violation = eps + jnp.abs(np.sum(received) - 1.0)
        return {"rmse": rmse, "constraint_violation": violation}

    # Assemble algorithms
    algorithms = {
        "DAE | Jet(3)": solver_residual(
            num_derivatives=3, time_span=time_span, precision_fun=precision_fun
        ),
        "DAE | Jet(4)": solver_residual(
            num_derivatives=4, time_span=time_span, precision_fun=precision_fun
        ),
        "ODE | TS1(3)": solver_ode(
            num_derivatives=3, time_span=time_span, precision_fun=precision_fun
        ),
        "ODE | TS1(4)": solver_ode(
            num_derivatives=4, time_span=time_span, precision_fun=precision_fun
        ),
        "ODE | TS1(7)": solver_ode(
            num_derivatives=7, time_span=time_span, precision_fun=precision_fun
        ),
        "ODE | LSODA (Scipy)": solver_scipy(
            method="LSODA", time_span=time_span, precision_fun=precision_fun
        ),
    }

    # Compute all work-precision diagrams
    _fig, ax = plt.subplots(ncols=2, figsize=(8, 3), constrained_layout=True)
    pbar = tqdm.tqdm(algorithms.items())
    for label, algo in pbar:
        pbar.set_description(label)
        param_to_wp = benchmark_util.workprec(algo, num_timing_calls=repeats)
        wp = param_to_wp(tolerances)

        ax[0].loglog(wp.precision["rmse"], wp.work.mean(axis=-1), ".-", label=label)
        ax[1].loglog(tolerances, wp.precision["constraint_violation"], "-", label=label)

    ax[0].set_title("Work-precision diagram")
    ax[0].set_xlabel("Precision (relative RMSE)")
    ax[0].set_ylabel("Work (avg. wall time)")
    ax[0].grid(linestyle="dotted", which="both")
    ax[0].legend(fontsize="xx-small")

    ax[1].set_title("Constraint violation")
    ax[1].set_xlabel("Tolerance (user input)")
    ax[1].set_ylabel("Algebraic constraint violation")
    ax[1].grid(linestyle="dotted", which="both")
    ax[1].legend(fontsize="xx-small")

    plt.show()


def solve_ivp_once(*, save_at, method, tol):
    """Compute plotting-values for the IVP."""

    def vf(t, y):
        del t
        k1, k2, k3 = 0.04, 3e7, 1e4
        f0 = -k1 * y[0] + k3 * y[1] * y[2]
        f1 = k1 * y[0] - k2 * y[1] ** 2 - k3 * y[1] * y[2]
        f2 = k2 * y[1] ** 2
        return np.stack([f0, f1, f2])

    y0 = jnp.array([1.0, 0.0, 0.0])

    t0, t1 = save_at[0], save_at[-1]
    solution = scipy.integrate.solve_ivp(
        vf,
        y0=y0,
        t_span=(t0, t1),
        t_eval=save_at,
        atol=1e-3 * tol,
        rtol=tol,
        method=method,
    )
    return solution.t, solution.y.T


def solver_ode(*, num_derivatives: int, time_span, precision_fun) -> Callable:
    """Construct a method that solves Robertson as an ODE."""

    @probdiffeq.residual_velocity
    def residual(u, du, /, *, t):
        return du - vf(u, t=t)

    @probdiffeq.ode
    def vf(y, *, t):
        del t
        k1, k2, k3 = 0.04, 3e7, 1e4
        f0 = -k1 * y[0] + k3 * y[1] * y[2]
        f1 = k1 * y[0] - k2 * y[1] ** 2 - k3 * y[1] * y[2]
        f2 = k2 * y[1] ** 2
        return jnp.stack([f0, f1, f2])

    t0, t1 = time_span
    y0 = jnp.array([1.0, 0.0, 0.0])

    @jax.jit
    def param_to_solution(tol):
        # Build a solver
        jetexpand = probdiffeq.jetexpand_ode_padded_scan(num=num_derivatives - 1)
        tcoeffs, _ = jetexpand(vf, (y0,), t=t0)

        ssm = probdiffeq.state_space_model_dense()

        base_scale = jnp.asarray([1e0, 1e-5, 1e-1])
        iwp = ssm.prior_wiener_integrated(tcoeffs, output_scale=base_scale)
        ts = ssm.constraint_ode_ts1(vf)
        strategy = probdiffeq.strategy_filter()

        solver = probdiffeq.solver_dynamic(strategy=strategy, constraint=ts)
        error = probdiffeq.error_state_std(constraint=ts)

        solve = ivpsolve.solve_adaptive_terminal_values(solver=solver, error=error)
        solution = solve(iwp, t0=t0, t1=t1, atol=1e-3 * tol, rtol=tol)

        return precision_fun(solution.u.mean[0])

    return param_to_solution


def solver_residual(*, num_derivatives: int, time_span, precision_fun) -> Callable:
    """Construct a method that solves Robertson as a DAE."""

    @probdiffeq.residual_velocity
    def differential(u, du, /, *, t):
        del t
        return du[:2] - dynamics(u)

    def dynamics(y):
        k1, k2, k3 = 0.04, 3e7, 1e4
        f0 = -k1 * y[0] + k3 * y[1] * y[2]
        f1 = k1 * y[0] - k2 * y[1] ** 2 - k3 * y[1] * y[2]
        return jnp.stack([f0, f1])

    @probdiffeq.residual_position
    def algebraic(u, *, t):
        del t
        return u[0] + u[1] + u[2] - 1

    differential = differential.jet_lift(lift_by=num_derivatives - 1)
    algebraic = algebraic.jet_lift(lift_by=num_derivatives)
    residual = probdiffeq.residual_from_stack(differential, algebraic)

    @jax.jit
    def param_to_solution(tol):
        t0, t1 = time_span
        y0 = [jnp.array([1.0, 0.0, 0.0])]
        nlstsq = probdiffeq.lstsq_constrained_gauss_newton(
            maxiter=10, tol=jnp.finfo(y0[0].dtype).eps ** 0.5
        )
        jetexpand = probdiffeq.jetexpand_residual(nlstsq=nlstsq, num=num_derivatives)
        tcoeffs, _ = jetexpand(residual, y0, t=t0)

        ssm = probdiffeq.state_space_model_dense()

        base_scale = jnp.asarray([1e0, 1e-5, 1e-1])
        iwp = ssm.prior_wiener_integrated(tcoeffs, output_scale=base_scale)

        # We build a Jet constraint
        taylor_point = probdiffeq.taylor_point_maximum_a_posteriori(nlstsq)
        jet = ssm.constraint_residual(residual, taylor_point=taylor_point)
        strategy = probdiffeq.strategy_filter()

        # For proper DAEs, non-iterated solver's simply don't cut it
        solver = probdiffeq.solver_dynamic(strategy=strategy, constraint=jet)

        # The state-error-estimate doesn't care about the dimension
        # of the DAE, which is exactly what we need here
        error = probdiffeq.error_state_std(constraint=jet)

        # TODO: build PID controllers (is this "gustafsson"?) for iterated solvers?
        solve = ivpsolve.solve_adaptive_terminal_values(solver=solver, error=error)
        solution = solve(iwp, t0=t0, t1=t1, atol=1e-3 * tol, rtol=tol)

        return precision_fun(solution.u.mean[0])

    return param_to_solution


def solver_scipy(*, method: str, time_span, precision_fun) -> Callable:
    """Construct a solver that wraps SciPy's solution routines."""

    def vf(t, y):
        del t
        k1, k2, k3 = 0.04, 3e7, 1e4
        f0 = -k1 * y[0] + k3 * y[1] * y[2]
        f1 = k1 * y[0] - k2 * y[1] ** 2 - k3 * y[1] * y[2]
        f2 = k2 * y[1] ** 2
        return np.stack([f0, f1, f2])

    y0 = jnp.array([1.0, 0.0, 0.0])

    def param_to_solution(tol):
        solution = scipy.integrate.solve_ivp(
            vf,
            y0=y0,
            t_span=time_span,
            t_eval=time_span,
            atol=1e-3 * tol,
            rtol=tol,
            method=method,
        )
        return precision_fun(jnp.asarray(solution.y[:, -1]))

    return param_to_solution


main()
