import statistics
from collections.abc import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate

from probdiffeq import ivpsolve, probdiffeq, taylor

# Fail this notebook on NaN detection (to catch those in the CI)
jax.config.update("jax_debug_nans", True)


def main(t0=1e-6, t1=1e5) -> None:
    """Run the script."""
    # Set up all the configs
    jax.config.update("jax_enable_x64", True)

    def differential(u, du, /, *, t):
        del t
        return du[:2] - dynamics(u)

    def dynamics(y):
        k1, k2, k3 = 0.04, 3e7, 1e4
        f0 = -k1 * y[0] + k3 * y[1] * y[2]
        f1 = k1 * y[0] - k2 * y[1] ** 2 - k3 * y[1] * y[2]
        return jnp.stack([f0, f1])

    def algebraic(u, *, t):
        del t
        return u[0] + u[1] + u[2] - 1

    # This base scale is critical to Robertson, because
    # the solutions live on vastly different scales
    # (but don't vary much within these scales).
    # TODO: what is the best "base output scale" for the solver?
    #       this should be an expectation-maximisation thing right?
    base_scale = jnp.asarray([1e0, 1e-5, 0.1])

    # For DAEs, not all variables are differential, and we need to have
    #   and idea which ones arent to stabilise the solver initialisation
    y0 = [jnp.array([1.0, 0.0, 0.0])]
    M = jnp.asarray([[-0.04, 0.0, 0.0], [0.04, 0.0, 0.0], [0.0, 0.0, 0.0]])

    lstsq = taylor.nonlinear_lstsq_levenberg_marquardt(maxiter=100)
    y0, _info = taylor.daejet_nonlinear_lstsq(
        lambda *a: differential(*a, t=t0),
        lambda *a: algebraic(*a, t=t0),
        y0,
        num=4,
        nonlinear_lstsq=lstsq,
    )

    # TODO: clean up the IOUP api. also, what about isotropic etc.?
    # is_differential = [jnp.array([True, True, True])]
    init, ioup, ssm = probdiffeq.prior_iwp(
        y0,
        output_scale=base_scale,
        # is_differential=is_differential,
    )

    # We build a Jet constraint
    jet = probdiffeq.constraint_dae_jet(differential, algebraic, ssm=ssm, iterate=True)
    strategy = probdiffeq.strategy_smoother_fixedpoint(ssm=ssm)

    # TODO: should the dynamic solver calibrate inside the iteration step?
    # For proper DAEs, non-iterated solver's simply don't cut it
    solver = probdiffeq.solver_dynamic(
        strategy=strategy, prior=ioup, constraint=jet, ssm=ssm
    )
    print(solver)

    # The state-error-estimate doesn't care about the dimension
    # of the DAE, which is exactly what we need here
    error = probdiffeq.error_state_std(constraint=jet, prior=ioup, ssm=ssm)

    # Linear spacing on a log-scale
    save_at = 2.0 ** jnp.linspace(jnp.log2(t0), jnp.log2(t1), num=100)
    solve = ivpsolve.solve_adaptive_save_at(solver=solver, error=error)
    solution = solve(init, save_at=save_at, atol=1e-8, rtol=1e-8)
    print(solution.num_steps)

    _fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(5, 5), sharex=True)
    ax[0][0].set_title("Robertson solution")

    # Plot a special index
    i = 0
    ax[0][0].semilogx(save_at, solution.u.mean[i][:, 0] / scipy.special.factorial(i))
    ax[1][0].semilogx(save_at, solution.u.mean[i][:, 1] / scipy.special.factorial(i))
    ax[2][0].semilogx(save_at, solution.u.mean[i][:, 2] / scipy.special.factorial(i))

    ax[0][1].loglog(save_at, solution.u.std[i][:, 0] / scipy.special.factorial(i))
    ax[1][1].loglog(save_at, solution.u.std[i][:, 1] / scipy.special.factorial(i))
    ax[2][1].loglog(save_at, solution.u.std[i][:, 2] / scipy.special.factorial(i))

    ax[0][0].set_ylabel("State $y_1$")
    ax[1][0].set_ylabel("State $y_2$")
    ax[2][0].set_ylabel("State $y_3$")
    ax[2][0].set_xlabel("Time $t$")
    ax[0][0].set_xlim((t0, t1))

    plt.tight_layout()
    plt.show()


def solver_dae(*, num_derivatives: int, time_span) -> Callable:
    """Construct a method that solves Robertson as a DAE."""

    def root(u, du, /, *, t):
        del t
        return [vf_differential(u, du), vf_algebraic(u)]

    def vf_differential(y, du):
        k1, k2, k3 = 0.04, 3e7, 1e4
        f0 = -k1 * y[0] + k3 * y[1] * y[2]
        f1 = k1 * y[0] - k2 * y[1] ** 2 - k3 * y[1] * y[2]
        return jnp.stack([du[0] - f0, du[1] - f1])

    def vf_algebraic(u):
        return u[0] + u[1] + u[2] - 1

    @jax.jit
    def param_to_solution(tol):

        # This base scale is critical to Robertson, because
        # the solutions live on vastly different scales
        # (but don't vary much within these scales).
        # TODO: what is the best "base output scale" for the solver?
        #       this should be an expectation-maximisation thing right?
        base_scale = jnp.asarray([1e0, 1e-5, 1e-1])

        # For DAEs, not all variables are differential, and we need to have
        #   and idea which ones arent to stabilise the solver initialisation
        y0 = [jnp.array([1.0, 0.0, 0.0])]
        is_differential = [jnp.array([True, True, False])]
        init, ibm, ssm = probdiffeq.prior_iwp(
            y0,
            output_scale=base_scale,
            diffuse_derivatives=num_derivatives,
            is_differential=is_differential,
        )

        # We build a Jet constraint
        jet = probdiffeq.constraint_root_jet(root, ssm=ssm)
        ts = probdiffeq.constraint_root_ts1(root, ssm=ssm)
        strategy = probdiffeq.strategy_filter(ssm=ssm)

        # For proper DAEs, non-iterated solver's simply don't cut it
        solver = probdiffeq.solver_dynamic(
            strategy=strategy, prior=ibm, constraint=jet, ssm=ssm, constraint_init=jet
        )

        # The state-error-estimate doesn't care about the dimension
        # of the DAE, which is exactly what we need here
        error = probdiffeq.error_state_std(constraint=ts, prior=ibm, ssm=ssm)

        # Integral controllers just work better than proportional-integral ones
        # TODO: build PID controllers (is this "gustafsson"?) for iterated solvers?
        solve = ivpsolve.solve_adaptive_terminal_values(
            solver=solver, error=error, clip_dt=True
        )
        t0, t1 = time_span
        solution = solve(init, t0=t0, t1=t1, atol=1e-3 * tol, rtol=tol)

        return jax.block_until_ready(solution.u.mean[0])

    return param_to_solution


def solver_scipy(*, method: str, time_span) -> Callable:
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
        return jnp.asarray(solution.y[:, -1])

    return param_to_solution


def rmse_relative(expected: jax.Array) -> Callable:
    """Compute the absolute RMSE."""
    expected = jnp.asarray(expected)

    def rmse(received):
        received = jnp.asarray(received)
        error_absolute = jnp.abs(expected - received)

        error_relative = error_absolute / (1e-5 + jnp.abs(expected))
        rmse = jnp.linalg.norm(error_relative) / jnp.sqrt(error_relative.size)

        algebraic = jnp.sum(received)
        return rmse, algebraic

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
