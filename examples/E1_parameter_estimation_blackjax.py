"""Estimate parameters (via BlackJAX).

This tutorial explains how to estimate unknown parameters of
initial value problems (IVPs) using Markov Chain Monte Carlo (MCMC)
methods as provided by [BlackJAX](https://blackjax-devs.github.io/blackjax/).
"""

import functools

import blackjax
import jax
import jax.experimental.ode
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffeqzoo import backend, ivps

from probdiffeq import ivpsolve, probdiffeq, taylor

# IVP examples in JAX
if not backend.has_been_selected:
    backend.select("jax")


def main():
    """Use BlackJAX's samplers to estimate ODE parameters."""
    # First, set up an IVP and create some artificial data by
    # simulating the system with "incorrect" initial conditions.

    f, u0, (t0, t1), f_args = ivps.lotka_volterra()

    @jax.jit
    def vf(y, /, *, t):  # noqa: ARG001
        """Evaluate the Lotka-Volterra vector field."""
        return f(y, *f_args)

    theta_true = u0 + 0.5 * jnp.flip(u0)
    theta_guess = u0  # initial guess

    # Construct solvers
    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=t0), (theta_guess,), num=2)
    _init, ssm = probdiffeq.ssm_taylor(tcoeffs, ssm_fact="isotropic")
    iwp = probdiffeq.prior_wiener_integrated(ssm=ssm, output_scale=10.0)
    ts0 = probdiffeq.constraint_ode_ts0(vf, ssm=ssm)
    strategy = probdiffeq.strategy_filter(ssm=ssm)
    solver = probdiffeq.solver(strategy=strategy, prior=iwp, constraint=ts0, ssm=ssm)
    error = probdiffeq.error_residual_std(constraint=ts0, prior=iwp, ssm=ssm)

    save_at = jnp.linspace(t0, t1, num=250, endpoint=True)
    solve_save_at = solve_adaptive(vf, solver=solver, error=error, save_at=save_at)

    # Visualise the initial guess and the data

    _fig, ax = plt.subplots(figsize=(5, 3))

    data_kwargs = {"alpha": 0.5, "color": "gray"}
    ax.annotate("Data", (13.0, 30.0), **data_kwargs)
    sol = solve_save_at(theta_true)
    ax = plot_solution(sol.t, sol.u.mean[0], ax=ax, **data_kwargs)

    guess_kwargs = {"color": "C3"}
    ax.annotate("Initial guess", (7.5, 20.0), **guess_kwargs)
    sol = solve_save_at(theta_guess)
    ax = plot_solution(sol.t, sol.u.mean[0], ax=ax, **guess_kwargs)
    plt.show()

    # Set up a log-posterior density function that we can plug into BlackJAX.
    # Choose a Gaussian prior centered at the initial guess with a large variance.
    mean = theta_guess
    cov = jnp.eye(2) * 30  # fairly uninformed prior
    ts = jnp.linspace(t0, t1, endpoint=True, num=100)
    log_M = log_posterior(vf, theta_true, solver=solver, ts=ts, mean=mean, cov=cov)
    print(jnp.exp(log_M(theta_true)), ">=", jnp.exp(log_M(theta_guess)), "?")

    # From here on, BlackJAX takes over:
    initial_position = theta_guess
    rng_key = jax.random.PRNGKey(0)
    warmup = blackjax.window_adaptation(blackjax.nuts, log_M, progress_bar=True)
    warmup_results, _ = warmup.run(rng_key, initial_position, num_steps=200)
    initial_state = warmup_results.state
    step_size = warmup_results.parameters["step_size"]
    inverse_mass_matrix = warmup_results.parameters["inverse_mass_matrix"]
    nuts_kernel = blackjax.nuts(
        logdensity_fn=log_M,
        step_size=step_size,
        inverse_mass_matrix=inverse_mass_matrix,
    )

    # Inference loop
    rng_key, _ = jax.random.split(rng_key, 2)
    states = inference_loop(
        rng_key, kernel=nuts_kernel, initial_state=initial_state, num_samples=150
    )

    # Now that we have samples of $\theta$, let's plot the corresponding solutions:
    solution_samples = jax.vmap(solve_save_at)(states.position)

    # Visualise the initial guess and the data

    _fig, ax = plt.subplots()

    sample_kwargs = {"color": "C0"}
    ax.annotate("Samples", (2.75, 31.0), **sample_kwargs)
    for ts, us in zip(solution_samples.t, solution_samples.u.mean[0]):
        ax = plot_solution(ts, us, ax=ax, linewidth=0.1, alpha=0.75, **sample_kwargs)

    data_kwargs = {"color": "gray"}
    ax.annotate("Data", (18.25, 40.0), **data_kwargs)
    sol = solve_save_at(theta_true)
    ax = plot_solution(
        sol.t, sol.u.mean[0], ax=ax, linewidth=4, alpha=0.5, **data_kwargs
    )

    guess_kwargs = {"color": "gray"}
    ax.annotate("Initial guess", (6.0, 12.0), **guess_kwargs)
    sol = solve_save_at(theta_guess)
    ax = plot_solution(
        sol.t, sol.u.mean[0], ax=ax, linestyle="dashed", alpha=0.75, **guess_kwargs
    )
    plt.show()

    # In parameter space, this is what it looks like:
    xlim = 14, jnp.amax(states.position[:, 0]) + 0.5
    ylim = 14, jnp.amax(states.position[:, 1]) + 0.5

    xs = jnp.linspace(*xlim, endpoint=True, num=300)
    ys = jnp.linspace(*ylim, endpoint=True, num=300)
    Xs, Ys = jnp.meshgrid(xs, ys)

    Thetas = jnp.stack((Xs, Ys))
    log_M_vmapped_x = jax.vmap(log_M, in_axes=-1, out_axes=-1)
    log_M_vmapped = jax.vmap(log_M_vmapped_x, in_axes=-1, out_axes=-1)
    Zs = log_M_vmapped(Thetas)

    _fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(8, 3))

    ax_samples, ax_heatmap = ax

    ax_samples.set_title("Posterior samples (parameter space)", fontsize="medium")
    ax_samples.plot(
        states.position[:, 0], states.position[:, 1], ".", alpha=0.5, markersize=4
    )
    ax_samples.plot(theta_true[0], theta_true[1], "P", label="Truth", markersize=8)
    ax_samples.plot(
        theta_guess[0], theta_guess[1], "P", label="Initial guess", markersize=8
    )
    ax_samples.legend()

    ax_heatmap.set_title("Target density", fontsize="medium")
    im = ax_heatmap.contourf(Xs, Ys, jnp.exp(Zs), cmap="cividis", alpha=0.8)
    plt.colorbar(im)
    plt.show()


def solve_adaptive(vf, *, solver, error, save_at):
    """Create an adaptive solver (for visualisation)."""

    @jax.jit
    def solve(theta):
        """Evaluate the parameter-to-solution map, solving on an adaptive grid."""
        tcoeffs = taylor.odejet_padded_scan(
            lambda y: vf(y, t=save_at[0]), (theta,), num=2
        )
        init, _ssm = probdiffeq.ssm_taylor(tcoeffs, ssm_fact="isotropic")
        solve = ivpsolve.solve_adaptive_save_at(solver=solver, error=error)
        return solve(init, save_at=save_at, dt0=0.1, atol=1e-4, rtol=1e-2)

    return solve


def plot_solution(t, u, *, ax, marker=".", **plotting_kwargs):
    """Plot the IVP solution."""
    for d in [0, 1]:
        ax.plot(t, u[:, d], marker="None", **plotting_kwargs)
        ax.plot(t[0], u[0, d], marker=marker, **plotting_kwargs)
        ax.plot(t[-1], u[-1, d], marker=marker, **plotting_kwargs)
    return ax


def log_posterior(vf, theta_true, *, solver, ts, mean, cov, obs_std=0.1):
    """Create a log-posterior density function."""
    tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=ts[0]), (theta_true,), num=2)
    init, _ssm = probdiffeq.ssm_taylor(tcoeffs, ssm_fact="isotropic")
    solve = ivpsolve.solve_fixed_grid(solver=solver)
    sol = solve(init, grid=ts)
    data = sol.u.mean[0][-1]

    @jax.jit
    def logposterior(theta):
        """Evaluate the logposterior-function of the data."""
        tcoeffs = taylor.odejet_padded_scan(lambda y: vf(y, t=ts[0]), (theta,), num=2)
        init, ssm = probdiffeq.ssm_taylor(tcoeffs, ssm_fact="isotropic")
        solve = ivpsolve.solve_fixed_grid(solver=solver)
        solution = solve(init, grid=ts)
        y_T = jax.tree.map(lambda s: s[-1], solution.u.marginals)
        loss = probdiffeq.loss_lml_terminal_values(ssm=ssm)
        logpdf_data = loss(data, std=obs_std, marginals=y_T)
        logpdf_prior = jax.scipy.stats.multivariate_normal.logpdf(
            theta, mean=mean, cov=cov
        )
        return logpdf_data + logpdf_prior

    return logposterior


@functools.partial(jax.jit, static_argnames=["kernel", "num_samples"])
def inference_loop(rng_key, kernel, initial_state, num_samples):
    """Run BlackJAX' inference loop."""

    def one_step(state, rng_key):
        state, _ = kernel.step(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


if __name__ == "__main__":
    main()
