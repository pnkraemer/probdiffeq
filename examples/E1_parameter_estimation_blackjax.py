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

from probdiffeq import ivpsolve, probdiffeq

# IVP examples in JAX
if not backend.has_been_selected:
    backend.select("jax")


def main():
    """Use BlackJAX's samplers to estimate ODE parameters."""
    # Set up an initial value problem

    f, u0, (t0, t1), f_args = ivps.lotka_volterra()

    @probdiffeq.ode
    def vf(y, /, *, t):
        """Evaluate the Lotka-Volterra vector field."""
        del t
        return f(y, *f_args)

    # Construct solvers
    solve = solve_fixed(vf, t0=t0, t1=t1, num=200)

    # Determine true parameters and create data
    theta_true = u0 + 0.5 * jnp.flip(u0)
    solution_true = solve(theta_true)
    data = solution_true.u.mean[0]

    # Determine initial guesses
    theta0 = u0

    # Set up a log-posterior density function that we can plug into BlackJAX.
    # Choose a Gaussian prior centered at the initial guess with a large variance.
    mean = theta0
    cov = jnp.eye(2) * 30
    log_M = log_posterior(solve=solve, data=data, mean=mean, cov=cov)
    log_M(theta0)

    # From here on, BlackJAX takes over:

    # Warmup
    print("\nRunning window adaptation...", end="")
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key, num=2)
    warmup = blackjax.window_adaptation(blackjax.nuts, log_M, progress_bar=False)
    warmup_results, _ = warmup.run(subkey, theta0, num_steps=200)
    print("done.")

    # Inference loop
    print("\nRunning inference loop...", end="")
    nuts_kernel = blackjax.nuts(
        logdensity_fn=log_M,
        step_size=warmup_results.parameters["step_size"],
        inverse_mass_matrix=warmup_results.parameters["inverse_mass_matrix"],
    )
    key, subkey = jax.random.split(key, num=2)
    states = inference_loop(
        subkey, kernel=nuts_kernel, initial_state=warmup_results.state, num_samples=150
    )
    samples = states.position
    print("done.")

    # Create the plot
    _fig, ax = plt.subplot_mosaic(
        [["smp", "dens"]], sharex=True, sharey=True, figsize=(8, 3)
    )

    # Plot the samples
    ax["smp"].set_title("Posterior samples (parameter space)", fontsize="medium")
    ax["smp"].plot(samples[:, 0], samples[:, 1], ".", alpha=0.5, markersize=4)
    ax["smp"].plot(theta_true[0], theta_true[1], "P", label="Truth", markersize=8)
    ax["smp"].plot(theta0[0], theta0[1], "P", label="Initial guess", markersize=8)
    ax["smp"].legend()

    # Create a meshgrid for plotting the density
    xlim = 17, jnp.amax(samples[:, 0]) + 0.5
    ylim = 17, jnp.amax(samples[:, 1]) + 0.5
    xs = jnp.linspace(*xlim, endpoint=True, num=200)
    ys = jnp.linspace(*ylim, endpoint=True, num=200)
    Xs, Ys = jnp.meshgrid(xs, ys)

    # Evaluate the density
    Thetas = jnp.stack((Xs, Ys))
    log_M_vmapped_x = jax.vmap(log_M, in_axes=-1, out_axes=-1)
    log_M_vmapped = jax.vmap(log_M_vmapped_x, in_axes=-1, out_axes=-1)
    Zs = log_M_vmapped(Thetas)

    # Plot the density
    ax["dens"].set_title("Target density", fontsize="medium")
    im = ax["dens"].pcolormesh(Xs, Ys, jnp.exp(Zs), cmap="cividis", alpha=0.8)
    plt.colorbar(im)
    plt.show()


def solve_fixed(vf, *, t0, t1, num):
    """Create an adaptive solver (for visualisation)."""
    ssm = probdiffeq.state_space_model_isotropic()
    ts0 = ssm.constraint_ode_ts0(vf)
    strategy = probdiffeq.strategy_filter()
    solver = probdiffeq.solver(strategy=strategy, constraint=ts0)
    solve_fn = ivpsolve.solve_fixed_grid(solver=solver)
    grid = jnp.linspace(t0, t1, num=num, endpoint=True)

    @jax.jit
    def solve(theta, /):
        """Evaluate the parameter-to-solution map, solving on an adaptive grid."""
        jetexpand = probdiffeq.jetexpand_ode_padded_scan(num=3)
        tcoeffs, _ = jetexpand(vf, (theta,), t=t0)

        iwp = ssm.prior_wiener_integrated(tcoeffs)
        sol = solve_fn(iwp, grid=grid)
        return jax.tree.map(lambda s: s[-1], sol)

    return solve


def log_posterior(*, solve, data, mean, cov, obs_std=0.1):
    """Create a log-posterior density function."""
    loss = probdiffeq.loss_lml_terminal_values()
    logpdf_normal = jax.scipy.stats.multivariate_normal.logpdf

    @jax.jit
    def logposterior(theta, /):
        """Evaluate the logposterior-function of the data."""
        solution = solve(theta)
        logpdf_data = loss(data, std=obs_std, marginals=solution.u)
        logpdf_prior = logpdf_normal(theta, mean=mean, cov=cov)
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
