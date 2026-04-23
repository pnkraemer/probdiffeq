"""Learn a DAE."""

import functools

import equinox
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

from probdiffeq import diffeqjet, ivpsolve, probdiffeq
from probdiffeq.util import nlstsq_util

# Fail this notebook on NaN detection (to catch those in the CI)
jax.config.update("jax_debug_nans", True)


class Trafo:
    """Coordinate transformation to make the optimisation problem well-posed."""

    def __init__(self, scale):
        # e.g. jnp.array([1., 1e-5, 1e-3])
        self.scale = jnp.array(scale)

    def observed_to_latent(self, x, eps=1e-6):
        """Simplex R^3 -> unconstrained R^2."""
        x = x / self.scale
        x = jnp.clip(x, eps, 1.0 - eps)
        x = x / x.sum()  # renormalise
        return jnp.log(x[:-1] / x[-1])

    def latent_to_observed(self, u):
        """Unconstrained R^2 -> simplex R^3."""
        u_full = jnp.append(u, 0.0)
        u_full = u_full - jnp.max(u_full)
        e = jnp.exp(u_full)
        x = e / e.sum()

        # Rescale back
        x *= self.scale
        return x / x.sum()


# try: std=0, std=1e-2, std=1e-4, etc. (smaller std -> "longer" gradient in the imshow, but also larger values)
# also try: tol=1e-2, 1e-4, etc. (larger tolerance, noisier gradients in the imshow)
def main(
    t0=1e-6,
    t1=1e5,
    num_data=20,
    tol=1e-8,
    std=1e2,
    seed=1,
    epochs=1000,
    temper_every=100,
    temper_by=0.1,
    plot_loss=False,
) -> None:
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

    def while_loop(cond, body, init):
        """Evaluate a bounded while loop."""
        return equinox.internal.while_loop(
            cond, body, init, kind="checkpointed", max_steps=256
        )

    # Linear spacing on a log-scale
    save_at = 2.0 ** jnp.linspace(jnp.log2(t0), jnp.log2(t1), num=num_data)
    solve = solver(differential, algebraic, tol=tol, while_loop=while_loop)

    # This base scale is critical to Robertson, because
    # the solutions live on vastly different scales
    # (but don't vary much within these scales).
    output_scale = jnp.asarray([0.8, 2e-05, 0.2])

    # True condition
    key = jax.random.PRNGKey(seed)
    p_true = 10 * jax.random.uniform(key, shape=(2,)) - 5.0

    trafo = Trafo(output_scale)
    # p_true = trafo.observed_to_latent(jnp.array([1.0, 0.0, 0.0]))

    # Initial guess: p0 ~ U(-5, 5)
    key = jax.random.PRNGKey(seed + 1)
    p_guess = 10 * jax.random.uniform(key, shape=(2,)) - 5.0

    # Create data
    solution_true = solve(p_true, save_at=save_at, output_scale=output_scale)
    inputs = solution_true.t
    labels = solution_true.u.mean[0]

    # Fake SSM (to get the conditioning-functions to build a loss)
    _, ssm = probdiffeq.ssm_taylor([jnp.zeros((3,))], diffuse_derivatives=3)

    # Loss
    loss = loss_data_fit(solve, inputs, labels, ssm=ssm)
    value_and_grad = jax.jit(jax.value_and_grad(loss, has_aux=True))

    if plot_loss:
        # Build a parameter-space meshgrid
        xs = jnp.linspace(-1, 12, num=10)
        ys = jnp.linspace(-1, 12, num=10)
        mesh = jnp.stack(jnp.meshgrid(xs, ys))

        # Vectorise the loss
        loss_p = functools.partial(loss, std=std, output_scale=output_scale)
        for idx in [1, 2]:
            loss_p = jax.vmap(loss_p, in_axes=idx, out_axes=-1)

        # Call the vectorised loss
        vals, _ = jax.jit(loss_p)(mesh)

        # Plot
        plt.title(f"Loss landscape (tol={tol}, std={std})", fontsize="medium")
        plt.scatter(
            p_true[0], p_true[1], marker="X", color="black", label="True", zorder=10
        )
        plt.scatter(
            p_guess[0], p_guess[1], marker="X", color="black", label="True", zorder=10
        )
        img = plt.pcolormesh(mesh[0], mesh[1], vals, cmap="managua")
        plt.colorbar(img)
        plt.xlabel("p0")
        plt.ylabel("p1")
        plt.legend()
        plt.show()

    # Evaluate the initial loss and gradient
    (_value0, solution_guess0), _gradient0 = value_and_grad(
        p_guess, std=std, output_scale=output_scale
    )

    # Initialise the optimiser
    # Since we temper hard, we can use large learning rates
    optim = optax.adam(0.25)
    opt_state = optim.init(p_guess)

    for i in range(epochs):
        # Optimisation step:
        (_, solution_guess), gradient = value_and_grad(
            p_guess, std=std, output_scale=output_scale
        )

        # Print the progress
        # Don't print the loss value because the tempering makes it uninformative
        if i % 10 == 0:
            u0_guess = trafo.latent_to_observed(p_guess)
            u0_true = trafo.latent_to_observed(p_true)
            msg = f"u0={u0_guess}, std={std}, target={u0_true}"
            print(f"Epoch {i:3d} | {msg}")

        # Update
        updates, opt_state = optim.update(gradient, opt_state)
        p_guess = optax.apply_updates(p_guess, updates)

        # Our own version of diffusion tempering
        #   The output scale is set in stone because otherwise,
        #   Robertson is just too hard to solve.
        #   But we can temper by reducing the standard deviations
        if i > 0 and i % temper_every == 0:
            # The exact schedule is arbitrary tbh...
            std = jnp.maximum(std * temper_by, 1e-12)

    # Plot the before-after
    fig, ax = plt.subplots(
        ncols=2, nrows=3, figsize=(5, 5), sharex=True, constrained_layout=True
    )
    ax[0][0].set_title("Robertson solution", fontsize="medium")
    ax[0][1].set_title("Standard deviations", fontsize="medium")

    # Plot means and standard deviations of the true solution, initial guess, and final guess
    results = {
        "True": solution_true,
        "Initial": solution_guess0,
        "Final": solution_guess,
    }
    for label, solution in results.items():
        ax[0][0].semilogx(save_at, solution.u.mean[0][:, 0], label=label)
        ax[1][0].semilogx(save_at, solution.u.mean[0][:, 1])
        ax[2][0].semilogx(save_at, solution.u.mean[0][:, 2])

        ax[0][1].loglog(save_at, solution.u.std[0][:, 0])
        ax[1][1].loglog(save_at, solution.u.std[0][:, 1])
        ax[2][1].loglog(save_at, solution.u.std[0][:, 2])

    ax[0][0].legend(fontsize="small")
    ax[0][0].set_ylabel("State $y_1$", fontsize="medium")
    ax[1][0].set_ylabel("State $y_2$", fontsize="medium")
    ax[2][0].set_ylabel("State $y_3$", fontsize="medium")
    ax[2][0].set_xlabel("Time $t$", fontsize="medium")
    ax[2][1].set_xlabel("Time $t$", fontsize="medium")
    ax[0][0].set_xlim((t0, t1))

    fig.align_ylabels()
    plt.show()


def loss_data_fit(solve, inputs, labels, *, ssm):
    """Create a loss that measures the data fit."""

    def loss(y0, std, output_scale):
        std *= output_scale
        std_ts = jnp.ones_like(inputs)[:, None] * std[None, ...]
        loss_lml = probdiffeq.loss_lml_timeseries(ssm=ssm)
        sol = solve(y0, save_at=inputs, output_scale=output_scale)
        lml = loss_lml(labels, std=std_ts, posterior=sol.solution_full)
        return -lml, sol

    return loss


def solver(differential, algebraic, tol, while_loop):
    """Create a reverse-mode differentiable probabilistic solver."""

    @jax.jit
    def solve(p_sqrt, save_at, output_scale):

        trafo = Trafo(output_scale)
        y0 = trafo.latent_to_observed(p_sqrt)
        t0, _t1 = save_at[0], save_at[-1]

        def differential_auto(u, du):
            return differential(u, du, t=t0)

        def algebraic_auto(u):
            return algebraic(u, t=t0)

        nlstsq = nlstsq_util.nlstsq_constrained_gauss_newton(
            maxiter=10, tol=tol, while_loop=while_loop
        )
        y0, _info = diffeqjet.daejet_nlstsq(
            differential_auto, algebraic_auto, [y0], num=3, nlstsq=nlstsq
        )
        init, ssm = probdiffeq.ssm_taylor(y0)

        prior = probdiffeq.prior_wiener_integrated(ssm=ssm, output_scale=output_scale)

        # We build a Jet constraint. Iteration is key, because DAEs are proper stiff.
        jet = probdiffeq.constraint_jet_dae(
            differential, algebraic, ssm=ssm, nlstsq=nlstsq
        )
        strategy = probdiffeq.strategy_smoother_fixedpoint(ssm=ssm)
        solver = probdiffeq.solver_dynamic(
            strategy=strategy, prior=prior, constraint=jet, ssm=ssm
        )

        # The state-error-estimate doesn't care about the dimension
        # of the DAE, which is exactly what we need here
        error = probdiffeq.error_state_std(constraint=jet, prior=prior, ssm=ssm)

        solve = ivpsolve.solve_adaptive_save_at(
            solver=solver, error=error, while_loop=while_loop
        )
        return solve(init, save_at=save_at, atol=tol, rtol=tol)

    return solve


if __name__ == "__main__":
    main()
