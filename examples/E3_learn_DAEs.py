"""Learn a DAE."""

import equinox
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import tqdm

from probdiffeq import ivpsolve, probdiffeq, taylor
from probdiffeq.util import nlstsq_util

# Fail this notebook on NaN detection (to catch those in the CI)
jax.config.update("jax_debug_nans", True)

# TODO: move DAE constraint into initialisation
# TODO: Make probdiffeq compatible with parametrised diffeqs
# TODO: use EM for updating the initial condition
# TODO: learn one of Robertson's differential parameters
# TODO: Train a neural network to match the differential part?


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

    def while_loop(cond, body, init):
        """Evaluate a bounded while loop."""
        return equinox.internal.while_loop(
            cond, body, init, kind="checkpointed", max_steps=256
        )

    # Linear spacing on a log-scale
    save_at = 2.0 ** jnp.linspace(jnp.log2(t0), jnp.log2(t1), num=150)
    solve = solver(differential, algebraic, tol=1e-6, while_loop=while_loop)

    # This base scale is critical to Robertson, because
    # the solutions live on vastly different scales
    # (but don't vary much within these scales).
    output_scale = jnp.asarray([0.8, 2e-05, 0.2])

    # True condition and initial guess
    y0_true = jnp.sqrt(jnp.array([1.0, 0.0, 0.0]) / output_scale)
    y0_guess = jnp.sqrt(jnp.array([0.5 - 1e-5, 1e-5, 0.5]) / output_scale)

    # Create data
    solution_true = solve(y0_true, save_at=save_at, output_scale=output_scale)
    inputs = solution_true.t
    labels = solution_true.u.mean[0]

    # Fake SSM (to get the conditioning-functions to build a loss)
    _, ssm = probdiffeq.ssm_taylor([jnp.zeros((3,))], diffuse_derivatives=3)

    # Loss
    loss = loss_data_fit(solve, inputs, labels, ssm=ssm)
    value_and_grad = jax.jit(jax.value_and_grad(loss, has_aux=True))

    # Initialise diffusion tempering
    std = 1e0 * output_scale

    # Evaluate the initial loss and gradient
    (_value0, solution_guess0), _gradient0 = value_and_grad(
        y0_guess, std=std, output_scale=output_scale
    )

    # Initialise the optimiser
    optim = optax.adam(0.25)  # If we temper hard, we can use large LRs
    opt_state = optim.init(y0_guess)
    pbar = tqdm.tqdm(range(200))
    for i in pbar:
        # Optimisation step:
        (value, solution_guess), gradient = value_and_grad(
            y0_guess, std=std, output_scale=output_scale
        )
        pbar.set_description(f"{(value):.3f}, {y0_guess**2 * output_scale}, {std}")

        updates, opt_state = optim.update(gradient, opt_state)
        y0_guess = optax.apply_updates(y0_guess, updates)

        # Square, normalise (to satisfy the algebraic constraint),
        # and go back to the square-root
        y0_guess_square = y0_guess**2 * output_scale
        y0_guess_square /= jnp.sum(y0_guess_square)
        y0_guess = jnp.sqrt(y0_guess_square / output_scale)

        # Our own version of diffusion tempering
        #   The output scale is set in stone because otherwise,
        #   Robertson is just too hard to solve.
        #   But we can temper by reducing the standard deviations
        if i % 50 == 0:
            # The exact schedule is arbitrary tbh...
            std = jnp.maximum(std / 2.0, 1e-8 * output_scale)

    fig, ax = plt.subplots(ncols=2, nrows=3, figsize=(5, 5), sharex=True)
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

    plt.tight_layout()
    fig.align_ylabels()
    plt.show()


def loss_data_fit(solve, inputs, labels, *, ssm):
    """Create a loss that measures the data fit."""

    def loss(y0, std, output_scale):
        std_ts = jnp.ones_like(inputs)[:, None] * std[None, ...]
        loss_lml = probdiffeq.loss_lml_timeseries(ssm=ssm)
        sol = solve(y0, save_at=inputs, output_scale=output_scale)
        lml = loss_lml(labels, std=std_ts, posterior=sol.solution_full)
        return -lml, sol

    return loss


def solver(differential, algebraic, tol, while_loop):
    """Create a reverse-mode differentiable probabilistic solver."""

    @jax.jit
    def solve(y0_sqrt, save_at, output_scale):

        y0 = y0_sqrt**2 * output_scale
        t0, _t1 = save_at[0], save_at[-1]

        def differential_auto(u, du):
            return differential(u, du, t=t0)

        def algebraic_auto(u):
            return algebraic(u, t=t0)

        nlstsq = nlstsq_util.nlstsq_constrained_gauss_newton(
            maxiter=10, tol=tol, while_loop=while_loop
        )
        y0, _info = taylor.daejet_nlstsq(
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
