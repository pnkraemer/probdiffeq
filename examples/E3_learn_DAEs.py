"""Learn a DAE."""

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


def main(
    t0=1e-6, t1=1e5, num_data=10, tol=1e-10, std_log=-1.0, seed=1, epochs=1000
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

    # def while_loop(cond, body, init):
    #     """Evaluate a bounded while loop."""
    #     return equinox.internal.while_loop(
    #         cond, body, init, kind="checkpointed", max_steps=256
    #     )

    def while_loop(cond, body, init):
        return jax.lax.while_loop(cond, body, init_val=init)

    # This base scale is critical to Robertson, because
    # the solutions live on vastly different scales
    # (but don't vary much within these scales).
    output_scale = jnp.asarray([0.8, 2e-05, 0.2])
    trafo = Trafo(output_scale)

    # Linear spacing on a log-scale
    save_at = 2.0 ** jnp.linspace(jnp.log2(t0), jnp.log2(t1), num=num_data)
    solve = solver(differential, algebraic, tol=tol, while_loop=while_loop, trafo=trafo)

    # True condition
    key = jax.random.PRNGKey(seed)
    p_true = 10 * jax.random.uniform(key, shape=(2,)) - 5.0

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
    loss = loss_data_fit(solve, ssm=ssm)
    # value_and_grad = jax.jit(jax.value_and_grad(loss, argnums=0, has_aux=True))

    # xs = jnp.linspace(-5, 3, num=3)
    # ys = jnp.linspace(-5, 3, num=3)
    # mesh = jnp.stack(jnp.meshgrid(xs, ys))

    # loss_p = functools.partial(loss, std_log=std_log, output_scale=output_scale, inputs=inputs, labels=labels)

    # loss_p = jax.vmap(loss_p, in_axes=1, out_axes=-1)
    # loss_p = jax.vmap(loss_p, in_axes=2, out_axes=-1)
    # loss_p = jax.jit(loss_p)

    # print(p_true)
    # print(p_guess)

    # values, _ = loss_p(mesh)

    # print(jnp.amin(values))
    # plt.pcolormesh(mesh[0], mesh[1], jnp.log(values))
    # plt.colorbar()
    # plt.show()

    # Evaluate the initial loss and gradient
    # (_, solution_guess0), gradient0 = value_and_grad(
    #     p_guess, std_log, output_scale=output_scale, inputs=inputs, labels=labels
    # )
    # print(solution_guess0)
    raise RuntimeError(
        "TODO: investigate which part of the ODE solver has terrible gradients... make a loss that is a function of the initialisation, then try fixed steps, then try adaptive steps in different configs..."
    )

    print(
        "Fwd:",
        jax.jacfwd(loss, has_aux=True, argnums=0)(
            p_guess, std_log, output_scale=output_scale, inputs=inputs, labels=labels
        )[0],
    )
    gradient = jnp.zeros((2,))
    eps = 1e-4
    for i in [0, 1]:
        p_guess = p_guess.at[i].add(-eps)
        (value0, _) = loss(
            p_guess, std_log, output_scale=output_scale, inputs=inputs, labels=labels
        )
        p_guess = p_guess.at[i].add(2 * eps)
        (value1, _) = loss(
            p_guess, std_log, output_scale=output_scale, inputs=inputs, labels=labels
        )
        p_guess = p_guess.at[i].add(-eps)

        gradient = gradient.at[i].set((value1 - value0) / (2 * eps))

    print("Finite differences", gradient)

    assert False

    # Initialise the optimiser
    # Since we temper hard, we can use large learning rates
    optim = optax.adam(0.5)
    opt_state = optim.init((p_guess, std_log))

    i = 0
    while True:
        i += 1
        # Minibatch

        # Optimisation step:
        (value, solution_guess), gradient = value_and_grad(
            p_guess, std_log, output_scale=output_scale, inputs=inputs, labels=labels
        )
        print(value)
        print(gradient)
        print()
        # Print the progress
        # Don't print the loss value because the tempering makes it uninformative
        if i % 10 == 0:
            u0_guess = trafo.latent_to_observed(p_guess)
            u0_true = trafo.latent_to_observed(p_true)
            msg = f"loss={value:.4e}, u0={u0_guess}, std={jnp.exp(std_log):.1e}, target={u0_true}"
            print(f"Epoch {i:3d} | {msg}")

        # Update
        updates, opt_state = optim.update(gradient, opt_state)
        p_guess, std_log = optax.apply_updates((p_guess, std_log), updates)

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
        ax[0][0].semilogx(save_at, solution.u.mean[0][:, 0], label=label, alpha=0.75)
        ax[1][0].semilogx(save_at, solution.u.mean[0][:, 1], alpha=0.75)
        ax[2][0].semilogx(save_at, solution.u.mean[0][:, 2], alpha=0.75)

        ax[0][1].loglog(save_at, solution.u.std[0][:, 0], alpha=0.75)
        ax[1][1].loglog(save_at, solution.u.std[0][:, 1], alpha=0.75)
        ax[2][1].loglog(save_at, solution.u.std[0][:, 2], alpha=0.75)

    ax[0][0].legend(fontsize="small")
    ax[0][0].set_ylabel("State $y_1$", fontsize="medium")
    ax[1][0].set_ylabel("State $y_2$", fontsize="medium")
    ax[2][0].set_ylabel("State $y_3$", fontsize="medium")
    ax[2][0].set_xlabel("Time $t$", fontsize="medium")
    ax[2][1].set_xlabel("Time $t$", fontsize="medium")
    ax[0][0].set_xlim((t0, t1))

    fig.align_ylabels()
    plt.show()


def loss_data_fit(solve, *, ssm):
    """Create a loss that measures the data fit."""

    def loss(y0, std_log, output_scale, inputs, labels):
        std = jnp.exp(std_log) * output_scale
        std_ts = jnp.ones_like(inputs)[:, None] * std[None, ...]

        loss_lml = probdiffeq.loss_lml_timeseries(ssm=ssm)
        sol = solve(y0, save_at=inputs, output_scale=output_scale)

        # diff = sol.u.mean[0] - labels
        # return jnp.mean(diff**2), sol

        lml = loss_lml(labels, std=std_ts, posterior=sol.solution_full)
        return -lml, sol

    return loss


def solver(differential, algebraic, tol, while_loop, trafo):
    """Create a reverse-mode differentiable probabilistic solver."""

    @jax.jit
    def solve(p_sqrt, save_at, output_scale):

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
