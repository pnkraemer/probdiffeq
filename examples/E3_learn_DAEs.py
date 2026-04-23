"""Learn a DAE."""

import functools

import jax
import jax.numpy as jnp
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
    t0=1e-6, t1=1e5, num_data=20, tol=1e-7, std_log=-5.0, seed=1412321, epochs=1000
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

    loss = functools.partial(
        loss, std_log=std_log, output_scale=output_scale, inputs=inputs, labels=labels
    )
    loss = jax.jit(loss)

    optim = optax.sgd(1.0)
    opt_state = optim.init(p_guess)

    for epoch in range(epochs):
        # Finite difference gradient
        # TODO: fix the autodiff gradients...
        eps = 1e-4
        grad = jnp.zeros_like(p_guess)
        for i in [0, 1]:
            p_guess = p_guess.at[i].add(-eps)
            v0, _ = loss(p_guess)
            p_guess = p_guess.at[i].add(2 * eps)
            v1, _ = loss(p_guess)
            grad = grad.at[i].set((v1 - v0) / (2 * eps))

        # Optimiser step
        updates, opt_state = optim.update(grad, opt_state)
        p_guess = optax.apply_updates(p_guess, updates)

        jnp.set_printoptions(3)
        if epoch % 10 == 0:
            y_guess = trafo.latent_to_observed(p_guess)
            y_true = trafo.latent_to_observed(p_true)
            print(f"Epoch={epoch:4d}, estim={y_guess}, true={y_true}")


def loss_data_fit(solve, *, ssm):
    """Create a loss that measures the data fit."""

    def loss(y0, std_log, output_scale, inputs, labels):
        std = jnp.exp(std_log) * output_scale
        std_ts = jnp.ones_like(inputs)[:, None] * std[None, ...]

        loss_lml = probdiffeq.loss_lml_timeseries(ssm=ssm)
        sol = solve(y0, save_at=inputs, output_scale=output_scale)

        diff = sol.u.mean[0] - labels
        return jnp.mean(diff**2), sol

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
