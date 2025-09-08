# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Diffusion tempering & NODEs
#

# +
"""Train a neural ODE with ProbDiffEq and Optax using diffusion tempering."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

from probdiffeq import ivpsolve, ivpsolvers, stats


def main(num_data=100, epochs=1_000, print_every=100, hidden=(20,), lr=0.2):
    """Train a neural ODE using diffusion tempering."""
    # Create some data and construct a neural ODE
    grid = jnp.linspace(0, 1, num=num_data)
    data = jnp.sin(2.5 * jnp.pi * grid) * jnp.pi * grid
    stdev = 1e-1
    output_scale = 1e2
    vf, u0, (t0, t1), f_args = vf_neural_ode(hidden=hidden, t0=0.0, t1=1)

    # Create a loss (this is where probabilistic numerics enters!)
    loss = loss_log_marginal_likelihood(vf=vf, t0=t0)
    loss0, info0 = loss(
        f_args, u0=u0, grid=grid, data=data, stdev=stdev, output_scale=output_scale
    )

    # Plot the data and the initial guess
    plt.title(f"Initial estimate | Loss: {loss0:.2f}")
    plt.plot(grid, data, "x", label="Data", color="C0")
    plt.plot(grid, info0["sol"].u[0], "-", label="Estimate", color="C1")
    plt.legend()
    plt.show()

    # Construct an optimiser
    optim = optax.adam(lr)
    train_step = train_step_optax(optim, loss=loss)

    # Train the model
    state = optim.init(f_args)
    print("Loss after...")
    for i in range(epochs):
        (f_args, state), info = train_step(
            f_args,
            state,
            u0=u0,
            grid=grid,
            data=data,
            stdev=stdev,
            output_scale=output_scale,
        )

        # Print progressbar
        if i % print_every == print_every - 1:
            print(f"...{(i + 1)} epochs: loss={info['loss']:.3e}")

        # Diffusion tempering: https://arxiv.org/abs/2402.12231
        # To all users: Adjust this tempering and
        # see how it affects parameter estimation.
        if i % 100 == 0:
            output_scale /= 10.0

    # Plot the results
    plt.title(f"Final estimate | Loss: {info['loss']:.2f}")
    plt.plot(grid, data, "x", label="Data", color="C0")
    plt.plot(grid, info0["sol"].u[0], "-", label="Initial estimate", color="C1")
    plt.plot(grid, info["sol"].u[0], "-", label="Final estimate", color="C2")
    plt.legend()
    plt.show()


def vf_neural_ode(*, hidden: tuple, t0: float, t1: float):
    """Build a neural ODE."""
    f_args, mlp = model_mlp(hidden=hidden, shape_in=(2,), shape_out=(1,))
    u0 = jnp.asarray([0.0])

    @jax.jit
    def vf(y, *, t, p):
        """Evaluate the neural ODE vector field."""
        y_and_t = jnp.concatenate([y, t[None]])
        return mlp(p, y_and_t)

    return vf, (u0,), (t0, t1), f_args


def model_mlp(
    *, hidden: tuple, shape_in: tuple = (), shape_out: tuple = (), activation=jnp.tanh
):
    """Construct an MLP."""
    assert len(shape_in) <= 1
    assert len(shape_out) <= 1

    shape_prev = shape_in
    weights = []
    for h in hidden:
        W = jnp.empty((h, *shape_prev))
        b = jnp.empty((h,))
        shape_prev = (h,)
        weights.append((W, b))

    W = jnp.empty((*shape_out, *shape_prev))
    b = jnp.empty(shape_out)
    weights.append((W, b))

    p_flat, unravel = jax.flatten_util.ravel_pytree(weights)

    def fwd(w, x):
        for A, b in w[:-1]:
            x = jnp.dot(A, x) + b
            x = activation(x)

        A, b = w[-1]
        return jnp.dot(A, x) + b

    key = jax.random.PRNGKey(1)
    p_init = jax.random.normal(key, shape=p_flat.shape, dtype=p_flat.dtype)
    return unravel(p_init), fwd


def loss_log_marginal_likelihood(vf, *, t0):
    """Build a loss function from an ODE problem."""

    @jax.jit
    def loss(
        p: jax.Array,
        *,
        u0: tuple,
        grid: jax.Array,
        data: jax.Array,
        stdev: jax.Array,
        output_scale: jax.Array,
    ):
        """Loss function: log-marginal likelihood of the data."""
        # Build a solver
        tcoeffs = (*u0, vf(*u0, t=t0, p=p))
        init, ibm, ssm = ivpsolvers.prior_wiener_integrated(
            tcoeffs, output_scale=output_scale, ssm_fact="isotropic"
        )
        ts0 = ivpsolvers.correction_ts0(lambda *a, **kw: vf(*a, **kw, p=p), ssm=ssm)
        strategy = ivpsolvers.strategy_smoother(ssm=ssm)
        solver_ts0 = ivpsolvers.solver(strategy, prior=ibm, correction=ts0, ssm=ssm)

        # Solve
        sol = ivpsolve.solve_fixed_grid(init, grid=grid, solver=solver_ts0, ssm=ssm)

        # Evaluate loss
        marginal_likelihood = stats.log_marginal_likelihood(
            data[:, None],
            standard_deviation=jnp.ones_like(grid) * stdev,
            posterior=sol.posterior,
            ssm=sol.ssm,
        )
        return -1 * marginal_likelihood, {"sol": sol}

    return loss


def train_step_optax(optimizer, loss):
    """Implement a training step using Optax."""

    @jax.jit
    def update(params, opt_state, **loss_kwargs):
        """Update the optimiser state."""
        value_and_grad = jax.value_and_grad(loss, argnums=0, has_aux=True)
        (value, info), grads = value_and_grad(params, **loss_kwargs)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        info["loss"] = value
        return (params, opt_state), info

    return update


if __name__ == "__main__":
    main()
