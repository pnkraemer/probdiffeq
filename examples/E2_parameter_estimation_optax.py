"""Estimate parameters (via Optax).

We create some data,
compute the marginal likelihood of this data under the ODE posterior
(which is something deterministic solvers cannot do),
and optimize the parameters with `optax`.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

from probdiffeq import ivpsolve, probdiffeq

# Fail this notebook on NaN detection (to catch those in the CI)
jax.config.update("jax_debug_nans", True)

# Lotka-Volterra predator-prey model
u0 = jnp.asarray([20.0, 20.0])
t0, t1 = 0.0, 20.0
rate_constants = jnp.asarray([0.5, 0.05, 0.5, 0.05])  # (a, b, c, d)


def main():
    """Learn an ODE with Optax."""
    # Define the problem

    def vf(y, t, *, p):  # noqa: ARG001
        """Evaluate the Lotka-Volterra vector field."""
        a, b, c, d = p[0], p[1], p[2], p[3]
        return jnp.asarray([a * y[0] - b * y[0] * y[1], -c * y[1] + d * y[0] * y[1]])

    grid = jnp.linspace(t0, t1, endpoint=True, num=50)
    solve = solver(vf, u0, grid=grid)

    # Create a dataset
    parameter_true = rate_constants + 0.05
    parameter_guess = rate_constants
    solution_true = solve(parameter_true)
    data = solution_true.u.mean[0]

    # We make an initial guess, but it does not lead to a good data fit:
    initial = solve(parameter_guess)

    # Use probdiffeq to form the loss function:
    loss = loss_marginal_likelihood(solve=solve, data=data)
    value_and_grad = jax.jit(jax.value_and_grad(loss))

    # We can differentiate the function forward- and reverse-mode
    print("Value and gradient:")
    print(value_and_grad(parameter_guess))

    # Enter Optax:
    print()
    print("Training:")
    optim = optax.adam(learning_rate=1e-2)
    update = build_update(optimizer=optim, value_and_grad=value_and_grad)
    p = parameter_guess
    state = optim.init(p)
    for i in range(20):
        for _ in range(20):
            p, state = update(p, state)

        print(f"After {(i + 1) * 20} iterations:", p)

    # The solution looks much better:
    final = solve(p)
    fig, ax = plt.subplots(figsize=(5, 3), dpi=100, constrained_layout=True)
    ax.set_title("Learning a Predator-Prey model", fontsize="medium")
    ax.set_xlabel("Predators", fontsize="medium")
    ax.set_ylabel("Prey", fontsize="medium")
    ax.plot(
        data[:, 0], data[:, 1], "X", markersize=8, label="Data", color="k", alpha=0.2
    )
    ax.plot(
        initial.u.mean[0][:, 0],
        initial.u.mean[0][:, 1],
        color="C0",
        label="Initial guess",
        linestyle="dashed",
    )
    ax.plot(final.u.mean[0][:, 0], final.u.mean[0][:, 1], color="C1", label="Optimised")
    ax.legend(fontsize="small")
    fig.align_ylabels()
    plt.show()


def solver(vf, u0, *, grid):
    """Construct a solver."""
    ssm = probdiffeq.state_space_model_isotropic()
    strategy = probdiffeq.strategy_smoother_fixedpoint()

    def while_loop(cond, body, init):
        return eqx.internal.while_loop(cond, body, init, kind="bounded", max_steps=8)

    def solve(p):
        """Evaluate the parameter-to-solution map."""
        tcoeffs = (u0, vf(u0, grid[0], p=p))
        iwp = ssm.prior_wiener_integrated(tcoeffs, output_scale=10.0)

        @probdiffeq.ode
        def vf_p(y, /, *, t):
            return vf(y, t=t, p=p)

        ts0 = ssm.constraint_ode_ts0(vf_p)
        solver_obj = probdiffeq.solver(strategy=strategy, constraint=ts0)
        error = probdiffeq.error_state_std(constraint=ts0)
        solve_fn = ivpsolve.solve_adaptive_save_at(
            solver=solver_obj, error=error, while_loop=while_loop
        )
        return solve_fn(iwp, save_at=grid, atol=1e-4, rtol=1e-2)

    return solve


def loss_marginal_likelihood(*, data, solve, std=1e-1):
    """Create a loss function."""
    loss_lml = probdiffeq.loss_lml_timeseries()

    @jax.jit
    def loss(params, /):
        """Evaluate the data fit as a function of the parameters."""
        sol = solve(params)
        std_array = jnp.ones_like(sol.t) * std
        lml = loss_lml(data, std=std_array, posterior=sol.solution_full.posterior)
        return -lml

    return loss


def build_update(*, optimizer, value_and_grad):
    """Build a function for executing a single step in the optimization."""

    @jax.jit
    def update(params, opt_state):
        """Update the optimiser state."""
        _loss, grads = value_and_grad(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    return update


if __name__ == "__main__":
    main()
