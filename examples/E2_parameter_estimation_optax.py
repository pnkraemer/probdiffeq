"""Estimate parameters (via Optax).

We create some data,
compute the marginal likelihood of this data _under the ODE posterior_
(which is something you cannot do with non-probabilistic solvers!),
and optimize the parameters with `optax`.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from diffeqzoo import backend, ivps

from probdiffeq import ivpsolve, probdiffeq

if not backend.has_been_selected:
    backend.select("jax")  # ivp examples in jax

# Fail this notebook on NaN detection (to catch those in the CI)
jax.config.update("jax_debug_nans", True)


def main():
    """Learn an ODE with Optax."""
    # Create a problem and some fake-data:

    # +
    f, u0, (t0, t1), f_args = ivps.lotka_volterra()
    f_args = jnp.asarray(f_args)

    def vf(y, t, *, p):  # noqa: ARG001
        """Evaluate the Lotka-Volterra vector field."""
        return f(y, *p)

    grid = jnp.linspace(t0, t1, endpoint=True, num=200)
    solve, ssm = solver(vf, u0, grid=grid)

    parameter_true = f_args + 0.05
    parameter_guess = f_args
    solution_true = solve(parameter_true)
    data = solution_true.u.mean[0]

    # We make an initial guess, but it does not lead to a good data fit:
    initial = solve(parameter_guess)

    # Use probdiffeq to compute a parameter-to-data fit function.
    loss = loss_marginal_likelihood(solve=solve, data=data, ssm=ssm)
    value_and_grad = jax.jit(jax.value_and_grad(loss))

    # We can differentiate the function forward- and reverse-mode
    # (the latter is possible because we use fixed steps)
    print("Value and gradient:")
    print(value_and_grad(parameter_guess))

    # Enter Optax:
    print()
    print("Training:")
    optim = optax.adam(learning_rate=1e-2)
    update = build_update(optimizer=optim, value_and_grad=value_and_grad)
    p = parameter_guess
    state = optim.init(p)
    for i in range(10):
        for _ in range(10):
            p, state = update(p, state)

        print(f"After {(i + 1) * 10} iterations:", p)

    # The solution looks much better:
    final = solve(p)
    _fig, ax = plt.subplots(figsize=(5, 3), dpi=120, constrained_layout=True)
    ax.set_title("Learning a Predator-Prey model", fontsize="medium")
    ax.set_xlabel("Predators", fontsize="medium")
    ax.set_ylabel("Prey", fontsize="medium")
    ax.plot(
        data[:, 0], data[:, 1], "o", markersize=8, label="Data", color="k", alpha=0.2
    )
    ax.plot(
        initial.u.mean[0][:, 0],
        initial.u.mean[0][:, 1],
        color="C0",
        label="Initial guess",
    )
    ax.plot(final.u.mean[0][:, 0], final.u.mean[0][:, 1], color="C1", label="Optimised")
    ax.legend(fontsize="small")
    plt.show()


def solver(vf, u0, *, grid):
    """Construct a solver."""
    ssm = probdiffeq.state_space_model(ssm_fact="isotropic")

    def solve(p):
        """Evaluate the parameter-to-solution map."""
        tcoeffs = (u0, vf(u0, grid[0], p=p))
        init, iwp = probdiffeq.prior_wiener_integrated(
            tcoeffs, ssm=ssm, output_scale=10.0
        )

        @probdiffeq.ode
        def vf_p(y, /, *, t):
            return vf(y, t=t, p=p)

        ts0 = probdiffeq.constraint_ode_ts0(vf_p, ssm=ssm)
        strategy = probdiffeq.strategy_smoother_fixedinterval()
        solver_obj = probdiffeq.solver(strategy=strategy, prior=iwp, constraint=ts0)
        solve_fn = ivpsolve.solve_fixed_grid(solver=solver_obj)
        return solve_fn(init, grid=grid)

    return solve, ssm


def loss_marginal_likelihood(*, data, solve, ssm, std=1e-1):
    """Create a loss function."""
    loss_lml = probdiffeq.loss_lml_timeseries(ssm=ssm)

    @jax.jit
    def loss(params, /):
        """Evaluate the data fit as a function of the parameters."""
        sol = solve(params)
        std_array = jnp.ones_like(sol.t) * std
        lml = loss_lml(data, std=std_array, posterior=sol.solution_full)
        return -lml

    return loss


# +
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
