"""Taylor coefficients as central data structures."""

import collections

import jax
import jax.numpy as jnp

from probdiffeq import ivpsolve, probdiffeq

# Fail this notebook on NaN detection (to catch those in the CI)
jax.config.update("jax_debug_nans", True)


def main():
    """Explore different Taylor coefficients."""
    # We start by defining an ODE.

    @probdiffeq.ode
    def vf(y, /, *, t):
        """Evaluate the dynamics of the logistic ODE."""
        del t  # unused argument
        return 2 * y * (1 - y)

    u0 = jnp.asarray(0.1)
    t0, t1 = 0.0, 5.0

    # Here is a wrapper arounds Probdiffeq's solution routine.

    # It's time to solve some ODEs:

    jetexpand = probdiffeq.jetexpand_ode_padded_scan(num=2)
    tcoeffs_list, _ = jetexpand(vf, (u0,), t=t0)
    solution = jax.jit(solve, static_argnums=[0])(vf, tcoeffs_list, t0=t0, t1=t1)

    print()
    print("Probabilistic solution:")
    print(jax.tree.map(jnp.shape, solution))

    # The type of solution.u matches that of the initial condition.

    print()
    print("Solution matches initial condition:")
    print(jax.tree.map(jnp.shape, tcoeffs_list))
    print(jax.tree.map(jnp.shape, solution.u))

    # Anything that behaves like a list work.
    # For example, we can use lists or tuples, but also named tuples.

    CustomTCoeffs = collections.namedtuple(
        "CustomTCoeffs", ["state", "velocity", "acceleration"]
    )
    tcoeffs = CustomTCoeffs(*tcoeffs_list)
    solution = jax.jit(solve, static_argnums=[0])(vf, tcoeffs, t0=t0, t1=t1)

    print()
    print("The target is a named tuple:")
    print(jax.tree.map(jnp.shape, tcoeffs))
    print(jax.tree.map(jnp.shape, solution))
    print(jax.tree.map(jnp.shape, solution.u))

    # The same applies to statistical quantities that we can extract from the solution.
    # For example, the standard deviation or samples from the solution object:

    key = jax.random.PRNGKey(seed=15)
    ssm = probdiffeq.state_space_model(ssm_fact="dense")
    posterior = solution.solution_full
    sample_one = posterior.sample(key, ssm=ssm)
    sample_many = posterior.sample(key, ssm=ssm, shape=(1, 2, 3))

    print()
    print("Samples inherit structure:")
    print(jax.tree.map(jnp.shape, solution.u.mean))
    print(jax.tree.map(jnp.shape, solution.u.std))
    print(jax.tree.map(jnp.shape, sample_one))
    print(jax.tree.map(jnp.shape, sample_many))
    assert isinstance(solution.u.mean, CustomTCoeffs)
    assert isinstance(solution.u.std, CustomTCoeffs)
    assert isinstance(sample_one, CustomTCoeffs)
    assert isinstance(sample_many, CustomTCoeffs)


def solve(vf, tc, *, t0, t1):
    """Solve the ODE."""
    ssm = probdiffeq.state_space_model(ssm_fact="dense")
    init, prior = probdiffeq.prior_wiener_integrated(tc, ssm=ssm)
    ts0 = probdiffeq.constraint_ode_ts0(vf, ssm=ssm)
    strategy = probdiffeq.strategy_smoother_fixedpoint(ssm=ssm)
    solver = probdiffeq.solver_mle(
        strategy=strategy, prior=prior, constraint=ts0, ssm=ssm
    )
    ts = jnp.linspace(t0, t1, endpoint=True, num=10)
    error = probdiffeq.error_residual_std(constraint=ts0, prior=prior, ssm=ssm)
    solve_fn = ivpsolve.solve_adaptive_save_at(solver=solver, error=error)
    return solve_fn(init, save_at=ts, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    main()
