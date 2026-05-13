"""Get started.

Solve the logistic equation and explore different solvers.
"""

import jax
import jax.numpy as jnp

from probdiffeq import diffeqjet, ivpsolve, probdiffeq

# Fail this notebook on NaN detection (to catch those in the CI)
jax.config.update("jax_debug_nans", True)


def main():
    """Solve the logistic equation."""
    # Define a differential equation

    @jax.jit
    def vf(y, /, *, t):
        """Evaluate the dynamics of the logistic ODE."""
        del t  # unused argument
        return 2 * y * (1 - y)

    u0 = jnp.asarray(0.1)
    t0, t1 = 0.0, 5.0

    # Set up a state-space model over Taylor coefficients

    ssm = probdiffeq.state_space_model()

    # Build a solver
    tcoeffs = diffeqjet.odejet_padded_scan(lambda y: vf(y, t=t0), (u0,), num=1)
    init, iwp = probdiffeq.prior_wiener_integrated(tcoeffs, ssm=ssm)
    ts = probdiffeq.constraint_ode_ts1(vf, ssm=ssm)
    strategy = probdiffeq.strategy_filter(ssm=ssm)
    solver = probdiffeq.solver_mle(ssm=ssm, strategy=strategy, prior=iwp, constraint=ts)
    error = probdiffeq.error_residual_std(constraint=ts, prior=iwp, ssm=ssm)

    # Solve the ODE. Try different solution routines.

    save_at = jnp.linspace(t0, t1, num=100, endpoint=True)
    solve = ivpsolve.solve_adaptive_save_at(solver=solver, error=error)
    solution = jax.jit(solve)(init, save_at, atol=1e-3, rtol=1e-3)

    print(f"\ninitial = {jax.tree.map(jnp.shape, init)}")
    print(f"\nsolution = {jax.tree.map(jnp.shape, solution)}")


if __name__ == "__main__":
    main()
