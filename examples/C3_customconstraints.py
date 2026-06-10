"""Customise the constraints.

This tutorial extends the demonstration of solving second-order problems directly.

For background information on what's presented below, see:

  > Bosch, Nathanael, Filip Tronarp, and Philipp Hennig.
    "Pick-and-mix information operators for probabilistic ODE solvers."
    International Conference on Artificial Intelligence and Statistics.
    PMLR, 2022.

"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from probdiffeq import ivpsolve, probdiffeq

# Fail this notebook on NaN detection (to catch those in the CI)
jax.config.update("jax_debug_nans", True)


def main():
    """Enforce the probabilistic solver to preserve Hamiltonians."""
    # Define the problem.
    # Solve at relatively poor tolerances to make the Hamiltonian drift more obvious.
    t0, t1 = 0.0, 100.0  # reeeeally long time-integration
    save_at = jnp.linspace(t0, t1, endpoint=True, num=500)
    atol, rtol = 1e-2, 1e-2

    u0_1st = jnp.array([1.0, 0.0, 0.0, 1.0])

    # A good solution conserves the Hamiltonian.

    H0 = 1.0

    # Set up the first-order solver (for illustration).
    tcoeffs = [u0_1st]
    ssm = probdiffeq.state_space_model_dense()
    init, iwp = ssm.prior_wiener_integrated(tcoeffs, diffuse_derivatives=2)
    ts1 = ssm.constraint_ode_ts1(vf_1st)
    strategy = probdiffeq.strategy_smoother_fixedpoint()
    solver_1st = probdiffeq.solver_mle(strategy=strategy, prior=iwp, constraint=ts1)
    error = probdiffeq.error_state_std(constraint=ts1, prior=iwp)
    solve = ivpsolve.solve_adaptive_save_at(solver=solver_1st, error=error)

    sol_1 = jax.jit(solve)(init, save_at=save_at, atol=atol, rtol=rtol)
    ham_1 = jax.vmap(hamiltonian_1st)(sol_1.u.mean[0])

    # The harmonic oscillator calls for a custom information operator because
    # we know: (i) the ODE is second order; (ii) the Hamiltonian should be conserved.

    # Set up the custom-residual solver.
    # We don't use high orders because high-order initialisation
    # of custom-information-operator solvers is an open problem.
    # But for low-order solvers, custom residuals work well.
    u0, du0 = jnp.split(u0_1st, 2)
    tcoeffs = [u0, du0]
    ssm = probdiffeq.state_space_model_dense()
    init, iwp = ssm.prior_wiener_integrated(tcoeffs, diffuse_derivatives=1)

    # Use this constraint function for custom residuals:
    residual_constraint = ssm.constraint_residual(residual)
    strategy = probdiffeq.strategy_smoother_fixedpoint()
    solver_2nd = probdiffeq.solver_mle(
        strategy=strategy, prior=iwp, constraint=residual_constraint
    )

    # Custom residuals with residual-based error estimates
    # require norming-then-scaling
    # (instead of scaling-then-norming, which is the default),
    # because scaling-then-norming assumes that the residual pytree
    # has the same structure as the target pytree.

    error = probdiffeq.error_state_std(constraint=residual_constraint, prior=iwp)
    solve = ivpsolve.solve_adaptive_save_at(solver=solver_2nd, error=error)
    sol_2 = jax.jit(solve)(init, save_at=save_at, atol=1e-2, rtol=1e-2)
    ham_2 = jax.vmap(hamiltonian_2nd)(sol_2.u.mean[0], sol_2.u.mean[1])

    # Plot both solutions.
    # See how much better the custom residual solver preserves the Hamiltonian?

    _fig, ax = plt.subplots(ncols=2, figsize=(8, 3), constrained_layout=True)

    ax[0].set_title("Differential equation solution", fontsize="medium")
    ax[0].plot(
        sol_1.u.mean[0][:, 0], sol_1.u.mean[0][:, 1], "-", label="Standard solver"
    )
    ax[0].plot(
        sol_2.u.mean[0][:, 0], sol_2.u.mean[0][:, 1], "-", label="Custom residual"
    )
    ax[0].legend()
    ax[0].set_xlabel("$x_1$")
    ax[0].set_ylabel("$x_2$")

    eps = jnp.sqrt(jnp.finfo(sol_2.t).eps)
    ax[1].set_title("Hamiltonian error", fontsize="medium")
    ax[1].semilogy(sol_1.t, eps + jnp.abs(ham_1 - H0), label="Standard solver")
    ax[1].semilogy(sol_2.t, eps + jnp.abs(ham_2 - H0), label="Custom residual")
    ax[1].set_xlabel("Time $t$")
    ax[1].set_ylabel("Error")
    ax[1].legend()
    plt.show()


@probdiffeq.ode
def vf_1st(y, /, *, t):
    """Evaluate the harmonic oscillator dynamics."""
    u, du = jnp.split(y, 2)
    return jnp.concatenate([du, vf_2nd(u, du, t=t)])


def hamiltonian_1st(y):
    """Evaluate the Hamiltonian of the harmonic oscillator."""
    u, du = jnp.split(y, 2)
    return hamiltonian_2nd(u, du)


def hamiltonian_2nd(u, du):
    """Evaluate the Hamiltonian of the harmonic oscillator."""
    kinetic = 0.5 * jnp.dot(du, du)
    potential = 0.5 * jnp.dot(u, u)
    return kinetic + potential


@probdiffeq.residual_state_velocity_acceleration
def residual(u, du, ddu, /, *, t):
    """Evaluate a custom residual for the harmonic oscillator."""
    deriv = ddu - vf_2nd(u, du, t=t)
    hamil = hamiltonian_2nd(u, du) - 1.0
    return [deriv, hamil]  # any PyTree goes


def vf_2nd(y, dy, *, t):  # noqa: ARG001
    """Evaluate the harmonic oscillator as a 2nd-order problem."""
    return -y


if __name__ == "__main__":
    main()
