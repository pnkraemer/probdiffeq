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

# # Second-order systems

# +
"""Demonstrate how to implement custom information operators."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from probdiffeq import ivpsolve, probdiffeq

# TODO: always add machine epsilon to initial Tcoeffs?
# TODO: demo mass matrices?
# TODO: think about adaptive steps
# TODO: fix all linter issues and tests
# TODO: do a root ts1 by stop_gradient'ing the vf?


# Define the problem


@jax.jit
def vf_1st(y, t):  # noqa: ARG001
    u, du = jnp.split(y, 2)
    return jnp.concatenate([du, -u])


def hamiltonian_1st(y):
    u, du = jnp.split(y, 2)
    kinetic = 0.5 * jnp.dot(du, du)
    potential = 0.5 * jnp.dot(u, u)
    return kinetic + potential


@jax.jit
def vf_2nd(y, dy, *, t):  # noqa: ARG001
    """Evaluate the three-body problem as a second-order IVP."""
    return -y


def hamiltonian_2nd(u, du):
    kinetic = 0.5 * jnp.dot(du, du)
    potential = 0.5 * jnp.dot(u, u)
    return kinetic + potential


t0, t1 = 0.0, 1_000.0  # reeeeally long time-integration
u0_1st = jnp.array([1.0, 0.0, 0.0, 1.0])
save_at = jnp.linspace(t0, t1, endpoint=True, num=500)


# Hamiltonian at t=0 (a good solution conserves it)
H0 = hamiltonian_1st(u0_1st)


# +

# Set up the first-order solver (for reference)

zeros, ones = jnp.zeros_like(u0_1st), jnp.ones_like(u0_1st)
tcoeffs = [u0_1st, zeros, zeros]
tcoeffs_std = [zeros, ones, ones]
init, ibm, ssm = probdiffeq.prior_wiener_integrated(tcoeffs, tcoeffs_std=tcoeffs_std)
ts1 = probdiffeq.constraint_ode_ts1(ssm=ssm)
strategy = probdiffeq.strategy_smoother_fixedpoint(ssm=ssm)
solver_1st = probdiffeq.solver_mle(
    vf_1st, strategy=strategy, prior=ibm, constraint=ts1, ssm=ssm
)
errorest = probdiffeq.errorest_local_residual_cached(prior=ibm, ssm=ssm)
solve = ivpsolve.solve_adaptive_save_at(solver=solver_1st, errorest=errorest)

solution_1 = jax.jit(solve)(init, save_at=save_at, atol=1e-4, rtol=1e-2)
hamiltonian_1 = jax.vmap(hamiltonian_1st)(solution_1.u.mean[0])


# +


# Set up the custom information operator.
# We know: (i) the ODE is second order; (ii) the Hamiltonian should be conserved


def root(vf, u, du, ddu):
    deriv = ddu - vf(u, du)
    hamil = hamiltonian_2nd(u, du) - H0
    return [deriv, hamil]  # any PyTree goes


u0, du0 = jnp.split(u0_1st, 2)

zeros, ones = jnp.zeros_like(u0), jnp.ones_like(u0)
tcoeffs = [u0, du0, zeros]
tcoeffs_std = [zeros, 1e-8 + zeros, ones]  # avoid NaNs
init, ibm, ssm = probdiffeq.prior_wiener_integrated(tcoeffs, tcoeffs_std=tcoeffs_std)
ts1 = probdiffeq.constraint_root_ts1(root, ssm=ssm, ode_order=2)
strategy = probdiffeq.strategy_smoother_fixedpoint(ssm=ssm)
solver_2nd = probdiffeq.solver_mle(
    vf_2nd, strategy=strategy, prior=ibm, constraint=ts1, ssm=ssm
)
error_norm = probdiffeq.errorest_error_norm_rms_then_scale()
errorest = probdiffeq.errorest_local_residual_cached(
    prior=ibm, ssm=ssm, error_norm=error_norm
)
solve = ivpsolve.solve_adaptive_save_at(solver=solver_2nd, errorest=errorest)

solution_2 = jax.jit(solve)(init, save_at=save_at, atol=1e-4, rtol=1e-2)
hamiltonian_2 = jax.vmap(hamiltonian_2nd)(solution_2.u.mean[0], solution_2.u.mean[1])


# Plot solution
fig, ax = plt.subplots(ncols=2, figsize=(8, 3), constrained_layout=True)

ax[0].set_title("Differential equation solution")
ax[0].plot(
    solution_1.u.mean[0][:, 0], solution_1.u.mean[0][:, 1], ".", label="Standard solver"
)
ax[0].plot(
    solution_2.u.mean[0][:, 0], solution_2.u.mean[0][:, 1], ".", label="Custom root"
)
ax[0].legend()
ax[0].set_xlabel("$x_1$")
ax[0].set_ylabel("$x_2$")

eps = jnp.finfo(solution_2.t).eps
ax[1].set_title("Hamiltonian error")
ax[1].semilogy(solution_1.t, eps + jnp.abs(hamiltonian_1 - H0), label="Standard solver")
ax[1].semilogy(solution_2.t, eps + jnp.abs(hamiltonian_2 - H0), label="Custom root")
ax[1].set_xlabel("Time $t$")
ax[1].set_ylabel("Error")
ax[1].legend()
plt.show()

plt.show()
