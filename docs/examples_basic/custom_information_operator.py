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


t0, t1 = 0.0, 5.0
u0_1st = jnp.array([1.0, 0.0, 0.0, 1.0])


# Hamiltonian at t=0 (a good solution conserves it)
H0 = hamiltonian_1st(u0_1st)


# +

# Set up the first-order solver (for reference)

zeros, ones = jnp.zeros_like(u0_1st), jnp.ones_like(u0_1st)
tcoeffs = [u0_1st, zeros, zeros, zeros]
tcoeffs_std = [zeros, ones, ones, ones]
init, ibm, ssm = probdiffeq.prior_wiener_integrated(tcoeffs, tcoeffs_std=tcoeffs_std)
ts1 = probdiffeq.constraint_ode_ts1(ssm=ssm)
strategy = probdiffeq.strategy_smoother_fixedinterval(ssm=ssm)
solver_1st = probdiffeq.solver_mle(
    vf_1st, strategy=strategy, prior=ibm, constraint=ts1, ssm=ssm
)
solve = ivpsolve.solve_fixed_grid(solver=solver_1st)

# +

grid = jnp.linspace(t0, t1, endpoint=True, num=100)
solution = jax.jit(solve)(init, grid=grid)
hamiltonian = jax.vmap(hamiltonian_1st)(solution.u.mean[0])


# +

fig, ax = plt.subplots(ncols=2, figsize=(8, 3), constrained_layout=True)
ax[0].plot(solution.t, solution.u.mean[0], marker=".")
ax[1].semilogy(solution.t, jnp.abs(hamiltonian - H0))
ax[1].set_ylim((1e-8, 1e-3))

plt.show()


# +


# Set up the custom information operator.
# We know: (i) the ODE is second order; (ii) the Hamiltonian should be conserved


def root(vf, *u_and_du_and_ddu):
    *u_and_du, ddu = u_and_du_and_ddu
    deriv = ddu - vf(*u_and_du)
    hamil = hamiltonian_2nd(*u_and_du) - H0
    return [deriv, hamil]


u0, du0 = jnp.split(u0_1st, 2)

zeros, ones = jnp.zeros_like(u0), jnp.ones_like(u0)
tcoeffs = [u0, du0, zeros, zeros]
tcoeffs_std = [zeros, 1e-10 + zeros, ones, ones]  # avoid NaNs
init, ibm, ssm = probdiffeq.prior_wiener_integrated(tcoeffs, tcoeffs_std=tcoeffs_std)
ts1 = probdiffeq.constraint_root_ts1(root, ssm=ssm, ode_order=2)
strategy = probdiffeq.strategy_smoother_fixedinterval(ssm=ssm)
solver_2nd = probdiffeq.solver_mle(
    vf_2nd, strategy=strategy, prior=ibm, constraint=ts1, ssm=ssm
)
solve = ivpsolve.solve_fixed_grid(solver=solver_2nd)


grid = jnp.linspace(t0, t1, endpoint=True, num=100)
solution = jax.jit(solve)(init, grid=grid)
hamiltonian = jax.vmap(hamiltonian_2nd)(solution.u.mean[0], solution.u.mean[1])

fig, ax = plt.subplots(ncols=2, figsize=(8, 3), constrained_layout=True)
ax[0].plot(solution.t, solution.u.mean[0], marker=".")
ax[0].plot(solution.t, solution.u.mean[1], marker=".")
ax[1].semilogy(solution.t, jnp.abs(hamiltonian - H0))
ax[1].set_ylim((1e-8, 1e-3))
plt.show()

plt.show()
