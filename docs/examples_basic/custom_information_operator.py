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

from probdiffeq import ivpsolve, probdiffeq, taylor

t0, t1 = 0.0, 5.0


@jax.jit
def vf_1st(y, t):  # noqa: ARG001
    u, du = jnp.split(y, 2)
    return jnp.concatenate([du, -u])


def hamiltonian_1st(y):
    u, du = jnp.split(y, 2)
    kinetic = 0.5 * jnp.dot(du, du)
    potential = 0.5 * jnp.dot(u, u)
    return kinetic + potential


u0 = jnp.array([1.0, 0.0, 0.0, 1.0])

zeros, ones = jnp.zeros_like(u0), jnp.ones_like(u0)
tcoeffs = [u0, zeros, zeros, zeros]
tcoeffs_std = [zeros, ones, ones, ones]

init, ibm, ssm = probdiffeq.prior_wiener_integrated(
    tcoeffs, tcoeffs_std=tcoeffs_std, output_scale=1.0, ssm_fact="dense"
)
ts1 = probdiffeq.constraint_ode_ts1(ssm=ssm)
strategy = probdiffeq.strategy_smoother_fixedinterval(ssm=ssm)
solver_1st = probdiffeq.solver_mle(
    vf_1st, strategy=strategy, prior=ibm, constraint=ts1, ssm=ssm
)
solve = ivpsolve.solve_fixed_grid(solver=solver_1st)

# -

grid = jnp.linspace(t0, t1, endpoint=True, num=150)
solution = jax.jit(solve)(init, grid=grid)
hamiltonian_drift = jax.vmap(hamiltonian_1st)(solution.u.mean[0]) - hamiltonian_1st(u0)

fig, ax = plt.subplots(ncols=2, figsize=(8, 3), constrained_layout=True)
ax[0].plot(solution.t, solution.u.mean[0], marker=".")
ax[1].semilogy(solution.t, jnp.abs(hamiltonian_drift))

plt.show()

# The default configuration assumes that the ODE to be solved is of first order.
# Now, the same game with a second-order ODE

# +


@jax.jit
def vf_2(y, dy, *, t):  # noqa: ARG001
    """Evaluate the three-body problem as a second-order IVP."""
    return -y


def hamiltonian_2(u, du):
    """
    Energy:
    H = 1/2 |du|^2 + 1/2 |u|^2
    """
    kinetic = 0.5 * jnp.dot(du, du)
    potential = 0.5 * jnp.dot(u, u)
    return kinetic + potential


u0, du0 = jnp.split(u0, 2)

# # One derivative more than above because we don't transform to first order
tcoeffs = taylor.odejet_padded_scan(lambda *ys: vf_2(*ys, t=t0), (u0, du0), num=3)
init, ibm, ssm = probdiffeq.prior_wiener_integrated(
    tcoeffs, output_scale=1.0, ssm_fact="dense"
)
# ts0 = probdiffeq.constraint_root_ts1(root, ode_order=2, ssm=ssm)
# strategy = probdiffeq.strategy_filter(ssm=ssm)
# solver_2nd = probdiffeq.solver_mle(
#     vf_2, strategy=strategy, prior=ibm, constraint=ts0, ssm=ssm
# )
# errorest_2nd = probdiffeq.errorest_local_residual_cached(prior=ibm, ssm=ssm)

# # -

# solve = ivpsolve.solve_adaptive_save_at(solver=solver_2nd, errorest=errorest_2nd)
# solution = jax.jit(solve)(init, save_at=save_at, atol=1e-5, rtol=1e-5)


# fig, ax = plt.subplots(ncols=2, figsize=(8, 3), constrained_layout=True)
# ax[0].plot(solution.u.mean[0][:, 0], solution.u.mean[0][:, 1], marker=".")
# ax[1].semilogy(
#     solution.t,
#     jnp.abs(
#         jax.vmap(hamiltonian)(solution.u.mean[0], solution.u.mean[1])
#         - hamiltonian(u0, du0)
#     ),
# )
# plt.show()


# # -


def root(vf, *u_and_du_and_ddu):
    *u_and_du, ddu = u_and_du_and_ddu
    deriv = ddu - vf(*u_and_du)
    hamil = hamiltonian_2(*u_and_du) - hamiltonian_2(u0, du0)
    return jnp.concatenate([deriv, 100 * hamil[None]])


ts0 = probdiffeq.constraint_root_ts1(root, ode_order=2, ssm=ssm)
strategy = probdiffeq.strategy_smoother_fixedpoint(ssm=ssm)
solver_2nd = probdiffeq.solver_mle(
    vf_2, strategy=strategy, prior=ibm, constraint=ts0, ssm=ssm
)
errorest_2nd = probdiffeq.errorest_local_residual_cached(prior=ibm, ssm=ssm)

solve = ivpsolve.solve_adaptive_save_at(solver=solver_2nd, errorest=errorest_2nd)
solution = jax.jit(solve)(init, save_at=save_at, atol=1e-2, rtol=1e-2, damp=1e-8)
print(solution)

fig, ax = plt.subplots(ncols=2, figsize=(8, 3), constrained_layout=True)
ax[0].plot(solution.u.mean[0][:, 0], solution.u.mean[0][:, 1], marker=".")
ax[1].semilogy(
    solution.t,
    1e-8
    + jnp.abs(
        jax.vmap(hamiltonian_2)(solution.u.mean[0], solution.u.mean[1])
        - hamiltonian_2(u0, du0)
    ),
)
plt.show()
