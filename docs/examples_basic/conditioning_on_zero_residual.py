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

# # How probabilistic solvers work
#
# Probabilistic solvers condition a prior distribution
# on satisfying a zero-ODE-residual on a specified grid.
#

# +
"""Demonstrate how probabilistic solvers work via conditioning on constraints."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffeqzoo import backend

from probdiffeq import ivpsolve, probdiffeq, taylor

# +
if not backend.has_been_selected:
    backend.select("jax")  # ivp examples in jax


# +
# Create an ODE problem


@jax.jit
def vector_field(y, t):  # noqa: ARG001
    """Evaluate the logistic ODE vector field."""
    return 10.0 * y * (2.0 - y)


t0, t1 = 0.0, 0.5
u0 = jnp.asarray([0.1])

# +
# Assemble the discretised prior (with and without the correct Taylor coefficients)

NUM_DERIVATIVES = 2
tcoeffs_mean = [u0] * (NUM_DERIVATIVES + 1)
tcoeffs_std = [jnp.ones_like(u0)] * (NUM_DERIVATIVES + 1)
ts = jnp.linspace(t0, t1, num=500, endpoint=True)
markov_seq_prior, ssm = probdiffeq.prior_wiener_integrated_discrete(
    ts, tcoeffs_mean, tcoeffs_std=tcoeffs_std, output_scale=100.0, ssm_fact="dense"
)

tcoeffs = taylor.odejet_padded_scan(
    lambda y: vector_field(y, t=t0), (u0,), num=NUM_DERIVATIVES
)
markov_seq_tcoeffs, _ssm = probdiffeq.prior_wiener_integrated_discrete(
    ts, tcoeffs, output_scale=100.0, ssm_fact="dense"
)

# +
# Compute the posterior

init, ibm, ssm = probdiffeq.prior_wiener_integrated(
    tcoeffs, output_scale=1.0, ssm_fact="dense"
)
ts1 = probdiffeq.constraint_ode_ts1(ssm=ssm)
strategy = probdiffeq.strategy_smoother_fixedpoint(ssm=ssm)
solver = probdiffeq.solver(
    vector_field, strategy=strategy, prior=ibm, constraint=ts1, ssm=ssm
)
errorest = probdiffeq.errorest_local_residual_cached(prior=ibm, ssm=ssm)

dt0 = ivpsolve.dt0(lambda y: vector_field(y, t=t0), (u0,))
solve = ivpsolve.solve_adaptive_save_at(solver=solver, errorest=errorest)
sol = solve(init, save_at=ts, dt0=dt0, atol=1e-1, rtol=1e-1)
markov_seq_posterior = sol.posterior


# +
# Compute samples

num_samples = 5
key = jax.random.PRNGKey(seed=1)
samples_prior = strategy.markov_sample(
    key, markov_seq_prior, shape=(num_samples,), reverse=False
)
samples_tcoeffs = strategy.markov_sample(
    key, markov_seq_tcoeffs, shape=(num_samples,), reverse=False
)
samples_posterior = strategy.markov_sample(
    key, markov_seq_posterior, shape=(num_samples,), reverse=True
)

# +
# Plot the results

fig, (axes_state, axes_residual, axes_log_abs) = plt.subplots(
    nrows=3, ncols=3, sharex=True, sharey="row", constrained_layout=True, figsize=(8, 5)
)
axes_state[0].set_title("Prior")
axes_state[1].set_title("w/ Initial condition")
axes_state[2].set_title("Posterior")

sample_style = {"marker": "None", "alpha": 0.99, "linewidth": 0.75}


def residual(x, t):
    """Evaluate the ODE residual."""
    return x[1] - jax.vmap(jax.vmap(vector_field), in_axes=(0, None))(x[0], t)


residual_prior = residual(samples_prior, ts)
residual_tcoeffs = residual(samples_tcoeffs, ts)
residual_posterior = residual(samples_posterior, ts)


for i in range(num_samples):
    # Plot all state-samples
    axes_state[0].plot(ts, samples_prior[0][i, ..., 0], **sample_style, color="C0")
    axes_state[1].plot(ts, samples_tcoeffs[0][i, ..., 0], **sample_style, color="C1")
    axes_state[2].plot(ts, samples_posterior[0][i, ..., 0], **sample_style, color="C2")

    # Plot all residual-samples
    axes_residual[0].plot(ts, residual_prior[i, ...], **sample_style, color="C0")
    axes_residual[1].plot(ts, residual_tcoeffs[i, ...], **sample_style, color="C1")
    axes_residual[2].plot(ts, residual_posterior[i, ...], **sample_style, color="C2")

    # Plot all log-residual samples
    axes_log_abs[0].plot(
        ts, jnp.log10(jnp.abs(residual_prior))[i, ...], **sample_style, color="C0"
    )
    axes_log_abs[1].plot(
        ts, jnp.log10(jnp.abs(residual_tcoeffs))[i, ...], **sample_style, color="C1"
    )
    axes_log_abs[2].plot(
        ts, jnp.log10(jnp.abs(residual_posterior))[i, ...], **sample_style, color="C2"
    )


# Set the x- and y-ticks/limits
axes_state[0].set_xticks((t0, (t0 + t1) / 2, t1))
axes_state[0].set_xlim((t0, t1))

axes_state[0].set_ylim((-1, 3))
axes_state[0].set_yticks((-1, 1, 3))

axes_residual[0].set_ylim((-10.0, 20))
axes_residual[0].set_yticks((-10.0, 5, 20))

axes_log_abs[0].set_ylim((-6, 4))
axes_log_abs[0].set_yticks((-6, -1, 4))

# Label the x- and y-axes
axes_state[0].set_ylabel("Solution")
axes_residual[0].set_ylabel("Residual")
axes_log_abs[0].set_ylabel(r"Log-residual")
axes_log_abs[0].set_xlabel("Time $t$")
axes_log_abs[1].set_xlabel("Time $t$")
axes_log_abs[2].set_xlabel("Time $t$")

# Show the result
fig.align_ylabels()
plt.show()
