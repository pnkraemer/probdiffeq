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

# # Taylor coefficients all the way up

# +
"""Demonstrate how central Taylor coefficient estimation is."""

import collections

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffeqzoo import backend, ivps

from probdiffeq import ivpsolve, ivpsolvers, taylor
from probdiffeq.util.doc_util import notebook

# -

plt.rcParams.update(notebook.plot_style())
plt.rcParams.update(notebook.plot_sizes())

# +
if not backend.has_been_selected:
    backend.select("jax")  # ivp examples in jax

jax.config.update("jax_platform_name", "cpu")
# -

# LOREM

# +
f, u0, (t0, t1), f_args = ivps.rigid_body()


def vf(*y, t):  # noqa: ARG001
    return f(*y, *f_args)


# -

# To build a probabilistic solver, we need to build a specific state-space model.
# To build this specific state-space model, we interact with Taylor coefficients.
# Taylor coefficients of ODE solutions are computed with the functions in Probdiffeq.

# +
tcoeffs = taylor.odejet_padded_scan(lambda *y: vf(*y, t=t0), [u0], num=2)
print(type(tcoeffs))
# -

# LOREM

# +

prior, ssm = ivpsolvers.prior_ibm(tcoeffs, ssm_fact="dense")
ts0 = ivpsolvers.correction_ts0(ssm=ssm)
strategy = ivpsolvers.strategy_filter(prior, ts0, ssm=ssm)
solver = ivpsolvers.solver_mle(strategy, ssm=ssm)
init = solver.initial_condition()

ts = jnp.linspace(t0, t1, endpoint=True, num=10)
adaptive_solver = ivpsolve.adaptive(solver, atol=1e-2, rtol=1e-2, ssm=ssm)
solution = ivpsolve.solve_adaptive_save_at(
    vf, init, save_at=ts, adaptive_solver=adaptive_solver, dt0=0.1, ssm=ssm
)
print(jax.tree.map(jnp.shape, solution))

# This corresponds to the initial Taylor coefficients:
print(jax.tree.map(jnp.shape, solution.u))


# -

# The solution field matches the type of the initial Taylor coeffient:

# +

# Anything that behaves like a list works
Taylor = collections.namedtuple("Taylor", ["state", "velocity", "acceleration"])

tcoeffs = Taylor(*tcoeffs)
print(jax.tree.map(jnp.shape, solution.u))


prior, ssm = ivpsolvers.prior_ibm(tcoeffs, ssm_fact="dense")
ts0 = ivpsolvers.correction_ts0(ssm=ssm)
strategy = ivpsolvers.strategy_filter(prior, ts0, ssm=ssm)
solver = ivpsolvers.solver_mle(strategy, ssm=ssm)
init = solver.initial_condition()

ts = jnp.linspace(t0, t1, endpoint=True, num=10)
adaptive_solver = ivpsolve.adaptive(solver, atol=1e-2, rtol=1e-2, ssm=ssm)
solution = ivpsolve.solve_adaptive_save_at(
    vf, init, save_at=ts, adaptive_solver=adaptive_solver, dt0=0.1, ssm=ssm
)
print(jax.tree.map(jnp.shape, solution))

# This corresponds to the initial Taylor coefficients:
print(jax.tree.map(jnp.shape, solution.u))

# LOREM
