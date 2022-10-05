"""Initial value problem solvers."""
import abc
from typing import Any, NamedTuple

import jax.numpy as jnp

from odefilter.prob import ibm


class AbstractIVPSolver(abc.ABC):
    """Abstract solver for IVPs."""

    @abc.abstractmethod
    def init_fn(self, *, ivp, params):
        """Initialise the IVP solver state."""
        raise NotImplementedError

    @abc.abstractmethod
    def step_fn(self, state, *, ode_function, t1, params):
        """Perform a step."""
        raise NotImplementedError


def ek0(*, init, num_derivatives):
    """EK0 solver."""
    init_alg, init_params = init

    alg = _EK0(init=init_alg)

    a, q_sqrtm = ibm.system_matrices_1d(num_derivatives=num_derivatives)
    params = _EK0.Params(
        num_derivatives=num_derivatives, init=init_params, a=a, q_sqrtm=q_sqrtm
    )
    return alg, params


class _EK0(AbstractIVPSolver):
    """EK0."""

    class State(NamedTuple):
        t: float
        u: Any
        error_estimate: Any

        hidden_state: Any

    class Params(NamedTuple):
        num_derivatives: int
        init: Any

        a: Any
        q_sqrtm: Any

    def __init__(self, *, init):
        self.init = init

    def init_fn(self, *, ivp, params):
        f, u0, t0 = ivp.ode_function.f, ivp.initial_values, ivp.t0
        m0_mat = self.init(f=f, u0=u0, num_derivatives=params.num_derivatives)
        m0_mat = m0_mat[:, None]
        c_sqrtm0 = jnp.zeros((params.num_derivatives + 1, params.num_derivatives + 1))

        return self.State(
            t=t0, u=m0_mat[0], hidden_state=(m0_mat, c_sqrtm0), error_estimate=jnp.nan
        )

    def step_fn(self, *, state, ode_function, dt, params):
        t, (m0, c_sqrtm0) = state.t, state.hidden_state

        p, p_inv = ibm.preconditioner_diagonal(
            dt=dt, num_derivatives=params.num_derivatives
        )

        u_new, _, error_estimate = _attempt_step_forward_only(
            f=ode_function.f,
            m=m0,
            c_sqrtm=c_sqrtm0,
            p=p,
            p_inv=p_inv,
            a=params.a,
            q_sqrtm=params.q_sqrtm,
        )
        error_estimate = dt * error_estimate

        t_new = t + dt
        return self.State(
            t=t_new, u=u_new[0][0], hidden_state=u_new, error_estimate=error_estimate
        )
