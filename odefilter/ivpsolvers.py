"""Solvers for IVPs."""

import abc
from collections import namedtuple
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from odefilter import sqrtm
from odefilter.prob import ibm


class AbstractIVPSolver(abc.ABC):
    """Abstract solver for IVPs."""

    @abc.abstractmethod
    def init_fn(
        self,
        ivp,
        params,
    ):
        """Initialise the IVP solver state."""
        raise NotImplementedError

    @abc.abstractmethod
    def perform_step_fn(self, state0, *, ode_function, t1, params):
        """Perform a step."""
        raise NotImplementedError


def ek0(*, num_derivatives, control, init):
    """Create an ODEFilter that implements the EK0."""
    controller, control_params = control
    init_algorithm, init_params = init

    # Assemble solver
    alg = _EK0(
        num_derivatives=num_derivatives,
        control=controller,
        init=init_algorithm,
    )

    # Assemble parameters
    a, q_sqrtm = ibm.system_matrices_1d(num_derivatives=num_derivatives)
    params = _EK0Params(a=a, q_sqrtm=q_sqrtm, control=control_params, init=init_params)
    return alg, params


class _EK0Params(NamedTuple):
    a: Any
    q_sqrtm: Any

    control: Any
    init: Any


_Stats = namedtuple(
    "_Stats",
    (
        "f_evaluation_count",
        "steps_accepted_count",
        "steps_attempted_count",
        "dt_min",
        "dt_max",
    ),
)

_KroneckerEK0State = namedtuple(
    "_KroneckerEK0State", ("t", "u", "dt_proposed", "error_norm", "stats")
)
"""EK0 State."""


class _EK0:
    """The Kronecker EK0, but only for computing the terminal value.

    Uses adaptive steps and proportional control.
    Uses time-varying, scalar-valued diffusion.
    Uses Taylor-mode initialisation for num_derivatives >= 5,
    and forward-mode initialisation otherweise.
    """

    def __init__(self, *, control, init, num_derivatives=5):

        self.num_derivatives = num_derivatives
        self.control = control
        self.init = init

    def init_fn(self, *, ivp, params):

        f, u0, t0 = ivp.ode_function.f, ivp.initial_values, ivp.t0
        m0_mat = self.init(f=f, u0=u0, num_derivatives=self.num_derivatives)
        m0_mat = m0_mat[:, None]
        c_sqrtm0 = jnp.zeros((self.num_derivatives + 1, self.num_derivatives + 1))
        dt0 = self.control.propose_first_dt(f=f, u0=u0, params=params.control)

        stats = _Stats(
            f_evaluation_count=0,
            steps_accepted_count=0,
            steps_attempted_count=0,
            dt_min=jnp.inf,
            dt_max=0,
        )
        state = _KroneckerEK0State(
            t=t0,
            u=(m0_mat, c_sqrtm0),
            dt_proposed=dt0,
            error_norm=1.0,
            stats=stats,
        )
        return state

    def perform_step_fn(self, state0, *, ode_function, t1, params):
        """Perform a successful step."""
        larger_than_1 = 1.1
        init_val = _KroneckerEK0State(
            t=state0.t,
            u=state0.u,
            dt_proposed=state0.dt_proposed,
            error_norm=larger_than_1,
            stats=state0.stats,
        )
        state = jax.lax.while_loop(
            cond_fun=lambda s: s.error_norm > 1,
            body_fun=lambda s: self.attempt_step_fn(
                s,
                ode_function=ode_function,
                t1=t1,
                params=params,
            ),
            init_val=init_val,
        )

        dt_min = jnp.minimum(state.stats.dt_min, state.t - state0.t)
        dt_max = jnp.maximum(state.stats.dt_max, state.t - state0.t)
        stats = state.stats._replace(
            dt_min=dt_min,
            dt_max=dt_max,
            steps_accepted_count=state.stats.steps_accepted_count + 1,
        )
        state = _KroneckerEK0State(
            t=state.t,
            u=state.u,
            dt_proposed=state.dt_proposed,
            error_norm=state.error_norm,
            stats=stats,
        )
        return state

    def attempt_step_fn(self, state0, *, ode_function, t1, params):

        # Never exceed the terminal value
        dt_clipped = jnp.minimum(state0.dt_proposed, t1 - state0.t)
        t_new = state0.t + dt_clipped

        # Compute preconditioner and make a step
        u_new, error = self._attempt_step_fn(
            u0=state0.u,
            dt=dt_clipped,
            ode_function=ode_function,
            params=params,
        )

        # Normalise the error
        m_new, _ = u_new
        m0, _ = state0.u
        u1_ref = jnp.abs(jnp.maximum(m_new[0, :], m0[0, :]))
        error_norm = self.control.normalise_error(
            error=error, u1_ref=u1_ref, params=params.control
        )

        # Propose a new time-step
        error_norm_previously_accepted = state0.error_norm
        scale_factor = self.control.scale_factor(
            error_norm=error_norm,
            error_norm_previously_accepted=error_norm_previously_accepted,
            params=params.control,
        )
        dt_proposed = scale_factor * dt_clipped

        stats = state0.stats._replace(
            f_evaluation_count=state0.stats.f_evaluation_count + 1,
            steps_attempted_count=state0.stats.steps_attempted_count + 1,
        )
        state = _KroneckerEK0State(
            t=t_new,
            u=u_new,
            dt_proposed=dt_proposed,
            error_norm=error_norm,
            stats=stats,
        )
        return state

    def _attempt_step_fn(self, *, dt, u0, ode_function, params):
        m0, c_sqrtm0 = u0
        p, p_inv = ibm.preconditioner_diagonal(
            dt=dt, num_derivatives=self.num_derivatives
        )
        # Attempt step
        u_new, _, error = _attempt_step_forward_only(
            f=ode_function.f,
            m=m0,
            c_sqrtm=c_sqrtm0,
            p=p,
            p_inv=p_inv,
            a=params.a,
            q_sqrtm=params.q_sqrtm,
        )
        error = dt * error
        return u_new, error


def _propose_first_dt_per_tol(*, f, u0, num_derivatives, rtol, atol):
    # Taken from:
    # https://github.com/google/jax/blob/main/jax/experimental/ode.py
    #
    # which uses the algorithm from
    #
    # E. Hairer, S. P. Norsett G. Wanner,
    # Solving Ordinary Differential Equations I: Nonstiff Problems, Sec. II.4.
    f0 = f(u0)
    scale = atol + u0 * rtol
    a = jnp.linalg.norm(u0 / scale)
    b = jnp.linalg.norm(f0 / scale)
    dt0 = jnp.where((a < 1e-5) | (b < 1e-5), 1e-6, 0.01 * a / b)

    u1 = u0 + dt0 * f0
    f1 = f(u1)
    c = jnp.linalg.norm((f1 - f0) / scale) / dt0
    dt1 = jnp.where(
        (b <= 1e-15) & (c <= 1e-15),
        jnp.maximum(1e-6, dt0 * 1e-3),
        (0.01 / jnp.max(b + c)) ** (1.0 / (num_derivatives + 1)),
    )
    return jnp.minimum(100.0 * dt0, dt1)


def raw_ek0(*, init, num_derivatives):
    init_alg, init_params = init

    alg = _RawEK0(init=init_alg)

    a, q_sqrtm = ibm.system_matrices_1d(num_derivatives=num_derivatives)
    params = _RawEK0.Params(
        num_derivatives=num_derivatives, init=init_params, a=a, q_sqrtm=q_sqrtm
    )
    return alg, params


class _RawEK0:
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


"""
from odefilter import problems, ivpsolvers, inits, controls

ode = problems.FirstOrderODE(f=lambda x: x*(1-x))
ivp = problems.InitialValueProblem(ode_function=ode, initial_values=0.4, t0=0., t1=2.)
alg, params = ivpsolvers.raw_ek0(init=inits.taylor_mode(), num_derivatives=2)

state = alg.init_fn(ivp=ivp, params=params)
state = alg.step_fn(state=state, ode_function=ivp.ode_function, dt=0.1, params=params)


alg, params = ivpsolvers.adaptive(solver=(alg, params), error_order=3, control=controls.proportional_integral(), atol=1e-3, rtol=1e-3)
"""
