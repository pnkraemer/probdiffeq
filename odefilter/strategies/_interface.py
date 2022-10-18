"""Inference interface."""

import abc
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class Strategy(abc.ABC):
    """Inference strategy interface."""

    implementation: Any

    def tree_flatten(self):
        children = (self.implementation,)
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)

    @abc.abstractmethod
    def init_fn(self, *, taylor_coefficients, t0):  # -> state
        raise NotImplementedError

    @abc.abstractmethod
    def step_fn(self, *, state, info_op, dt, parameters):
        raise NotImplementedError

    @abc.abstractmethod
    def extract_fn(self, *, state):  # -> solution
        # Don't jit the extract fun unless it is performance-critical!
        # Why? Because it would recompile every time the ODE parameters
        # change in solve(), provided the change is sufficient to change
        # the number of time-steps taken.
        # In the fully-jit-able functions, it is compiled automatically anyway.
        raise NotImplementedError

    @jax.jit
    def interpolate_fn(self, *, s0, s1, t):  # noqa: D102

        # Cases to switch between
        branches = [
            self._case_right_corner,
            self._case_interpolate,
        ]

        # Which case applies
        is_right_corner = (s1.t - t) ** 2 <= 1e-10
        is_in_between = jnp.logical_not(is_right_corner)

        index_as_array, *_ = jnp.where(
            jnp.asarray([is_right_corner, is_in_between]), size=1
        )
        index = jnp.reshape(index_as_array, ())
        return jax.lax.switch(index, branches, s0, s1, t)

    @abc.abstractmethod
    def _case_right_corner(self, s0, s1, t):
        raise NotImplementedError

    @abc.abstractmethod
    def _case_interpolate(self, s0, s1, t):
        raise NotImplementedError

    @abc.abstractmethod
    def dense_output(self, *, t, state, state_previous):
        raise NotImplementedError

    @abc.abstractmethod  # kwargs are for jax.numpy.searchsorted()
    def dense_output_searchsorted(self, *, ts, solution, **kwargs):
        raise NotImplementedError
