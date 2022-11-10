"""Interface for extrapolations."""

import abc
from typing import Generic, Tuple, TypeVar

import jax
import jax.numpy as jnp

R = TypeVar("R")  # think: random variables
C = TypeVar("C")  # think: my personal cache


@jax.tree_util.register_pytree_node_class
class AbstractExtrapolation(abc.ABC, Generic[R, C]):
    """Extrapolation model interface."""

    def tree_flatten(self):
        return (), ()

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls()

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def __eq__(self, other):
        equal = jax.tree_util.tree_map(lambda a, b: jnp.all(a == b), self, other)
        return jax.tree_util.tree_all(equal)

    @abc.abstractmethod
    def init_corrected(self, *, taylor_coefficients) -> R:
        raise NotImplementedError

    @abc.abstractmethod
    def init_output_scale_sqrtm(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def init_error_estimate(self) -> jax.Array:
        raise NotImplementedError

    @abc.abstractmethod
    def begin_extrapolation(self, m0, /, *, dt) -> Tuple[R, C]:
        raise NotImplementedError

    @abc.abstractmethod
    def complete_extrapolation(
        self, *, linearisation_pt: R, l0, cache: C, output_scale_sqrtm
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def revert_markov_kernel(
        self, *, linearisation_pt: R, l0, cache: C, output_scale_sqrtm
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def condense_backward_models(
        self, *, transition_init, noise_init: R, transition_state, noise_state: R
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def marginalise_backwards(self, *, init: R, linop, noise: R):
        raise NotImplementedError

    @abc.abstractmethod
    def marginalise_model(self, *, init: R, linop, noise: R):
        raise NotImplementedError

    @abc.abstractmethod
    def init_conditional(self, *, rv_proto):
        raise NotImplementedError
