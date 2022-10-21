"""Markov processes."""


from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util

T = TypeVar("T")
"""A type-variable to alias appropriate Normal-like random variables."""


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BackwardModel(Generic[T]):
    """Backward model for backward-Gauss--Markov process representations."""

    transition: Any
    noise: T

    def tree_flatten(self):
        children = self.transition, self.noise
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Posterior(Generic[T]):
    """Markov sequences as smoothing solutions."""

    t: float
    t_previous: float

    u: Any
    marginals: T
    marginals_filtered: T
    backward_model: BackwardModel[T]

    diffusion_sqrtm: float

    def tree_flatten(self):
        children = (
            self.t,
            self.t_previous,
            self.u,
            self.marginals,
            self.marginals_filtered,
            self.backward_model,
            self.diffusion_sqrtm,
        )
        aux = ()
        return children, aux

    @classmethod
    def tree_unflatten(cls, _aux, children):
        (
            t,
            t_previous,
            u,
            marginals,
            marginals_filtered,
            backward_model,
            diffusion_sqrtm,
        ) = children
        return cls(
            t=t,
            t_previous=t_previous,
            u=u,
            marginals=marginals,
            marginals_filtered=marginals_filtered,
            backward_model=backward_model,
            diffusion_sqrtm=diffusion_sqrtm,
        )

    def __len__(self):
        """Length of a solution object.

        Depends on the length of the underlying :attr:`t` attribute.
        """
        if jnp.ndim(self.t) < 1:
            raise ValueError("Solution object not batched :(")
        return self.t.shape[0]

    def __getitem__(self, item):
        """Access the `i`-th sub-solution."""
        if jnp.ndim(self.t) < 1:
            raise ValueError(f"Solution object not batched :(, {jnp.ndim(self.t)}")
        if isinstance(item, tuple) and len(item) > jnp.ndim(self.t):
            # s[2, 3] forbidden
            raise ValueError(f"Inapplicable shape: {item, jnp.shape(self.t)}")
        return Posterior(
            t=self.t[item],
            t_previous=self.t_previous[item],
            u=self.u[item],
            diffusion_sqrtm=self.diffusion_sqrtm[item],
            # todo: make iterable?
            marginals=jax.tree_util.tree_map(lambda x: x[item], self.marginals),
            # todo: make iterable?
            marginals_filtered=jax.tree_util.tree_map(
                lambda x: x[item], self.marginals_filtered
            ),
            # todo: make iterable?
            backward_model=jax.tree_util.tree_map(
                lambda x: x[item], self.backward_model
            ),
        )

    def __iter__(self):
        """Iterate through the filtering solution."""
        for i in range(self.t.shape[0]):
            yield self[i]
