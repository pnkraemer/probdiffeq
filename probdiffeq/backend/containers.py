"""Container types."""

import jax_dataclasses

# jax_dataclasses is one of few dataclass implementations that can handle
# nested dataclasses properly.
# We need A LOT OF nested dataclasses.
#
# What about the others?
# chex.dataclasses requires keyword-only initialisation
# tjax requires a very recent JAX version (which appears unnecessary for using it here)
# flax.struct.dataclass works, and jax_dataclasses.pytree_dataclass works.
# I prefer the jax_dataclasses dependency because it is a much smaller dependency.#
dataclass_pytree_node = jax_dataclasses.pytree_dataclass
