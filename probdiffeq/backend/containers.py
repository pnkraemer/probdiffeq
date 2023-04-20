"""Container types."""

import jax_dataclasses

# jax_dataclasses is the only implementation that can handle
# nested dataclasses properly. We need A LOT OF nested dataclasses.
# chex.dataclasses did not manage, flax.dataclasses came with some
# warnings that I did not want to introduce, and tjax required
# a very recent JAX version, which I found a bit of a harsh
# dependency restriction for such a small feature.
dataclass_pytree_node = jax_dataclasses.pytree_dataclass
