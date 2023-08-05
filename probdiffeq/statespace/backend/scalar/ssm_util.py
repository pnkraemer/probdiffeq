"""SSM utilities."""
import jax.numpy as jnp

from probdiffeq.statespace.backend import _ssm_util
from probdiffeq.statespace.backend.scalar import random


class SSMUtilBackEnd(_ssm_util.SSMUtilBackEnd):
    def stack_tcoeffs(self, tcoeffs, /, num_derivatives):
        if len(tcoeffs) != num_derivatives + 1:
            msg1 = "The number of Taylor coefficients does not match "
            msg2 = "the number of derivatives in the implementation."
            raise ValueError(msg1 + msg2)
        m0_matrix = jnp.stack(tcoeffs)
        m0_corrected = jnp.reshape(m0_matrix, (-1,), order="F")
        c_sqrtm0_corrected = jnp.zeros((num_derivatives + 1, num_derivatives + 1))
        return random.Normal(m0_corrected, c_sqrtm0_corrected)
