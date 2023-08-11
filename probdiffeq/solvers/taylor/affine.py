r"""Taylor-expand the solution of an initial value problem (IVP)."""

import functools
from typing import Callable, Tuple

import jax
import jax.experimental.jet
import jax.experimental.ode


@functools.partial(jax.jit, static_argnames=["vector_field", "num"])
def affine_recursion(*, vector_field: Callable, initial_values: Tuple, num: int, t):
    """Evaluate the Taylor series of an affine differential equation.

    !!! warning "Compilation time"
        JIT-compiling this function unrolls a loop of length `num`.

    """
    if num == 0:
        return initial_values

    vf = jax.tree_util.Partial(vector_field, t=t)
    fx, jvp_fn = jax.linearize(vf, *initial_values)

    tmp = fx
    fx_evaluations = [tmp := jvp_fn(tmp) for _ in range(num - 1)]
    return [*initial_values, fx, *fx_evaluations]
