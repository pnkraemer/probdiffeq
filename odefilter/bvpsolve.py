"""BVP solver."""

from functools import partial

import jax


@partial(jax.jit, static_argnames=("f", "df", "L", "dL", "R", "dR", "solver"))
def solve_bvp_separable(*, f, df, L, dL, R, dR, mesh, init_guess, solver):

    init_fn, collocate_fn, extract_fn = solver

    state = init_fn(mesh=mesh, guess=init_guess)
    error_norm, state = collocate_fn(state, f=f, df=df, L=L, dL=dL, R=R, dR=dR)
    return extract_fn(state)


@partial(jax.jit, static_argnames=("f", "df", "bc", "dbc", "solver"))
def solve_bvp(*, f, df, bc, dbc, mesh, init_guess, solver):

    init_fn, collocate_fn, extract_fn = solver

    state = init_fn(mesh=mesh, guess=init_guess)
    error_norm, state = collocate_fn(state, f=f, df=df, bc=bc, dbc=dbc)
    return extract_fn(state)
