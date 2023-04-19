"""Test."""
import jax
import jax.numpy as jnp

from probdiffeq import _adaptive, ivpsolvers
from probdiffeq.backend import testing
from probdiffeq.statespace import recipes
from probdiffeq.strategies import smoothers


@testing.fixture(name="setup")
def fixture_setup():
    impl = recipes.ts0_iso(num_derivatives=3)
    strategy = smoothers.Smoother(*impl)
    solver = ivpsolvers.DynamicSolver(strategy)
    adaptive = _adaptive.AdaptiveIVPSolver(solver)

    tcoeffs = [jnp.arange(1.0, 3.0)] * 4
    sol = solver.solution_from_tcoeffs(tcoeffs, t=1.0, output_scale=2.0)
    t, post = sol.t, sol.posterior
    return adaptive, (t, post)


def test_adaptive_extract_init_inverse(setup):
    """Assert Adaptive.extract(Adaptive.init(x)) == x."""
    adaptive, (t, post) = setup

    solution = (t, post, t, post, 2.0, 4.0), (1.0, 2.0)
    state = adaptive.init(*solution)
    solution_again = adaptive.extract(state)

    assert _tree_all_allclose(solution_again, solution)


def test_adaptive_init_extract_inverse(setup):
    """Assert Adaptive.init(Adaptive.extract(x)) == x."""
    adaptive, (t, post) = setup

    solution = (t, post, t, post, 2.0, 4.0), (1.0, 0.5)
    state = adaptive.init(*solution)

    @jax.tree_util.Partial
    def vf(x, t, p):
        return x

    # A few steps so the state changes sufficiently
    # for the test below to be meaningful
    for _ in range(4):
        state = adaptive.step(state, vector_field=vf, t1=100.0, parameters=())

    extracted = adaptive.extract(state)
    state_reinitialised = adaptive.init(*extracted)

    with jax.disable_jit():
        # print(state.error_norm_proposed)
        # print(state_reinitialised.error_norm_proposed)
        # print()
        # print(state.control)
        # print(state_reinitialised.control)
        # print()
        # print(state.proposed)
        # print(state_reinitialised.proposed)
        print()
        print(state.accepted)
        print(state_reinitialised.accepted)
        print()
        print(state.solution)
        print(state_reinitialised.solution)

        print()
        print()
        print()
        print()
        print()
        print()
        print(state)
        print(state_reinitialised)
    assert _tree_all_allclose(state_reinitialised, state)


def _tree_all_allclose(tree1, tree2, **kwargs):
    trees_is_allclose = _tree_allclose(tree1, tree2, **kwargs)
    return jax.tree_util.tree_all(trees_is_allclose)


def _tree_allclose(tree1, tree2, **kwargs):
    def allclose_partial(*args):
        return jnp.allclose(*args, **kwargs)

    return jax.tree_util.tree_map(allclose_partial, tree1, tree2)
