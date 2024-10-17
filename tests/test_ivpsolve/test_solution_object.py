"""Tests for interaction with the solution object."""

from probdiffeq import ivpsolve, ivpsolvers, taylor
from probdiffeq.backend import containers, functools, ode, testing
from probdiffeq.backend import numpy as np
from probdiffeq.backend.typing import Array


class Taylor(containers.NamedTuple):
    """A non-standard Taylor-coefficient data structure."""

    state: Array
    velocity: Array
    acceleration: Array


@testing.fixture(name="approximate_solution")
@testing.parametrize("fact", ["dense", "blockdiag", "isotropic"])
def fixture_approximate_solution(fact):
    vf, u0, (t0, t1) = ode.ivp_lotka_volterra()

    # Generate a solver
    tcoeffs = Taylor(*taylor.odejet_padded_scan(lambda y: vf(y, t=t0), u0, num=2))
    ibm, ssm = ivpsolvers.prior_ibm(tcoeffs, ssm_fact=fact)

    ts0 = ivpsolvers.correction_ts0(ssm=ssm)
    strategy = ivpsolvers.strategy_filter(ibm, ts0, ssm=ssm)
    solver = ivpsolvers.solver_mle(strategy, ssm=ssm)
    asolver = ivpsolve.adaptive(solver, atol=1e-2, rtol=1e-2, ssm=ssm)

    init = solver.initial_condition()
    return ivpsolve.solve_adaptive_save_every_step(
        vf, init, t0=t0, t1=t1, dt0=0.1, adaptive_solver=asolver, ssm=ssm
    )


def test_u_inherits_data_structure(approximate_solution):
    assert isinstance(approximate_solution.u, Taylor)

    solution_t1 = approximate_solution[-1]
    assert isinstance(solution_t1.u, Taylor)


def test_getitem_possible_for_terminal_values(approximate_solution):
    solution_t1 = approximate_solution[-1]
    assert isinstance(solution_t1, type(approximate_solution))


@testing.parametrize("item", [-2, 0, slice(1, -1, 1)])
def test_getitem_impossible_for_nonterminal_values(approximate_solution, item):
    with testing.raises(ValueError, match="non-terminal"):
        _ = approximate_solution[item]


@testing.parametrize("item", [-1, -2, 0, slice(1, -1, 1)])
def test_getitem_impossible_at_single_time_for_any_item(approximate_solution, item):
    # Allowed slicing:
    # solution_t1 is not batched now, so further slicing should be impossible
    solution_t1 = approximate_solution[-1]

    with testing.raises(ValueError, match="not batched"):
        _ = solution_t1[item]


def test_iter_impossible(approximate_solution):
    with testing.raises(ValueError, match="not batched"):
        for _ in approximate_solution:
            pass


@testing.fixture(name="approximate_solution_batched")
@testing.parametrize("fact", ["dense", "blockdiag", "isotropic"])
def fixture_approximate_solution_batched(fact):
    vf, (u0,), (t0, t1) = ode.ivp_lotka_volterra()

    # Generate a solver
    save_at = np.linspace(t0, t1, endpoint=True, num=4)

    def solve(init):
        tcoeffs = (init, vf(init, t=None))
        ibm, ssm = ivpsolvers.prior_ibm(tcoeffs, ssm_fact=fact)

        ts0 = ivpsolvers.correction_ts0(ssm=ssm)
        strategy = ivpsolvers.strategy_filter(ibm, ts0, ssm=ssm)
        solver = ivpsolvers.solver_mle(strategy, ssm=ssm)
        adaptive_solver = ivpsolve.adaptive(solver, atol=1e-2, rtol=1e-2, ssm=ssm)

        initcond = solver.initial_condition()
        return ivpsolve.solve_adaptive_save_at(
            vf,
            initcond,
            save_at=save_at,
            adaptive_solver=adaptive_solver,
            dt0=0.1,
            ssm=ssm,
        )

    u0_batched = u0[None, ...]
    solve = functools.vmap(solve)
    return solve(u0_batched)


def test_batched_getitem_possible(approximate_solution_batched):
    solution_type = type(approximate_solution_batched)
    for idx in (0,):
        approximate_solution = approximate_solution_batched[idx]
        assert isinstance(approximate_solution, solution_type)
        assert np.allclose(approximate_solution.t, approximate_solution_batched.t[idx])

        for u1, u2 in zip(approximate_solution.u, approximate_solution_batched.u[idx]):
            assert np.allclose(u1, u2)


def test_batched_iter_possible(approximate_solution_batched):
    solution_type = type(approximate_solution_batched)
    for idx, approximate_solution in enumerate(approximate_solution_batched):
        assert isinstance(approximate_solution, solution_type)
        assert np.allclose(approximate_solution.t, approximate_solution_batched.t[idx])
        for u1, u2 in zip(approximate_solution.u, approximate_solution_batched.u[idx]):
            assert np.allclose(u1, u2)
