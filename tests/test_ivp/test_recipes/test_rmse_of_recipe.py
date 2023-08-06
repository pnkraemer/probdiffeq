"""Assert that every recipe yields a decent ODE approximation."""
import diffeqzoo.ivps
import diffrax
import jax
import jax.numpy as jnp

from probdiffeq import ivpsolve, test_util
from probdiffeq.backend import statespace, testing
from probdiffeq.statespace import cubature, recipes


@testing.fixture(name="problem")
def fixture_problem():
    f, u0, (t0, _), f_args = diffeqzoo.ivps.lotka_volterra()
    t1 = 2.0  # Short time-intervals are sufficient for this test.

    @jax.jit
    def vf(x, *, t, p):  # pylint: disable=unused-argument
        return f(x, *p)

    return vf, u0, (t0, t1), f_args


@testing.case()
def case_ts0_iso():
    def ts0_iso_factory(ode_shape, num_derivatives):
        return recipes.ts0_iso(num_derivatives=num_derivatives)

    return ts0_iso_factory, 1.0


@testing.case()
def case_ts0_blockdiag():
    return recipes.ts0_blockdiag, jnp.ones((2,))


@testing.case()
def case_ts1_dense():
    return recipes.ts1_dense, 1.0


@testing.case()
def case_ts0_dense():
    return recipes.ts0_dense, 1.0


@testing.case()
def case_slr1_dense():
    return recipes.slr1_dense, 1.0


@testing.case()
def case_slr1_dense_gauss_hermite():
    def recipe(**kwargs):
        return recipes.slr1_dense(cubature_rule_fn=cubature.gauss_hermite, **kwargs)

    return recipe, 1.0


@testing.case()
def case_slr0_dense():
    return recipes.slr0_dense, 1.0


@testing.fixture(name="solution")
@testing.parametrize_with_cases("factorisation", cases=".", prefix="case_")
def fixture_recipe_solution(problem, factorisation):
    vf, u0, (t0, t1), f_args = problem

    problem_args = (vf, (u0,))
    problem_kwargs = {"t0": t0, "t1": t1, "parameters": f_args}
    impl_factory, output_scale = factorisation
    solver = test_util.generate_solver(
        num_derivatives=2, impl_factory=impl_factory, ode_shape=jnp.shape(u0)
    )
    adaptive_kwargs = {
        "solver": solver,
        "atol": 1e-2,
        "rtol": 1e-2,
        "output_scale": output_scale,
    }
    solution = ivpsolve.simulate_terminal_values(
        *problem_args, **problem_kwargs, **adaptive_kwargs
    )
    return solution


@testing.fixture(name="diffrax_solution")
def fixture_diffrax_solution(problem):
    vf, u0, (t0, t1), f_args = problem

    # Solve the IVP
    @jax.jit
    def vf_diffrax(t, y, args):
        return vf(y, t=t, p=args)

    term = diffrax.ODETerm(vf_diffrax)
    solver = diffrax.Dopri5()
    solution_object = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=0.1,
        y0=u0,
        args=f_args,
        saveat=diffrax.SaveAt(dense=True),
        stepsize_controller=diffrax.PIDController(atol=1e-10, rtol=1e-10),
    )

    def solution(t):
        return solution_object.evaluate(t)

    return solution


def test_terminal_value_simulation_matches_diffrax(solution, diffrax_solution):
    expected = diffrax_solution(solution.t)
    received = solution.u

    assert jnp.allclose(received, expected, rtol=1e-2)
