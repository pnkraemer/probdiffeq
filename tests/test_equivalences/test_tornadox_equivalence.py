"""Make sure that the results are equivalent to tornadox."""

import diffeqzoo.ivps
import jax
import jax.numpy as jnp
import pytest_cases
from tornadox import ek0, ek1, init, ivp, step

from odefilter import controls, ivpsolve, solvers
from odefilter.implementations import implementations
from odefilter.strategies import filters


@pytest_cases.fixture(scope="session", name="num")
def fixture_num():
    return 4


@pytest_cases.fixture(scope="session", name="control_params")
def fixture_control_params():
    return 0.2, 10.0, 0.95


@pytest_cases.fixture(scope="session", name="vanderpol")
def fixture_vanderpol():
    # van-der-Pol as a setup. We really don't want stiffness here.
    return diffeqzoo.ivps.van_der_pol_first_order(
        time_span=(0.0, 1.0), stiffness_constant=1.0
    )


@pytest_cases.fixture(scope="session", name="ivp_tornadox")
def fixture_ivp_tornadox(vanderpol):
    f, u0, (t0, t1), f_args = vanderpol

    @jax.jit
    def vf_tor(t, y):
        return f(y, *f_args)

    return ivp.InitialValueProblem(
        f=vf_tor, t0=t0, tmax=t1, y0=u0, df=jax.jit(jax.jacfwd(vf_tor, argnums=1))
    )


@pytest_cases.fixture(scope="session", name="ivp_odefilter")
def fixture_ivp_odefilter(vanderpol):
    f, u0, (t0, t1), f_args = vanderpol

    # ODE-filter
    @jax.jit
    def vf_ode(y, *, t, p):
        return f(y, *p)

    return vf_ode, (u0,), (t0, t1), f_args


@pytest_cases.fixture(scope="session", name="steprule_tornadox")
def fixture_steprule_tornadox(tolerances, control_params):
    atol, rtol = tolerances
    factor_min, factor_max, safety = control_params
    return step.AdaptiveSteps(
        max_changes=(factor_min, factor_max),
        safety_scale=safety,
        abstol=atol,
        reltol=rtol,
    )


@pytest_cases.fixture(scope="session", name="controller_odefilter")
def fixture_controller_odefilter(control_params):
    factor_min, factor_max, safety = control_params
    return controls.ClippedIntegral(
        safety=safety, factor_min=factor_min, factor_max=factor_max
    )


@pytest_cases.fixture(scope="session", name="solver_tornadox_kronecker_ek0")
def fixture_solver_tornadox_kronecker_ek0(num, steprule_tornadox):
    return ek0.KroneckerEK0(
        initialization=init.TaylorMode(),
        num_derivatives=num,
        steprule=steprule_tornadox,
    )


@pytest_cases.fixture(scope="session", name="solver_odefilter_kronecker_ek0")
def fixture_solver_odefilter_kronecker_ek0(num):
    implementation = implementations.IsoTS0.from_params(num_derivatives=num)
    strategy = filters.Filter(implementation=implementation)
    return solvers.DynamicSolver(strategy=strategy)


@pytest_cases.case
def case_solver_pair_kronecker_ek0(
    tolerances,
    ivp_tornadox,
    ivp_odefilter,
    solver_tornadox_kronecker_ek0,
    solver_odefilter_kronecker_ek0,
    controller_odefilter,
):
    # Solve with tornadox
    solution_tornadox = solver_tornadox_kronecker_ek0.solve(ivp_tornadox)

    # Solve with odefilter
    atol, rtol = tolerances
    vf_ode, u0, (t0, t1), f_args = ivp_odefilter
    solution_odefilter = ivpsolve.solve(
        vf_ode,
        initial_values=u0,
        t0=t0,
        t1=t1,
        solver=solver_odefilter_kronecker_ek0,
        atol=atol,
        rtol=rtol,
        control=controller_odefilter,
        parameters=f_args,
        reference_state_fn=lambda x, y: jnp.abs(x),
    )

    # Get both solutions into the same format

    @jax.vmap
    def cov(x):
        return x @ x.T

    d = u0[0].shape[0]

    @jax.vmap
    def kroncov(x):
        return jnp.kron(jnp.eye(d), x @ x.T)

    output_tornadox = (
        solution_tornadox.t,
        solution_tornadox.mean,
        cov(solution_tornadox.cov_sqrtm),
    )
    solution_odefilter = (
        solution_odefilter.t,
        solution_odefilter.marginals.mean,
        kroncov(solution_odefilter.marginals.cov_sqrtm_lower),
    )
    return output_tornadox, solution_odefilter


@pytest_cases.fixture(scope="session", name="solver_tornadox_reference_ek1")
def fixture_solver_tornadox_reference_ek1(num, steprule_tornadox):
    return ek1.ReferenceEK1(
        initialization=init.TaylorMode(),
        num_derivatives=num,
        steprule=steprule_tornadox,
    )


@pytest_cases.fixture(scope="session", name="solver_odefilter_reference_ek1")
def fixture_solver_odefilter_reference_ek1(num):
    implementation = implementations.TS1.from_params(
        num_derivatives=num, ode_dimension=2
    )
    strategy = filters.Filter(implementation=implementation)
    return solvers.DynamicSolver(strategy=strategy)


@pytest_cases.case
def case_solver_pair_reference_ek1(
    tolerances,
    ivp_tornadox,
    ivp_odefilter,
    solver_tornadox_reference_ek1,
    solver_odefilter_reference_ek1,
    controller_odefilter,
):
    # Solve with tornadox
    solution_tornadox = solver_tornadox_reference_ek1.solve(ivp_tornadox)

    # Solve with odefilter
    atol, rtol = tolerances
    vf_ode, u0, (t0, t1), f_args = ivp_odefilter
    solution_odefilter = ivpsolve.solve(
        vf_ode,
        initial_values=u0,
        t0=t0,
        t1=t1,
        solver=solver_odefilter_reference_ek1,
        atol=atol,
        rtol=rtol,
        control=controller_odefilter,
        parameters=f_args,
    )

    # Get both into the same format

    @jax.vmap
    def cov(x):
        return x @ x.T

    @jax.vmap
    def kronmean(x):
        return jnp.reshape(x, (-1,), order="F")

    output_tornadox = (
        solution_tornadox.t,
        kronmean(solution_tornadox.mean),
        cov(solution_tornadox.cov_sqrtm),
    )
    solution_odefilter = (
        solution_odefilter.t,
        solution_odefilter.marginals.mean,
        cov(solution_odefilter.marginals.cov_sqrtm_lower),
    )
    return output_tornadox, solution_odefilter


@pytest_cases.parametrize_with_cases(
    "solution_tornadox, solution_odefilter", cases=".", prefix="case_solver_pair_"
)
def test_outputs_equal(solution_tornadox, solution_odefilter):
    for sol_tornadox, sol_odefilter in zip(solution_tornadox, solution_odefilter):
        assert jnp.allclose(sol_tornadox, sol_odefilter)
