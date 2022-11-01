"""Make sure that the results are equivalent to tornadox."""

import jax
import jax.numpy as jnp
import pytest
import pytest_cases
from diffeqzoo import ivps
from tornadox import ek0, ek1, init, ivp, step

from odefilter import controls, ivpsolve, solvers
from odefilter.implementations import dense, isotropic
from odefilter.strategies import filters


@pytest_cases.case
def case_solver_pair_isotropic_ekf0(num, atol, rtol, factor_min, factor_max, safety):

    # van-der-Pol as a setup. We really don't want stiffness here.
    f, u0, (t0, t1), f_args = ivps.van_der_pol_first_order(
        time_span=(0.0, 1.0), stiffness_constant=1.0
    )

    # Tornadox

    @jax.jit
    def vf_tor(t, y):
        return f(y, *f_args)

    solver = ek0.KroneckerEK0(
        initialization=init.TaylorMode(),
        num_derivatives=num,
        steprule=step.AdaptiveSteps(
            max_changes=(factor_min, factor_max),
            safety_scale=safety,
            abstol=atol,
            reltol=rtol,
        ),
    )
    vdp = ivp.InitialValueProblem(f=vf_tor, t0=t0, tmax=t1, y0=u0)
    solution_tornadox = solver.solve(vdp)

    # ODE-filter
    @jax.jit
    def vf_ode(y, *, t, p):
        return f(y, *p)

    extrapolation = isotropic.IsoIBM.from_params(num_derivatives=num)
    correction = isotropic.IsoTaylorZerothOrder()
    ekf0_strategy = filters.Filter(extrapolation=extrapolation, correction=correction)
    ekf0 = solvers.DynamicSolver(strategy=ekf0_strategy)
    controller = controls.ClippedIntegral(
        safety=safety, factor_min=factor_min, factor_max=factor_max
    )
    solution_odefilter = ivpsolve.solve(
        vf_ode,
        initial_values=(u0,),
        t0=t0,
        t1=t1,
        solver=ekf0,
        atol=atol,
        rtol=rtol,
        control=controller,
        parameters=f_args,
        reference_state_fn=lambda x, y: jnp.abs(x),
    )

    @jax.vmap
    def cov(x):
        return x @ x.T

    d = u0.shape[0]

    @jax.vmap
    def kroncov(x):
        return jnp.kron(jnp.eye(d), x @ x.T)

    output_tornadox = (
        solution_tornadox.t,
        (solution_tornadox.mean),
        cov(solution_tornadox.cov_sqrtm),
    )
    solution_odefilter = (
        solution_odefilter.t,
        solution_odefilter.marginals.mean,
        kroncov(solution_odefilter.marginals.cov_sqrtm_lower),
    )
    return output_tornadox, solution_odefilter


@pytest_cases.case
def case_solver_pair_ekf1_dynamic(num, atol, rtol, factor_min, factor_max, safety):

    f, u0, (t0, t1), f_args = ivps.van_der_pol_first_order(
        time_span=(0.0, 1.0), stiffness_constant=4.0
    )

    @jax.jit
    def vf_tor(t, y):
        return f(y, *f_args)

    # Tornadox
    solver = ek1.ReferenceEK1(
        initialization=init.TaylorMode(),
        num_derivatives=num,
        steprule=step.AdaptiveSteps(
            max_changes=(factor_min, factor_max),
            safety_scale=safety,
            abstol=atol,
            reltol=rtol,
        ),
    )
    vdp = ivp.InitialValueProblem(
        f=vf_tor, t0=t0, tmax=t1, y0=u0, df=jax.jit(jax.jacfwd(vf_tor, argnums=1))
    )
    solution_tornadox = solver.solve(vdp)

    # ODE-filter
    @jax.jit
    def vf_ode(y, *, t, p):
        return f(y, *p)

    extrapolation = dense.IBM.from_params(ode_dimension=2, num_derivatives=num)
    correction = dense.TaylorFirstOrder(ode_dimension=2)
    ekf1_strategy = filters.Filter(extrapolation=extrapolation, correction=correction)
    ekf1 = solvers.DynamicSolver(strategy=ekf1_strategy)
    controller = controls.ClippedIntegral(
        safety=safety, factor_min=factor_min, factor_max=factor_max
    )
    solution_odefilter = ivpsolve.solve(
        vf_ode,
        initial_values=(u0,),
        t0=t0,
        t1=t1,
        solver=ekf1,
        atol=atol,
        rtol=rtol,
        control=controller,
        parameters=f_args,
        reference_state_fn=lambda x, y: jnp.maximum(jnp.abs(x), jnp.abs(y)),
    )

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


@pytest.mark.parametrize("num", [4])
@pytest.mark.parametrize("atol, rtol", [(1e-3, 1e-5)])
@pytest.mark.parametrize("factor_min, factor_max, safety", [(0.2, 10.0, 0.95)])
@pytest_cases.parametrize_with_cases(
    "solution_tornadox, solution_odefilter", cases=".", prefix="case_solver_pair_"
)
def test_outputs_equal(solution_tornadox, solution_odefilter):
    for sol_tornadox, sol_odefilter in zip(solution_tornadox, solution_odefilter):
        assert jnp.allclose(sol_tornadox, sol_odefilter)
