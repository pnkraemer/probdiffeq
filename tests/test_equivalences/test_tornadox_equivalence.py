"""Make sure that the results are equivalent to tornadox."""

import jax
import jax.numpy as jnp
import pytest
import pytest_cases
from diffeqzoo import ivps
from tornadox import ek0, ek1, init, ivp, step

from odefilter import controls, ivpsolve, recipes


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

    ekf0, info_op = recipes.ekf0_isotropic_dynamic(num_derivatives=num)
    controller = controls.ClippedIntegral(
        safety=safety, factor_min=factor_min, factor_max=factor_max
    )
    solution_odefilter = ivpsolve.solve(
        vf_ode,
        initial_values=(u0,),
        t0=t0,
        t1=t1,
        solver=ekf0,
        info_op=info_op,
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
def case_solver_pair_dynamic_ekf1(num, atol, rtol, factor_min, factor_max, safety):

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

    ekf1, info_op = recipes.ekf1_dynamic(ode_dimension=2, num_derivatives=num)
    controller = controls.ClippedIntegral(
        safety=safety, factor_min=factor_min, factor_max=factor_max
    )
    solution_odefilter = ivpsolve.solve(
        vf_ode,
        initial_values=(u0,),
        t0=t0,
        t1=t1,
        solver=ekf1,
        info_op=info_op,
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
